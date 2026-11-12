"""Вспомогательная задача по обычным кликам и общая парная проверка моделей"""


import argparse
import gc
import json
import time

import lightgbm as lgb

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

from next_click_eval import OUT, RECENT, prepare, evaluation_data, metrics

from next_click_gru import (ROOT, DEVICE, ProductGRU, encode_sequence, normalization,
                            concatenate, take, predict, fit, load_model, write_submission)
from next_click_behavior import load_behavior
from next_click_gru_finish import legacy_table, fit_legacy, predict_legacy

ALPHA = .1
BATCH = 256
SOURCE_EPOCHS = 21


def action_prefix(event, product, stamp, index):
    """Обрезать пять событий перед кликом, целевой продукт возвращается отдельно"""

    assert index >= 25 and 1 <= product[index] <= 69

    stop = index - 5
    c, t = encode_sequence(event[:stop], product[:stop], stamp[:stop], reference_time=stamp[stop - 1])

    return c, t, int(product[index] - 1)


def build_cache():
    path = OUT / 'train_auxiliary_v1.npz'

    if path.exists():
        return

    dictionary = pd.read_parquet(ROOT / 'data/event_dictionary.parquet')
    clicks = dictionary.loc[(dictionary.event_type == 'click') & ~dictionary.is_target_event,
                            'encoded_event'].to_numpy()
    categorical, timing, labels, metadata = [], [], [], []

    with np.load(OUT / 'train_extended.npz') as source:
        expected = source['full_m']

    count = 0

    for batch in pq.ParquetFile(ROOT / 'data/train.parquet').iter_batches(batch_size=64):
        arr = batch.column(batch.schema.get_field_index('events'))
        off, flat = arr.offsets.to_numpy(), arr.values
        ev = flat.field('encoded_event').to_numpy()
        pr = flat.field('encoded_product').fill_null(-1).to_numpy()
        ts = flat.field('timestamp').to_numpy()
        ids = batch.column(batch.schema.get_field_index('client_id')).to_numpy()

        for j, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
            e, p, t = ev[lo:hi], pr[lo:hi], ts[lo:hi]

            assert ids[j] == expected[count, 0] and t[-1] == expected[count, 2]

            observed = np.flatnonzero(np.isin(e, clicks) & (p >= 1) & (p <= 69))

            for i in observed[observed >= 25]:
                c, z, label = action_prefix(e, p, t, int(i))
                categorical.append(c)
                timing.append(z)
                labels.append(label)
                metadata.append((ids[j], t[i - 6], t[i], int(i)))

            count += 1

        if count % 20000 < 64:
            print('AUXILIARY_CACHE_PROGRESS', count, len(labels), flush=True)

    assert count == len(expected) == 100000

    m = np.asarray(metadata, dtype=np.int64)

    assert np.all(m[:, 1] <= m[:, 2]) and np.all(m[:, 2] < RECENT)

    np.savez_compressed(path, c=np.asarray(categorical), t=np.asarray(timing),
                        y=np.asarray(labels, dtype=np.int64), meta=m)
    print('CAUSAL_AUXILIARY_CACHE_OK', len(labels), len(np.unique(m[:, 0])), flush=True)


def auxiliary_logits(model, head, c, timing):
    """Общий encoder основной модели предсказывает продукт обычного клика."""
    embeddings = torch.cat([model.event(c[:, :, 0]), model.product(c[:, :, 1]), timing], -1)
    _, h = model.gru(model.dropout(embeddings))

    return head(model.dropout(h[-1]))


def device_batch(data, indices, keys):
    return [torch.as_tensor(data[k][indices], device=DEVICE,
                            dtype=torch.long if k in ['c', 'y'] else torch.float32) for k in keys]


def source_training():
    full, history = load_behavior('train'), load_behavior('train', history=True)
    train = concatenate(full, history)
    del full, history
    gc.collect()

    with np.load(OUT / 'train_auxiliary_v1.npz') as cache:
        auxiliary = {k: cache[k] for k in ['c', 't', 'y']}
        auxiliary_meta = cache['meta']

    assert set(auxiliary_meta[:, 0]) <= set(train['meta'].client_id)
    assert np.all(auxiliary_meta[:, 2] < RECENT)

    stats = normalization(train)
    monitor = take(train, np.arange(1024))

    for tag, alpha in [('control', 0.0), ('joint', ALPHA)]:
        torch.manual_seed(42)
        np.random.seed(42)
        torch.set_num_threads(4)
        model = ProductGRU(feature_count=110, product_count=len(train['products']), attention=True)

        for key, value in zip(['x_mean', 'x_std', 'context_mean', 'context_std'], stats):
            getattr(model, key).copy_(torch.from_numpy(value))

        model.product_codes.copy_(torch.from_numpy(train['products'] + 2))
        head = nn.Linear(model.gru.hidden_size, 69).to(DEVICE)
        model.to(DEVICE)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if tag == 'control':
            paired_initial = initial
        else:
            assert all(torch.equal(initial[k], paired_initial[k]) for k in initial)

        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=.001, weight_decay=.01)
        order_rng, aux_rng = np.random.default_rng(2026), np.random.default_rng(2027)
        logs = []
        print('AUXILIARY_SOURCE_START', tag, 'main', len(train['y']), 'auxiliary', len(auxiliary['y']),
              'batch', BATCH, 'alpha', alpha, flush=True)

        for epoch in range(1, SOURCE_EPOCHS + 1):
            start = time.monotonic()
            model.train()
            head.train()
            order = order_rng.permutation(len(train['y']))
            extra = aux_rng.integers(0, len(auxiliary['y']), size=len(order))
            total_main = total_aux = 0.0

            for begin in range(0, len(order), BATCH):
                rows, aux_rows = order[begin:begin + BATCH], extra[begin:begin + BATCH]
                batch = device_batch(train, rows, ['c', 't', 'x', 'context', 'y'])
                aux = device_batch(auxiliary, aux_rows, ['c', 't', 'y'])
                main_loss = nn.functional.cross_entropy(model(*batch[:-1]), batch[-1])
                # Both runs execute this path, preserving dropout calls and sampled auxiliary rows.
                aux_loss = nn.functional.cross_entropy(auxiliary_logits(model, head, *aux[:-1]), aux[-1])
                loss = main_loss + alpha * aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5)
                optimizer.step()
                total_main += main_loss.item() * len(rows)
                total_aux += aux_loss.item() * len(rows)

            p = predict(model, monitor)
            row = dict(epoch=epoch, main_loss=total_main / len(order), auxiliary_loss=total_aux / len(order),
                       alpha=alpha, accuracy=float((p.argmax(1) == monitor['y']).mean()),
                       seconds=round(time.monotonic() - start, 2), metric_role='training_monitor')
            logs.append(row)
            pd.DataFrame(logs).to_csv(OUT / f'aux_v1_{tag}_source_training.csv', index=False)
            print('AUXILIARY_SOURCE_EPOCH', tag, row, flush=True)

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(dict(state=state, auxiliary_state={k: v.detach().cpu() for k, v in head.state_dict().items()},
                        sequence=True, identity=True, epoch=SOURCE_EPOCHS, accuracy=logs[-1]['accuracy'],
                        product_count=len(train['products']), attention=True, hidden=64,
                        seed=42, feature_count=110, auxiliary_alpha=alpha), OUT / f'aux_v1_{tag}_source.pt')
        del model, head, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(source_prefix='aux_v1', report_prefix='auxiliary', loader=load_behavior, control_features=0,
             model_loader=load_model):
    """Адаптировать control/joint на одинаковых шести фолдах и сохранить компоненты.

    Внешние ответы не выбирают эпоху: все модели получают 24 эпохи. Если задан
    control_features, последние признаки контроля обнуляются. Историческое
    сравнение здесь идёт с Behavior; более поздний View Codes оценивается отдельно.
    """
    meta, products, splits = prepare()
    _, _, indices, fingerprint = evaluation_data()
    data = take(loader('test', history=True), indices)

    assert np.array_equal(data['meta'].to_numpy(), meta)

    table, table_products = legacy_table()
    table = table[indices]

    assert np.array_equal(products, table_products)

    booster_base = lgb.Booster(model_file=str(ROOT.parent / 'artifacts/candidate_full.txt'))
    sources = {tag: model_loader(f'{source_prefix}_{tag}_source')[0] for tag in ['control', 'joint']}

    with np.load(OUT / 'behavior_fixed_recent_oof.npz') as ref:
        assert np.array_equal(ref['labels'], data['y']) and np.array_equal(ref['clients'], meta[:, 0])

        previous_group = ref['behavior']

    with np.load(OUT / 'validation_forward_predictions_v1.npz') as ref:
        forward_rows = ref['recent_indices']

        assert np.array_equal(ref['meta'], meta[forward_rows])

        previous_forward = ref['behavior']

    outputs = {phase: {tag: np.full((len(meta), len(products)), np.nan) for tag in sources}
               for phase in ['client', 'forward']}
    neural_outputs = {phase: {tag: np.full((len(meta), len(products)), np.nan) for tag in sources}
                      for phase in ['client', 'forward']}
    booster_outputs = {phase: np.full((len(meta), len(products)), np.nan) for phase in ['client', 'forward']}
    rows = []

    for split in splits:
        train_indices, valid_indices, name = split['train'], split['valid'], split['name']

        assert not set(meta[train_indices, 0]) & set(meta[valid_indices, 0])

        if split['cutoff'] is None:
            phase = 'client'
            booster = fit_legacy(booster_base, table[train_indices], data['y'][train_indices], len(products))
        else:
            phase = 'forward'

            assert meta[train_indices, 3].max() < split['cutoff'] <= meta[valid_indices, 2].min()

            booster = lgb.Booster(model_file=str(OUT / f'validation_v1_{name}_booster.txt'))

        booster_probabilities = predict_legacy(booster, table[valid_indices])
        booster_outputs[phase][valid_indices] = booster_probabilities

        for tag, base in sources.items():
            training = take(data, train_indices)
            validation = take(data, valid_indices)

            if tag == 'control' and control_features:
                training['x'][:, :, -control_features:] = 0
                validation['x'][:, :, -control_features:] = 0

            monitor = take(training, np.arange(min(128, len(train_indices))))
            model, epoch = fit(training, monitor, f'{source_prefix}_{tag}_{name}', base=base, epochs=24,
                               lr=.0003, seed=42, fixed=True, monitor_role='training_monitor; no outer stopping')

            assert epoch == 24

            neural_predictions = predict(model, validation)
            neural_outputs[phase][tag][valid_indices] = neural_predictions
            outputs[phase][tag][valid_indices] = .75 * neural_predictions + .25 * booster_probabilities
            del model, training, validation, monitor
            gc.collect()
            torch.cuda.empty_cache()

        for tag in sources:
            score = dict(phase=phase, scope=name, model=tag,
                         **metrics(outputs[phase][tag][valid_indices], data['y'][valid_indices], meta[valid_indices, 0], outputs[phase]['control'][valid_indices]))
            rows.append(score)
            print('AUXILIARY_FOLD', score, flush=True)

    for phase, selected, previous in [('client', np.arange(len(meta)), previous_group),
                                     ('forward', forward_rows, previous_forward)]:
        selections = [('all', np.ones(len(selected), dtype=bool))]

        if phase == 'client':
            selections += [('prefix_from_April7', meta[selected, 2] >= RECENT)]

        for scope, mask in selections:
            valid_indices = selected[mask]
            reference_predictions = previous[mask]
            control, joint = outputs[phase]['control'][valid_indices], outputs[phase]['joint'][valid_indices]
            candidates = dict(previous=reference_predictions, control=control, joint=joint,
                              control_half=.5 * reference_predictions + .5 * control, joint_half=.5 * reference_predictions + .5 * joint)

            for tag, probabilities in candidates.items():
                row = dict(phase=phase, scope=scope, model=tag,
                           **metrics(probabilities, data['y'][valid_indices], meta[valid_indices, 0], reference_predictions))

                if tag in ['joint', 'joint_half']:
                    paired = metrics(probabilities, data['y'][valid_indices], meta[valid_indices, 0], candidates['control' if tag == 'joint' else 'control_half'])
                    row.update(vs_control_delta=paired['paired_delta'], vs_control_ci95=paired['paired_ci95'])

                rows.append(row)

    components = []

    for phase, selected in [('client', np.arange(len(meta))), ('forward', forward_rows)]:
        selections = [('all', selected)]

        if phase == 'client':
            selections.append(('prefix_from_April7', selected[meta[selected, 2] >= RECENT]))

        for scope, valid_indices in selections:
            for tag in sources:
                components.append(dict(phase=phase, scope=scope, model=f'neural_{tag}',
                    **metrics(neural_outputs[phase][tag][valid_indices], data['y'][valid_indices], meta[valid_indices, 0],
                              neural_outputs[phase]['control'][valid_indices])))

            components.append(dict(phase=phase, scope=scope, model='booster',
                **metrics(booster_outputs[phase][valid_indices], data['y'][valid_indices], meta[valid_indices, 0])))

    pd.DataFrame(rows).to_csv(OUT / f'{report_prefix}_validation_v1.csv', index=False)
    pd.DataFrame(components).to_csv(OUT / f'{report_prefix}_components_v1.csv', index=False)
    np.savez_compressed(OUT / f'{report_prefix}_oof_v1.npz', control=outputs['client']['control'],
                        joint=outputs['client']['joint'], forward_control=outputs['forward']['control'][forward_rows],
                        forward_joint=outputs['forward']['joint'][forward_rows], forward_rows=forward_rows,
                        previous=previous_group, previous_forward=previous_forward,
                        neural_control=neural_outputs['client']['control'], neural_joint=neural_outputs['client']['joint'],
                        forward_neural_control=neural_outputs['forward']['control'][forward_rows],
                        forward_neural_joint=neural_outputs['forward']['joint'][forward_rows],
                        booster=booster_outputs['client'], forward_booster=booster_outputs['forward'][forward_rows],
                        labels=data['y'], clients=meta[:, 0], products=products, metadata_sha256=np.array(fingerprint))
    report = dict(results=rows, metadata_sha256=fingerprint, role='Previously inspected historical labels',
                  components=components, reference_name='Behavior; this legacy reference is not the later View Codes best',
                  mixture_weights=dict(neural=.75, booster=.25),
                  selection='Compare joint against matched control AND the stored Behavior reference on both primary scopes')
    (OUT / f'{report_prefix}_report_v1.json').write_text(json.dumps(report, indent=2))
    print('AUXILIARY_EVALUATION_COMPLETE', flush=True)

    print(pd.DataFrame(rows)[['phase', 'scope', 'model', 'examples', 'correct', 'accuracy', 'paired_delta']].to_string(index=False), flush=True)


def finalize(source_prefix='aux_v1', report_prefix='auxiliary', exploratory_recipe=None, loader=load_behavior):
    """Экспортировать прошедший исторический фильтр рецепт или явно указанный опыт."""
    report = json.loads((OUT / f'{report_prefix}_report_v1.json').read_text())
    primary = {phase: {r['model']: r for r in report['results'] if r['phase'] == phase and r['scope'] == scope}
               for phase, scope in [('client', 'prefix_from_April7'), ('forward', 'all')]}
    choice = None

    for tag in ['joint', 'joint_half']:
        if all(primary[phase][tag]['paired_delta'] > 0 and primary[phase][tag]['vs_control_delta'] > 0
               for phase in primary):
            choice = tag
            break

    report['automatic_choice'] = choice

    if exploratory_recipe is not None:
        assert exploratory_recipe in ['joint', 'joint_half']
        assert all(primary[phase][exploratory_recipe]['paired_delta'] > 0 and
                   primary[phase][exploratory_recipe]['vs_control_delta'] >= 0 for phase in primary)

        choice = exploratory_recipe
        report['export_reason'] = 'User-requested exploratory candidate; positive vs Behavior, nonnegative vs control; strict automatic filter unchanged'

    report['prepared_candidate'] = choice
    (OUT / f'{report_prefix}_report_v1.json').write_text(json.dumps(report, indent=2))

    if choice is None:
        print('AUXILIARY_NO_SUBMISSION: no consistent gain over current and matched controls', flush=True)
        return

    _, _, indices, _ = evaluation_data()
    history = take(loader('test', history=True), indices)
    source, _ = load_model(f'{source_prefix}_joint_source')
    model, _ = fit(history, take(history, np.arange(128)), f'{source_prefix}_joint_adapt_full', base=source,
                   epochs=24, lr=.0003, seed=42, fixed=True, monitor_role='training_monitor')
    test = loader('test')
    table, products = legacy_table(history=False)

    assert np.array_equal(products, test['products'])

    booster = lgb.Booster(model_file=str(OUT / 'behavior_shared_booster.txt'))
    p = .75 * predict(model, test) + .25 * predict_legacy(booster, table)

    if choice == 'joint_half':
        with np.load(OUT / 'behavior_fixed_test.npz') as old:
            assert np.array_equal(old['products'], products)

            p = .5 * p + .5 * old['feature_only']

    np.savez_compressed(OUT / f'{report_prefix}_test_v1.npz', predictions=p, products=products)
    write_submission(p, products, f'submission_{report_prefix}.csv')
    print('AUXILIARY_CANDIDATE_PREPARED', choice, flush=True)


def self_check():
    event, product, stamp = np.arange(32), np.arange(32) % 69 + 1, np.arange(32) * 1000
    c, t, y = action_prefix(event, product, stamp, 31)
    changed_e, changed_p, changed_t = event.copy(), product.copy(), stamp.copy()
    changed_e[26:], changed_p[26:], changed_t[26:] = 999, 69, 999999
    c2, t2, _ = action_prefix(changed_e, changed_p, changed_t, 31)

    assert np.array_equal(c, c2) and np.array_equal(t, t2)
    assert y == product[31] - 1 and t[-1, 1] == 0 and c[-1, 0] == event[25] + 1

    model = ProductGRU(feature_count=110, product_count=42, attention=True)
    head = nn.Linear(64, 69)
    loss = nn.functional.cross_entropy(auxiliary_logits(model, head, torch.from_numpy(c[None].astype(np.int64)),
                                                        torch.from_numpy(t[None].astype(np.float32))), torch.tensor([y]))
    loss.backward()

    assert model.gru.weight_ih_l0.grad.norm() > 0 and model.product.weight.grad.norm() > 0
    assert head.weight.grad.norm() > 0

    print('CAUSAL_AUXILIARY_SELF_CHECK_OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--stage', choices=['cache', 'source', 'evaluate', 'finalize', 'all'], default='all')
    args = parser.parse_args()
    self_check()

    if not args.check:
        protocol = dict(alpha=ALPHA, source_epochs=SOURCE_EPOCHS, source_lr=.001,
                        adaptation_epochs=24, adaptation_lr=.0003, seed=42, main_batch=BATCH,
                        main_loss='CE42', auxiliary_loss='CE69 on ordinary clicks six events after causal cut',
                        data='official train only for auxiliary task; own prefix time reference; no auxiliary aggregates',
                        paired_control='same auxiliary head, batches, dropout calls and initial weights; alpha 0 vs 0.1',
                        inference='original ProductGRU; auxiliary head discarded',
                        mixtures='raw joint recipe and fixed 50/50 with previous Behavior, each with matched control',
                        storage='CPU source arrays; GPU minibatches; all new caches and weights on D:')
        path = OUT / 'auxiliary_protocol_v1.json'

        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('Frozen auxiliary recipe changed')

        path.write_text(json.dumps(protocol, indent=2))

        if args.stage in ['cache', 'all']:
            build_cache()

        if args.stage in ['source', 'all']:
            source_training()

        if args.stage in ['evaluate', 'all']:
            evaluate()

        if args.stage in ['finalize', 'all']:
            finalize()

"""Четыре признака по типам просмотра, которые вошли в лучшее решение"""

import argparse
import gc
import hashlib
import json

import numpy as np

import torch

from next_click_gru import ROOT, OUT, concatenate, fit, load_model, take
from next_click_behavior import load_behavior
from next_click_auxiliary import evaluate, finalize

VIEW_CODES = (399, 666, 162)
NAMES = ('view_399_count', 'view_399_age_log_seconds', 'view_666_count', 'view_162_count')


def features(categorical, timing, products):
    """Посчитать просмотры 399/666/162 и давность последнего 399 по каждому кандидату.

    categorical и timing уже содержат видимый хвост истории в кодировке GRU.
    products должен быть отсортирован; результат имеет форму [клиенты, продукты, 4].
    """
    result = np.zeros((len(categorical), len(products), len(NAMES)), dtype=np.float16)

    for code_index, event_code in enumerate(VIEW_CODES):
        example_indices, event_positions = np.nonzero(categorical[:, :, 0] == event_code + 1)
        candidate_indices = np.searchsorted(products, categorical[example_indices, event_positions, 1].astype(np.int32) - 2)
        valid_candidates = candidate_indices < len(products)
        valid_candidates[valid_candidates] &= products[candidate_indices[valid_candidates]] == categorical[example_indices[valid_candidates], event_positions[valid_candidates], 1] - 2
        example_indices, event_positions, candidate_indices = example_indices[valid_candidates], event_positions[valid_candidates], candidate_indices[valid_candidates]
        event_counts = np.zeros(result.shape[:2], np.int32)

        np.add.at(event_counts, (example_indices, candidate_indices), 1)
        result[:, :, (0, 2, 3)[code_index]] = np.log1p(event_counts)

        if event_code == 399:
            last_positions = np.full(result.shape[:2], -1, np.int32)
            np.maximum.at(last_positions, (example_indices, candidate_indices), event_positions)
            present_rows, product_columns = np.nonzero(last_positions >= 0)
            result[present_rows, product_columns, 1] = timing[present_rows, last_positions[present_rows, product_columns], 1] * 10

    return result


def build_cache():
    import pandas as pd
    dictionary = pd.read_parquet(ROOT/'data/event_dictionary.parquet')

    assert set(VIEW_CODES) <= set(dictionary.loc[dictionary.event_type == 'view', 'encoded_event'])

    for split in ('train', 'test'):
        path = OUT/f'{split}_view_codes_v1.npz'

        if path.exists():
            continue

        with np.load(OUT/f'{split}_extended.npz') as source:
            products = source['products']
            arrays = {scope: features(source[f'{scope}_c'], source[f'{scope}_t'], products)
                      for scope in ['full', 'history']}
            arrays['products'] = products
            arrays['full_m'], arrays['history_m'] = source['full_m'], source['history_m']

        np.savez_compressed(path, **arrays, names=np.array(NAMES))
        print('VIEW_CODES_CACHE', split, len(arrays['full']), len(arrays['history']), flush=True)


def load_codes(split, history=False):
    """Загрузить вход GRU: 82 базовых, 28 поведенческих и 4 View Codes признака."""
    data = load_behavior(split, history)
    scope = 'history' if history else 'full'

    with np.load(OUT/f'{split}_view_codes_v1.npz') as saved:
        assert saved['names'].tolist() == list(NAMES)
        assert np.array_equal(saved['products'], data['products'])
        assert np.array_equal(saved[f'{scope}_m'], data['meta'].to_numpy())

        extra = saved[scope].astype(np.float32)

    assert extra.shape == (len(data['y']), len(data['products']), len(NAMES))

    data['x'] = np.concatenate([data['x'], extra], axis=-1)
    return data


def expanded_base(name='behavior_fixed_source'):
    """Добавить четыре входа с нулевыми весами, сохранив исходный прогноз Behavior."""
    old, epoch = load_model(name)

    assert epoch == 21 and len(old.x_mean) == 110

    model = type(old)(feature_count=110 + len(NAMES), product_count=42, attention=True)
    previous = old.state_dict()
    current = model.state_dict()

    for name, value in previous.items():
        if name in ('head.0.weight', 'x_mean', 'x_std'):
            continue

        current[name].copy_(value)

    current['x_mean'][:110] = previous['x_mean']
    current['x_std'][:110] = previous['x_std']
    current['x_mean'][110:] = 0
    current['x_std'][110:] = 1
    current['head.0.weight'].zero_()
    current['head.0.weight'][:, :190] = previous['head.0.weight'][:, :190]

    current['head.0.weight'][:, 190 + len(NAMES):] = previous['head.0.weight'][:, 190:]
    model.load_state_dict(current)

    return model


def train_sources():
    """Сравнить новые признаки с контролем при одинаковых пяти source-эпохах."""
    full, history = load_codes('train'), load_codes('train', True)
    source = concatenate(full, history)
    del full, history
    gc.collect()
    base = expanded_base()

    for tag in ('control', 'joint'):
        data = source if tag == 'joint' else dict(source, x=source['x'].copy())

        if tag == 'control':
            data['x'][:, :, -len(NAMES):] = 0

        model, epoch = fit(data, take(data, np.arange(128)), f'view_codes_v1_{tag}_source',
                           base=base, epochs=5, lr=.0001, seed=42, fixed=True,
                           monitor_role='fixed additional source training')

        assert epoch == 5

        del model, data
        gc.collect()
        torch.cuda.empty_cache()


def self_check():
    p = np.array([3, 6])
    c = np.zeros((1, 4, 2), np.int16)
    t = np.zeros((1, 4, 2), np.float16)
    c[0, 0] = [400, 5]
    c[0, 1] = [667, 5]
    c[0, 2] = [163, 8]

    t[0, 0, 1] = 1.5
    f = features(c, t, p)

    assert f.shape == (1, 2, 4)
    assert np.allclose(f[0, 0], [np.log(2), 15, np.log(2), 0], atol=.01)
    assert np.allclose(f[0, 1], [0, 0, 0, np.log(2)], atol=.01)

    c[0, 3] = [400, 8]

    assert f[0, 0, 0] == features(c, t, p)[0, 0, 0]

    base = expanded_base().eval()
    original, _ = load_model('behavior_fixed_source')
    original.cpu()
    original.eval()

    with torch.no_grad():
        x = torch.zeros(1, 42, 110)
        context = torch.zeros(1, 11)
        a = original(torch.from_numpy(c.astype(np.int64)), torch.from_numpy(t.astype(np.float32)), x, context)
        b = base(torch.from_numpy(c.astype(np.int64)), torch.from_numpy(t.astype(np.float32)),
                 torch.cat([x, torch.from_numpy(features(c, t, np.arange(42)))], -1), context)

    assert torch.allclose(a, b, atol=1e-6)

    print('VIEW_CODES_SELF_CHECK_OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'cache', 'source', 'evaluate', 'export', 'check'], default='all')
    args = parser.parse_args()
    self_check()
    protocol = dict(view_codes=VIEW_CODES, features=NAMES, base='behavior_fixed_source', extra_epochs=5,
                    lr=.0001, seed=42, adaptation_epochs=24, adaptation_lr=.0003,
                    control='same expanded model and training but new features set to zero',
                    feature_source='last 160 already-visible events; no future views')
    path = OUT/'view_codes_protocol_v1.json'

    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        path.write_text(json.dumps(protocol, indent=2))

    if args.stage in ['all', 'cache']:
        build_cache()

    if args.stage in ['all', 'source']:
        train_sources()

    if args.stage in ['all', 'evaluate']:
        evaluate('view_codes_v1', 'view_codes', loader=load_codes, control_features=len(NAMES))

    if args.stage == 'export':
        finalize('view_codes_v1', 'view_codes', exploratory_recipe='joint', loader=load_codes)

"""New prefix-only behavioral features; fixed-recipe ablations, not a parameter sweep."""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import GroupKFold

from next_click_gru import ROOT, OUT, RECENT, take, concatenate, fit, load_model, predict, write_submission
from next_click_extended import load_extended

NAMES = ["view_sessions", "action_sessions", "target_sessions", "view_days", "action_days", "target_days",
         "view_sessions_last3", "target_sessions_last3", "view_session_trend", "view_repetition_per_session",
         "view_burst_concentration", "target_clicks_per_session", "views_since_last_target",
         "actions_since_last_target", "session_gap_view", "session_gap_action", "session_gap_target",
         "action_age_seconds", "action_distance", "action_exists", "action_event_diversity",
         "target_session_share", "action_recency_rank", "target_frequency_rank", "view_recency_rank",
         "match_last_action_1", "match_last_action_2", "match_last_action_3"]
SOURCE_EPOCHS = 21
ADAPT_EPOCHS = 24
SOURCE_LR = .001
ADAPT_LR = .0003


def behavior_features(e, p, t, products, kinds):
    if len(e) == 0:
        return np.zeros((len(products), len(NAMES)), dtype=np.float32)
    size = max(int(p.max()), int(products.max())) + 1
    pos = np.arange(len(e))
    known = p >= 0
    view = (kinds[e] == 1) & known
    target = (e == 469) & known
    action = (kinds[e] != 1) & known & ~np.isin(e, [99, 687])
    boundary = np.r_[True, np.diff(t) > 1800000] | np.isin(e, [99, 687])
    sessions = np.cumsum(boundary) - 1
    days = t // 86400000

    def count(mask):
        return np.bincount(p[mask], minlength=size)

    def distinct(mask, groups):
        keys = np.unique(groups[mask] * size + p[mask])
        return np.bincount(keys % size, minlength=size)

    def last(mask):
        result = np.full(size, -1, dtype=np.int64)
        np.maximum.at(result, p[mask], pos[mask])
        return result

    def rank(values, higher):
        good = values >= 0
        rank = ((values[None] > values[:, None]) if higher else (values[None] < values[:, None]))
        rank &= good[None]
        result = rank.sum(1) / max(1, good.sum())
        result[~good] = 1
        return result

    views, targets = count(view), count(target)
    vs, acts, tgs = (distinct(mask, sessions) for mask in (view, action, target))
    vd, ad, td = (distinct(mask, days) for mask in (view, action, target))
    recent = sessions >= sessions[-1] - 2
    recent_v, recent_t = distinct(view & recent, sessions), distinct(target & recent, sessions)
    lv, la, lt = last(view), last(action), last(target)
    keys, counts = np.unique(sessions[view] * size + p[view], return_counts=True)
    burst = np.zeros(size)
    np.maximum.at(burst, keys % size, counts)
    after_target = pos > lt[p]
    age = np.zeros(size)
    distance = np.zeros(size)
    exists = la >= 0
    age[exists] = np.maximum(0, (t[-1] - t[la[exists]]) / 1000)
    distance[exists] = len(e) - 1 - la[exists]

    def session_gap(indices):
        gap = np.full(size, sessions[-1] + 1)
        present = indices >= 0
        gap[present] = sessions[-1] - sessions[indices[present]]
        return np.log1p(gap)

    values = [np.log1p(vs), np.log1p(acts), np.log1p(tgs), np.log1p(vd), np.log1p(ad), np.log1p(td),
              np.log1p(recent_v), np.log1p(recent_t),
              recent_v / min(3, sessions[-1] + 1) - vs / (sessions[-1] + 1),
              np.log1p(views / np.maximum(vs, 1)), burst / np.maximum(views, 1),
              np.log1p(targets / np.maximum(tgs, 1)),
              np.log1p(count(view & after_target)), np.log1p(count(action & after_target)),
              session_gap(lv), session_gap(la), session_gap(lt), np.log1p(age), np.log1p(distance),
              exists.astype(float), np.log1p(distinct(action, e)),
              tgs / max(1, len(np.unique(sessions[target])))]
    matrix = [v[products] for v in values]
    matrix += [rank(la[products], True), rank(np.where(targets[products] > 0, targets[products], -1), True),
               rank(lv[products], True)]
    last_products = p[action][::-1][:3].tolist()
    last_products += [-1] * (3 - len(last_products))
    matrix += [(products == product).astype(float) for product in last_products]
    result = np.column_stack(matrix).astype(np.float32)
    assert result.shape == (len(products), len(NAMES)) and np.isfinite(result).all()
    return result


def self_check():
    kinds = np.zeros(1200, dtype=np.int8)
    kinds[1] = 1
    kinds[469] = 2
    e = np.array([1, 1, 469, 1, 2, 1])
    p = np.array([3, 3, 3, 6, 6, 3])
    t = np.array([0, 1000, 2000, 1860000, 1861000, 1920000])
    result = behavior_features(e, p, t, np.array([3, 6, 8]), kinds)
    c = {name: i for i, name in enumerate(NAMES)}
    assert np.isclose(result[0, c["view_sessions"]], np.log(3))
    assert np.isclose(result[0, c["view_burst_concentration"]], 2 / 3)
    assert np.isclose(result[0, c["views_since_last_target"]], np.log(2))
    assert result[1, c["match_last_action_1"]] == 1
    assert result[0, c["match_last_action_2"]] == 1
    assert result[2, c["action_exists"]] == 0
    assert result[2, c["action_recency_rank"]] == 1
    assert behavior_features(e[:0], p[:0], t[:0], np.array([3, 6, 8]), kinds).shape == result.shape
    midnight = behavior_features(np.array([1, 1]), np.array([3, 3]),
                                 np.array([86399500, 86400500]), np.array([3, 6, 8]), kinds)
    assert np.isclose(midnight[0, c["view_sessions"]], np.log(2))
    assert np.isclose(midnight[0, c["view_days"]], np.log(3))
    print("BEHAVIOR_FEATURES_SELF_CHECK_OK", flush=True)


def build_cache():
    self_check()
    dictionary = pd.read_parquet(ROOT / "data/event_dictionary.parquet")
    kinds = np.zeros(1200, dtype=np.int8)
    kinds[dictionary.encoded_event] = np.where(dictionary.event_type == "view", 1, 2)
    for split in ("train", "test"):
        path = OUT / f"{split}_behavior.npz"
        if path.exists():
            continue
        with np.load(OUT / f"{split}_extended.npz") as original:
            products, expected_full, expected_history = original["products"], original["full_m"], original["history_m"]
        full, history, full_m, history_m = [], [], [], []
        n = 0
        for batch in pq.ParquetFile(ROOT / f"data/{split}.parquet").iter_batches(batch_size=64):
            a = batch.column(batch.schema.get_field_index("events"))
            off, flat = a.offsets.to_numpy(), a.values
            ev, pr, ts = flat.field("encoded_event").to_numpy(), flat.field("encoded_product").fill_null(-1).to_numpy(), flat.field("timestamp").to_numpy()
            ids = batch.column(batch.schema.get_field_index("client_id")).to_numpy()
            labels = (batch.column(batch.schema.get_field_index("target_product")).to_numpy()
                      if split == "train" else np.full(len(ids), -1))
            for j, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
                e, p, t = ev[lo:hi], pr[lo:hi], ts[lo:hi]
                full.append(behavior_features(e, p, t, products, kinds).astype(np.float16))
                full_m.append((ids[j], labels[j], t[-1], t[-1]))
                observed = np.flatnonzero((e == 469) & np.isin(p, products))
                for i in observed[observed >= 25][-2:]:
                    if np.any(e[i - 5:i] == 469):
                        continue
                    history.append(behavior_features(e[:i - 5], p[:i - 5], t[:i - 5], products, kinds).astype(np.float16))
                    history_m.append((ids[j], p[i], t[i - 6], t[i]))
            n += len(ids)
            if n % 10000 < 64:
                print("behavior cache", split, n, flush=True)
        assert np.array_equal(np.asarray(full_m), expected_full)
        assert np.array_equal(np.asarray(history_m), expected_history)
        np.savez_compressed(path, full=np.asarray(full), history=np.asarray(history), products=products,
                            names=np.array(NAMES))
        print("BEHAVIOR_CACHE_OK", split, len(full), len(history), flush=True)


def load_behavior(split, history=False):
    data = load_extended(split, history)
    with np.load(OUT / f"{split}_behavior.npz") as extra:
        assert np.array_equal(extra["products"], data["products"])
        assert extra["names"].tolist() == NAMES
        matrix = extra["history" if history else "full"].astype(np.float32)
    assert matrix.shape[:2] == data["x"].shape[:2] and np.isfinite(matrix).all()
    data["x"] = np.concatenate([data["x"], matrix], axis=-1)
    return data


def temporal(loader, cutoff, end):
    full, history = loader("train"), loader("train", history=True)
    a = np.flatnonzero(full["meta"].end.to_numpy() < cutoff)
    b = np.flatnonzero((full["meta"].end.to_numpy() >= cutoff) & (full["meta"].end.to_numpy() < end))
    h = np.flatnonzero((history["meta"].click_time.to_numpy() < cutoff) &
                       history["meta"].client_id.isin(full["meta"].iloc[a].client_id))
    train, valid = concatenate(take(full, a), take(history, h)), take(full, b)
    assert not set(train["meta"].client_id) & set(valid["meta"].client_id)
    return train, valid


def temporal_ablation():
    protocol = dict(source_epochs=SOURCE_EPOCHS, adaptation_epochs=ADAPT_EPOCHS,
                    source_lr=SOURCE_LR, adaptation_lr=ADAPT_LR, seed=42, hidden=64,
                    attention=True, policy="fixed before new feature evaluation; no early stopping or grid",
                    features=NAMES, public_score=None)
    (OUT / "behavior_protocol.json").write_text(json.dumps(protocol, indent=2))
    scores = []
    for start, stop in [("2026-04-01", "2026-04-04"), ("2026-04-04", "2026-04-07")]:
        cutoff, end = (pd.Timestamp(x).value // 1000000 for x in (start, stop))
        for name, loader in [("baseline", load_extended), ("behavior", load_behavior)]:
            tag = f"{name}_fixed_{start[-2:]}"
            train, valid = temporal(loader, cutoff, end)
            if name == "baseline" and start == "2026-04-04":
                model, epochs = load_model("attention_time")
                assert epochs == SOURCE_EPOCHS
            else:
                model, _ = fit(train, valid, tag, epochs=SOURCE_EPOCHS, fixed=True,
                               attention=True, lr=SOURCE_LR, monitor_role="fixed_recipe_temporal_validation")
            p = predict(model, valid)
            score = float(np.mean(p.argmax(1) == valid["y"]))
            scores.append(dict(period=start + " to " + stop, model=name, examples=len(valid["y"]), accuracy=score))
            np.savez_compressed(OUT / f"{tag}_ablation.npz", predictions=p, labels=valid["y"],
                                clients=valid["meta"].client_id.to_numpy(), products=valid["products"])
            pd.DataFrame(scores).to_csv(OUT / "behavior_temporal_ablation.csv", index=False)
            print("FIXED_TEMPORAL_ABLATION", scores[-1], flush=True)
            del train, valid, model
            gc.collect()
            torch.cuda.empty_cache()
    summarize_temporal()


def summarize_temporal():
    rows = []
    for day, stop in [("01", "04"), ("04", "07")]:
        with np.load(OUT / f"baseline_fixed_{day}_ablation.npz") as data:
            baseline, labels, clients = data["predictions"], data["labels"], data["clients"]
        with np.load(OUT / f"behavior_fixed_{day}_ablation.npz") as data:
            feature = data["predictions"]
            assert np.array_equal(labels, data["labels"]) and np.array_equal(clients, data["clients"])
        for name, prediction in [("baseline", baseline), ("behavior", feature),
                                 ("conservative_GRU_50_50", .5 * baseline + .5 * feature)]:
            rows.append(dict(period=f"2026-04-{day} to 2026-04-{stop}", model=name,
                             examples=len(labels), accuracy=float(np.mean(prediction.argmax(1) == labels))))
    pd.DataFrame(rows).to_csv(OUT / "behavior_temporal_ablation.csv", index=False)


def recent_ablation():
    from next_click_gru_finish import legacy_table, fit_legacy, predict_legacy
    import lightgbm as lgb

    full, history = load_behavior("train"), load_behavior("train", history=True)
    source_train = concatenate(full, history)
    source, _ = fit(source_train, take(full, np.arange(1024)), "behavior_fixed_source",
                    epochs=SOURCE_EPOCHS, fixed=True, attention=True, lr=SOURCE_LR)
    del full, history, source_train
    gc.collect()
    baseline, _ = load_model("attention_source")
    a, b = load_extended("test", history=True), load_behavior("test", history=True)
    indices = np.flatnonzero(a["meta"].click_time.to_numpy() >= RECENT)
    a, b = take(a, indices), take(b, indices)
    assert a["meta"].equals(b["meta"])
    assert np.array_equal(a["y"], b["y"])
    source_booster = lgb.Booster(model_file=str(ROOT.parent / "artifacts/candidate_full.txt"))
    table, products = legacy_table()
    table = table[indices]
    p0, p1, pl = (np.zeros((len(a["y"]), len(products))) for _ in range(3))
    scores = []
    clients = a["meta"].client_id.to_numpy()
    for fold, (train, valid) in enumerate(GroupKFold(3).split(clients, groups=clients)):
        assert not set(clients[train]) & set(clients[valid])
        for name, data, base, out in [("baseline", a, baseline, p0), ("behavior", b, source, p1)]:
            model, _ = fit(take(data, train), take(data, valid), f"{name}_fixed_recent_fold{fold}",
                           epochs=ADAPT_EPOCHS, lr=ADAPT_LR, base=base, fixed=True,
                           monitor_role="fixed_recipe_client_validation")
            out[valid] = predict(model, take(data, valid))
            del model
        booster = fit_legacy(source_booster, table[train], a["y"][train], len(products))
        pl[valid] = predict_legacy(booster, table[valid])
        for name, values in [("baseline", p0), ("behavior", p1)]:
            mixed = .75 * values[valid] + .25 * pl[valid]
            scores.append(dict(fold=fold, model=name, examples=len(valid),
                               accuracy=float(np.mean(mixed.argmax(1) == a["y"][valid]))))
        print("FIXED_RECENT_FOLD", scores[-2:], flush=True)
    old = .75 * p0 + .25 * pl
    new = .75 * p1 + .25 * pl
    conservative = .5 * old + .5 * new
    labels = a["y"]
    paired = pd.DataFrame(dict(client=clients,
                              delta=(conservative.argmax(1) == labels).astype(float) - (old.argmax(1) == labels),
                              count=1)).groupby("client")[["delta", "count"]].sum().to_numpy()
    rng = np.random.default_rng(2026)
    sample = rng.integers(0, len(paired), size=(2000, len(paired)))
    bootstrap = paired[sample, 0].sum(1) / paired[sample, 1].sum(1)
    result = dict(examples=len(labels), clients=len(paired),
                  baseline_accuracy=float(np.mean(old.argmax(1) == labels)),
                  behavior_accuracy=float(np.mean(new.argmax(1) == labels)),
                  conservative_accuracy=float(np.mean(conservative.argmax(1) == labels)),
                  conservative_minus_baseline_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  role="paired fixed-recipe client OOF; data was inspected in earlier experiments",
                  public_score=None)
    (OUT / "behavior_fixed_recent.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(scores).to_csv(OUT / "behavior_fixed_recent_folds.csv", index=False)
    np.savez_compressed(OUT / "behavior_fixed_recent_oof.npz", baseline=old, behavior=new,
                        conservative=conservative, labels=labels, clients=clients, products=products)
    print("FIXED_BEHAVIOR_COMPARISON", result, flush=True)
    # No validation is used to choose the following iteration counts or mixing weights.
    adapted, _ = fit(b, take(b, np.arange(128)), "behavior_fixed_adapt_full",
                     epochs=ADAPT_EPOCHS, lr=ADAPT_LR, base=source, fixed=True)
    test = load_behavior("test")
    new_neural = predict(adapted, test)
    full_booster = fit_legacy(source_booster, table, labels, len(products))
    full_booster.save_model(str(OUT / "behavior_shared_booster.txt"))
    test_table, test_products = legacy_table(history=False)
    assert np.array_equal(products, test_products)
    new_boosted = predict_legacy(full_booster, test_table)
    feature_recipe = .75 * new_neural + .25 * new_boosted
    with np.load(OUT / "best_test.npz") as previous:
        assert np.array_equal(previous["products"], products)
        previous_predictions = previous["predictions"]
    candidate = .5 * feature_recipe + .5 * previous_predictions
    np.savez_compressed(OUT / "behavior_fixed_test.npz", predictions=candidate,
                        feature_only=feature_recipe, products=products)
    write_submission(feature_recipe, products, "submission_behavior_only.csv")
    write_submission(candidate, products, "submission_behavior_conservative.csv")


def paired_interval(candidate, baseline, labels, clients):
    values = pd.DataFrame(dict(client=clients,
                               delta=(candidate.argmax(1) == labels).astype(float) - (baseline.argmax(1) == labels),
                               count=1)).groupby("client")[["delta", "count"]].sum().to_numpy()
    sample = np.random.default_rng(2026).integers(0, len(values), size=(2000, len(values)))
    result = values[sample, 0].sum(1) / values[sample, 1].sum(1)
    return np.quantile(result, [.025, .975]).tolist()


def group_ablation():
    data = load_behavior("test", history=True)
    data = take(data, np.flatnonzero(data["meta"].click_time.to_numpy() >= RECENT))
    clients = data["meta"].client_id.to_numpy()
    full = np.zeros((len(clients), len(data["products"])))
    disabled = np.zeros_like(full)
    for fold, (_, b) in enumerate(GroupKFold(3).split(clients, groups=clients)):
        model, _ = load_model(f"behavior_fixed_recent_fold{fold}")
        valid = take(data, b)
        full[b] = predict(model, valid)
        with np.load(OUT / f"behavior_fixed_recent_fold{fold}_valid.npz") as expected:
            assert np.array_equal(expected["labels"], valid["y"])
            assert np.allclose(expected["predictions"], full[b], atol=2e-6)
        valid["x"][:, :, -len(NAMES):] = model.x_mean[-len(NAMES):].detach().cpu().numpy()
        disabled[b] = predict(model, valid)
        del model
    result = dict(with_new_group=float(np.mean(full.argmax(1) == data["y"])),
                  group_replaced_by_training_means=float(np.mean(disabled.argmax(1) == data["y"])),
                  examples=len(clients), role="diagnostic feature-group removal, not new independent validation")
    (OUT / "behavior_group_ablation.json").write_text(json.dumps(result, indent=2))
    print("BEHAVIOR_GROUP_ABLATION", result, flush=True)


def ensemble_ablation():
    from next_click_gru_finish import legacy_table, fit_legacy, predict_legacy
    import lightgbm as lgb

    protocol = dict(seeds=[42, 2026], source_epochs=SOURCE_EPOCHS, adaptation_epochs=ADAPT_EPOCHS,
                    source_lr=SOURCE_LR, adaptation_lr=ADAPT_LR, feature_seed_weights=[.5, .5],
                    previous_recipe_weight=.5, policy="fixed seed averaging; no best-seed or epoch selection")
    (OUT / "behavior_ensemble_protocol.json").write_text(json.dumps(protocol, indent=2))
    temporal_scores = []
    for start, stop in [("2026-04-01", "2026-04-04"), ("2026-04-04", "2026-04-07")]:
        cut, end = (pd.Timestamp(v).value // 1000000 for v in [start, stop])
        train, valid = temporal(load_behavior, cut, end)
        model, _ = fit(train, valid, "behavior_ensemble_fixed_" + start[-2:], epochs=SOURCE_EPOCHS,
                       lr=SOURCE_LR, seed=2026, attention=True, fixed=True,
                       monitor_role="fixed_two_seed_temporal_validation")
        p2 = predict(model, valid)
        with np.load(OUT / f"baseline_fixed_{start[-2:]}_ablation.npz") as d:
            p0 = d["predictions"]
            assert np.array_equal(d["labels"], valid["y"])
        with np.load(OUT / f"behavior_fixed_{start[-2:]}_ablation.npz") as d:
            p1 = d["predictions"]
            assert np.array_equal(d["labels"], valid["y"])
        for name, p in [("baseline", p0), ("new_features_two_seed", .5 * p1 + .5 * p2),
                        ("conservative_two_seed", .5 * p0 + .25 * p1 + .25 * p2)]:
            temporal_scores.append(dict(period=start + " to " + stop, model=name, examples=len(valid["y"]),
                                        accuracy=float(np.mean(p.argmax(1) == valid["y"]))))
        pd.DataFrame(temporal_scores).to_csv(OUT / "behavior_ensemble_temporal.csv", index=False)
        print("TWO_SEED_TEMPORAL", temporal_scores[-3:], flush=True)
        del train, valid, model
        gc.collect()
        torch.cuda.empty_cache()
    full, history = load_behavior("train"), load_behavior("train", history=True)
    train = concatenate(full, history)
    source, _ = fit(train, take(full, np.arange(1024)), "behavior_ensemble_source", epochs=SOURCE_EPOCHS,
                    lr=SOURCE_LR, seed=2026, attention=True, fixed=True)
    del full, history, train
    gc.collect()
    data = load_behavior("test", history=True)
    indices = np.flatnonzero(data["meta"].click_time.to_numpy() >= RECENT)
    data = take(data, indices)
    labels, clients, products = data["y"], data["meta"].client_id.to_numpy(), data["products"]
    table, _ = legacy_table()
    table = table[indices]
    base_booster = lgb.Booster(model_file=str(ROOT.parent / "artifacts/candidate_full.txt"))
    p2 = np.zeros((len(labels), len(products)))
    for fold, (a, b) in enumerate(GroupKFold(3).split(clients, groups=clients)):
        assert not set(clients[a]) & set(clients[b])
        model, _ = fit(take(data, a), take(data, b), f"behavior_ensemble_recent_fold{fold}",
                       epochs=ADAPT_EPOCHS, lr=ADAPT_LR, seed=2026, base=source, fixed=True,
                       monitor_role="fixed_two_seed_client_validation")
        booster = fit_legacy(base_booster, table[a], labels[a], len(products))
        p2[b] = .75 * predict(model, take(data, b)) + .25 * predict_legacy(booster, table[b])
        del model
    with np.load(OUT / "behavior_fixed_recent_oof.npz") as old:
        assert np.array_equal(old["labels"], labels) and np.array_equal(old["clients"], clients)
        p0, p1 = old["baseline"], old["behavior"]
    averaged = .5 * p1 + .5 * p2
    conservative = .5 * p0 + .5 * averaged
    result = dict(baseline_accuracy=float(np.mean(p0.argmax(1) == labels)),
                  feature_ensemble_accuracy=float(np.mean(averaged.argmax(1) == labels)),
                  conservative_accuracy=float(np.mean(conservative.argmax(1) == labels)),
                  conservative_minus_baseline_ci95=paired_interval(conservative, p0, labels, clients),
                  examples=len(labels), clients=len(np.unique(clients)), public_score=None,
                  role="fixed two-seed client OOF; previously inspected data")
    (OUT / "behavior_ensemble_recent.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / "behavior_ensemble_recent_oof.npz", baseline=p0, feature_ensemble=averaged,
                        conservative=conservative, labels=labels, clients=clients, products=products)
    print("TWO_SEED_FIXED_COMPARISON", result, flush=True)
    final, _ = fit(data, take(data, np.arange(128)), "behavior_ensemble_adapt_full", epochs=ADAPT_EPOCHS,
                   lr=ADAPT_LR, seed=2026, base=source, fixed=True)
    test = load_behavior("test")
    booster = fit_legacy(base_booster, table, labels, len(products))
    booster.save_model(str(OUT / "behavior_ensemble_booster.txt"))
    test_table, _ = legacy_table(history=False)
    new_seed_test = .75 * predict(final, test) + .25 * predict_legacy(booster, test_table)
    with np.load(OUT / "behavior_fixed_test.npz") as old: feature1 = old["feature_only"]
    with np.load(OUT / "best_test.npz") as old: previous = old["predictions"]
    feature_ensemble = .5 * feature1 + .5 * new_seed_test
    candidate = .5 * previous + .5 * feature_ensemble
    np.savez_compressed(OUT / "behavior_ensemble_test.npz", predictions=candidate,
                        feature_only=feature_ensemble, products=products)
    write_submission(feature_ensemble, products, "submission_behavior_ensemble_only.csv")
    write_submission(candidate, products, "submission_behavior_ensemble_conservative.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["check", "cache", "temporal", "recent", "ensemble", "importance"], required=True)
    args = parser.parse_args()
    if args.stage == "check": self_check()
    elif args.stage == "cache": build_cache()
    elif args.stage == "temporal": temporal_ablation()
    elif args.stage == "recent": recent_ablation()
    elif args.stage == "ensemble": ensemble_ablation()
    else: group_ablation()

"""Reproducible candidate baseline and adaptation using observed historical clicks."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold, train_test_split

from next_click_features import CONTEXT_NAMES, PRODUCT_NAMES, extract_features, self_check

FEATURE_NAMES = CONTEXT_NAMES + ["candidate"] + PRODUCT_NAMES
CATEGORICAL = [i for i, name in enumerate(FEATURE_NAMES)
               if (name.startswith("last_") and name != "last_gap_seconds") or name == "candidate"]
PARAMETERS = dict(n_estimators=473, learning_rate=0.05, num_leaves=127,
                  min_child_samples=150, reg_lambda=10, colsample_bytree=0.9,
                  n_jobs=4, random_state=42, verbosity=-1)
ADAPT_PARAMETERS = dict(n_estimators=27, learning_rate=0.04, num_leaves=15,
                        min_child_samples=40, reg_lambda=100, colsample_bytree=0.9,
                        n_jobs=4, random_state=42, verbosity=-1)
RANDOM_PERIOD = pd.Timestamp("2026-04-07").value // 1_000_000
ARTIFACTS = Path("artifacts")


def build_feature_cache():
    """Read batches, retaining at most 2,000 observed events per provided history."""
    self_check()
    ARTIFACTS.mkdir(exist_ok=True)
    dictionary = pd.read_parquet("data/event_dictionary.parquet")
    kinds = dictionary.set_index("encoded_event")["event_type"].to_dict()
    target = set(dictionary.loc[dictionary["is_target_event"], "encoded_event"])
    labels = pq.read_table("data/train.parquet", columns=["target_product"]).to_pandas()
    products = np.sort(labels["target_product"].unique())
    assert len(products) == 32
    for split in ("train", "test"):
        paths = [ARTIFACTS / f"{split}_{kind}_{suffix}"
                 for kind in ("candidate", "masked_history")
                 for suffix in ("features.npz", "meta.parquet")]
        if all(path.exists() for path in paths):
            continue
        contexts, matrices, metadata = [], [], []
        past_contexts, past_matrices, past_metadata = [], [], []
        for batch in pq.ParquetFile(f"data/{split}.parquet").iter_batches(batch_size=128):
            for row in batch.to_pylist():
                events = row.pop("events")
                if not events:
                    raise ValueError("Empty client history")
                context, matrix = extract_features(events, products, kinds, target)
                contexts.append(context)
                matrices.append(matrix)
                row["end"] = events[-1]["timestamp"]
                metadata.append(row)
                observed = [i for i, event in enumerate(events)
                            if i >= 25 and event["encoded_event"] in target
                            and event["encoded_product"] in products]
                # shortcut: two past clicks per client; expand only after separate validation.
                for i in observed[-2:]:
                    if any(event["encoded_event"] in target for event in events[i - 5:i]):
                        continue
                    prefix = events[:i - 5]
                    assert prefix[-1] is events[i - 6]
                    context, matrix = extract_features(prefix, products, kinds, target)
                    past_contexts.append(context)
                    past_matrices.append(matrix)
                    past_metadata.append(dict(client_id=row["client_id"],
                                              target_product=events[i]["encoded_product"],
                                              end=prefix[-1]["timestamp"],
                                              click_time=events[i]["timestamp"]))
            if len(metadata) % 10000 < 128:
                print(split, len(metadata), flush=True)
        for kind, context, matrix, meta in (
            ("candidate", contexts, matrices, metadata),
            ("masked_history", past_contexts, past_matrices, past_metadata),
        ):
            np.savez_compressed(ARTIFACTS / f"{split}_{kind}_features.npz",
                                context=np.asarray(context), product=np.asarray(matrix), products=products)
            pd.DataFrame(meta).to_parquet(ARTIFACTS / f"{split}_{kind}_meta.parquet", index=False)


def load_features(split, history=False):
    kind = "masked_history" if history else "candidate"
    with np.load(ARTIFACTS / f"{split}_{kind}_features.npz") as data:
        context, matrix, products = data["context"], data["product"], data["products"]
    meta = pd.read_parquet(ARTIFACTS / f"{split}_{kind}_meta.parquet")
    assert len(context) == len(matrix) == len(meta)
    assert context.shape[1] == len(CONTEXT_NAMES)
    assert matrix.shape[1:] == (len(products), len(PRODUCT_NAMES))
    return context, matrix, meta, products


def candidate_table(context, matrix, products, labels=None, negatives=None):
    """Sample ten other products during training; score all 32 at inference."""
    n, count, _ = matrix.shape
    if negatives is None:
        candidates = np.broadcast_to(np.arange(count), (n, count))
    else:
        labels = np.asarray(labels)
        assert np.isin(labels, products).all() and negatives < count
        positive = np.searchsorted(products, labels)
        
        noise = np.random.default_rng(42).random((n, count))
        noise[np.arange(n), positive] = 2
        negative = np.argpartition(noise, negatives, axis=1)[:, :negatives]
        candidates = np.column_stack([positive, negative])
    width = candidates.shape[1]
    
    x = np.column_stack([
        np.repeat(context, width, axis=0), products[candidates].reshape(-1),
        matrix[np.arange(n)[:, None], candidates].reshape(n * width, -1),
    ]).astype(np.float32)
    
    for i in CATEGORICAL:
        x[x[:, i] < 0, i] = np.nan
    y = None if labels is None else (products[candidates] == np.asarray(labels)[:, None]).astype(np.int8).ravel()
    if y is not None:
        assert (y.reshape(n, width).sum(axis=1) == 1).all()
    return x, y


def candidate_scores(model, context, matrix, products):
    x, _ = candidate_table(context, matrix, products)
    return model.predict(x, num_threads=4).reshape(-1, len(products))


def train_boosted(context, matrix, meta, products):
    x, y = candidate_table(context, matrix, products, meta["target_product"].to_numpy(), negatives=10)
    model = lgb.LGBMClassifier(**PARAMETERS)
    model.fit(x, y, categorical_feature=CATEGORICAL)
    model.booster_.save_model(str(ARTIFACTS / "candidate_full.txt"))
    return model.booster_


def validate_boosted(context, matrix, meta, products):
    """Keep both random and forward-in-time scores; neither is a Kaggle score."""
    indices = np.arange(len(meta))
    a, b = train_test_split(indices, test_size=0.2, stratify=meta["target_product"], random_state=42)
    cutoff = pd.Timestamp("2026-04-04").value // 1_000_000
    splits = {"random": (a, b), "time_04_04": (np.flatnonzero(meta["end"] < cutoff),
                                             np.flatnonzero(meta["end"] >= cutoff))}
    results = []
    for name, (a, b) in splits.items():
        x, y = candidate_table(context[a], matrix[a], products, meta.iloc[a]["target_product"].to_numpy(), negatives=10)
        model = lgb.LGBMClassifier(**PARAMETERS)
        model.fit(x, y, categorical_feature=CATEGORICAL)
        scores = candidate_scores(model.booster_, context[b], matrix[b], products)
        accuracy = float(np.mean(products[scores.argmax(axis=1)] == meta.iloc[b]["target_product"].to_numpy()))
        results.append(dict(split=name, accuracy=accuracy, examples=len(b)))
        print(results[-1], flush=True)
    return pd.DataFrame(results)


def adaptation_data():
    context, matrix, meta, products = load_features("test", history=True)
    good = meta["click_time"] >= RANDOM_PERIOD
    return context[good], matrix[good], meta[good].reset_index(drop=True), products


def accuracy_metric(labels, predictions):
    labels = labels.reshape(-1, 32)
    predictions = predictions.reshape(-1, 32)
    return "accuracy", float(np.mean(labels.argmax(axis=1) == predictions.argmax(axis=1))), True


def validate_adaptation(base):
    """Hold out clients during adaptation; the base sees only earlier source labels."""
    context, matrix, meta, products = adaptation_data()
    labels = meta["target_product"].to_numpy()
    baseline = candidate_scores(base, context, matrix, products)
    out_of_fold = np.zeros_like(baseline)
    rounds = []
    for fold, (a, b) in enumerate(GroupKFold(3).split(context, labels, meta["client_id"])):
        assert not set(meta.iloc[a]["client_id"]) & set(meta.iloc[b]["client_id"])
        x, y = candidate_table(context[a], matrix[a], products, labels[a])
        valid_x, valid_y = candidate_table(context[b], matrix[b], products, labels[b])
        parameters = dict(ADAPT_PARAMETERS, n_estimators=250, metric="None")
        model = lgb.LGBMClassifier(**parameters)
        model.fit(x, y, init_model=base, categorical_feature=CATEGORICAL,
                  eval_X=valid_x, eval_y=valid_y, eval_metric=accuracy_metric,
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        out_of_fold[b] = model.predict_proba(valid_x)[:, 1].reshape(-1, len(products))
        rounds.append(model.best_iteration_ - base.num_trees())
        print("adaptation fold", fold, "new trees", rounds[-1], flush=True)
    results = pd.DataFrame({"variant": ["source model", "adaptation, client OOF"],
                            "accuracy": [np.mean(products[baseline.argmax(1)] == labels),
                                         np.mean(products[out_of_fold.argmax(1)] == labels)],
                            "examples": [len(meta), len(meta)]})
    return results, int(np.median(rounds))


def train_adapted(base, rounds=27):
    context, matrix, meta, products = adaptation_data()
    x, y = candidate_table(context, matrix, products, meta["target_product"].to_numpy())
    model = lgb.LGBMClassifier(**dict(ADAPT_PARAMETERS, n_estimators=rounds))
    model.fit(x, y, init_model=base, categorical_feature=CATEGORICAL)
    model.booster_.save_model(str(ARTIFACTS / "candidate_adapted_full.txt"))
    return model.booster_


def write_submission(model, filename):
    context, matrix, meta, products = load_features("test")
    scores = candidate_scores(model, context, matrix, products)
    predictions = pd.Series(products[scores.argmax(axis=1)], index=meta["example_id"])
    submission = pd.read_csv("data/sample_submission.csv")
    assert meta["example_id"].is_unique and submission["example_id"].is_unique
    assert set(submission["example_id"]) == set(meta["example_id"])
    submission["encoded_product"] = submission["example_id"].map(predictions)
    assert submission["encoded_product"].notna().all()
    assert submission["encoded_product"].isin(products).all()
    assert list(submission.columns) == ["example_id", "encoded_product"]
    submission.to_csv(filename, index=False)
    return submission


def check_candidate_table():
    products = np.arange(32, dtype=np.int32)
    context = np.zeros((2, len(CONTEXT_NAMES)), dtype=np.float32)
    matrix = np.zeros((2, len(products), len(PRODUCT_NAMES)), dtype=np.float32)
    labels = np.array([0, 31])
    sampled, targets = candidate_table(context, matrix, products, labels, negatives=10)
    assert sampled.shape == (22, len(FEATURE_NAMES))
    assert np.array_equal(targets.reshape(2, 11)[:, 0], [1, 1])
    candidates = sampled[:, len(CONTEXT_NAMES)].reshape(2, 11)
    assert all(len(np.unique(row)) == 11 for row in candidates)
    full, targets = candidate_table(context, matrix, products, labels)
    assert full.shape == (64, len(FEATURE_NAMES))
    assert np.array_equal(full[:32, len(CONTEXT_NAMES)], products)
    assert accuracy_metric(targets, targets.astype(float))[1] == 1.0


if __name__ == "__main__":
    self_check()
    check_candidate_table()
    build_feature_cache()
    context, matrix, meta, products = load_features("train")
    base = train_boosted(context, matrix, meta, products)
    write_submission(base, "submission_boosted.csv")
    adapted = train_adapted(base)
    write_submission(adapted, "submission_adapted.csv")
    print("Created submission_boosted.csv and submission_adapted.csv")

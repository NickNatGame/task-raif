"""Freeze evaluation indices and audit saved OOF; this module never trains a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

OUT = Path(os.environ.get("RAIF_GRU_ARTIFACTS", "/mnt/d/raif-gru/artifacts"))
RECENT = pd.Timestamp("2026-04-07").value // 1_000_000
WINDOWS = [("2026-04-08", "2026-04-10"), ("2026-04-10", "2026-04-12"),
           ("2026-04-12", "2026-04-17")]


def partitions(meta):
    """Indices refer to the recent-example table, not the full historical cache."""
    splits = []
    for i, (train, valid) in enumerate(GroupKFold(3).split(meta, groups=meta[:, 0])):
        splits.append(dict(name=f"client_{i}", train=train, valid=valid, cutoff=None))
    for i, (start, stop) in enumerate(WINDOWS):
        cutoff, end = (pd.Timestamp(x).value // 1_000_000 for x in (start, stop))
        # Availability is determined by label time; prediction time is prefix end.
        valid = np.flatnonzero((meta[:, 2] >= cutoff) & (meta[:, 2] < end))
        train = np.flatnonzero((meta[:, 3] < cutoff) &
                               ~np.isin(meta[:, 0], meta[valid, 0]))
        splits.append(dict(name=f"forward_{i}", train=train, valid=valid, cutoff=int(cutoff)))
    for split in splits:
        train, valid = split["train"], split["valid"]
        assert len(train) and len(valid), f"Empty fold: {split['name']}"
        assert not set(meta[train, 0]) & set(meta[valid, 0])
        if split["cutoff"] is not None:
            assert meta[train, 3].max() < split["cutoff"] <= meta[valid, 2].min()
    return splits


def evaluation_data():
    with np.load(OUT / "test_extended.npz") as source:
        full_meta, products = source["history_m"].astype(np.int64), source["products"]
    indices = np.flatnonzero(full_meta[:, 3] >= RECENT)
    meta = full_meta[indices]
    assert np.all(meta[:, 2] <= meta[:, 3])
    assert np.isin(meta[:, 1], products).all()
    assert len(np.unique(meta[:, [0, 2, 3]], axis=0)) == len(meta)
    digest = hashlib.sha256(meta.tobytes() + indices.tobytes() + products.tobytes()).hexdigest()
    return meta, products, indices, digest


def metrics(probabilities, labels, clients, baseline=None):
    """Score aligned, held-out rows; the caller must obey the frozen training splits."""
    p, labels, clients = np.asarray(probabilities), np.asarray(labels), np.asarray(clients)
    if (p.ndim != 2 or labels.shape != (len(p),) or clients.shape != labels.shape
            or not len(p) or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= p.shape[1])
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=1e-4, rtol=0)):
        raise ValueError("Expected finite normalized probabilities, aligned clients and class indices")
    correct = p.argmax(1) == labels
    top = np.argsort(-p, axis=1)[:, :min(3, p.shape[1])]
    result = dict(examples=len(p), clients=len(np.unique(clients)), correct=int(correct.sum()),
                  accuracy=float(correct.mean()), top3=float((top == labels[:, None]).any(1).mean()),
                  log_loss=float(-np.log(np.clip(p[np.arange(len(p)), labels], 1e-15, 1)).mean()))
    _, group = np.unique(clients, return_inverse=True)
    counts = np.bincount(group)
    sampled = np.random.default_rng(2026).integers(0, len(counts), (2000, len(counts)))
    accuracy_bootstrap = np.bincount(group, weights=correct)[sampled].sum(1) / counts[sampled].sum(1)
    result["accuracy_ci95"] = np.quantile(accuracy_bootstrap, [.025, .975]).tolist()
    if baseline is not None:
        metrics(baseline, labels, clients)
        if np.asarray(baseline).shape != p.shape:
            raise ValueError("Baseline and candidate must have the same candidate columns")
        delta = correct.astype(float) - (np.asarray(baseline).argmax(1) == labels)
        totals = np.bincount(group, weights=delta)
        difference = totals[sampled].sum(1) / counts[sampled].sum(1)
        result.update(paired_delta=float(delta.mean()),
                      paired_ci95=np.quantile(difference, [.025, .975]).tolist())
    return result


def prepare():
    meta, products, indices, digest = evaluation_data()
    splits = partitions(meta)
    protocol = dict(version=1, metadata_sha256=digest, examples=len(meta),
                    clients=len(np.unique(meta[:, 0])), products=products.tolist(),
                    existing_data_role="Previously inspected historical labels; not a new blind holdout",
                    source="Official train and its past prefixes; no post-April-7 labels",
                    auxiliary_rule="Auxiliary labels, pseudo-labels, preprocessing and retrieval indices use training only; exclude validation clients from adaptation",
                    stopping="Fix recipe before outer scoring; any early stopping uses inner training data only",
                    forward_rule="Train label_time < cutoff <= validation prefix_end; purge validation clients from adaptation",
                    source_client_rule="Earlier official train may contain the same clients, matching the real train/test overlap; adaptation groups remain disjoint",
                    limitations=["The six-event horizon is empirical, not verified by the organizer",
                                 "Synthetic historical examples may differ from final test cuts",
                                 "Late forward folds are small; none is an independent final test",
                                 "Old client-OOF predictions cannot be relabeled as forward predictions"],
                    folds=[dict(name=s["name"], train=len(s["train"]), valid=len(s["valid"]),
                                cutoff=s["cutoff"],
                                role="same-policy grouped diagnostic" if s["cutoff"] is None
                                else "causal forward diagnostic; requires newly fitted models") for s in splits])
    path = OUT / "evaluation_protocol_v1.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen evaluation changed; keep version 1 and explicitly define a new protocol")
    path.write_text(json.dumps(protocol, indent=2))
    arrays = dict(meta=meta, products=products, history_indices=indices,
                  row_id=np.array([f"history_{i}" for i in indices]), metadata_sha256=np.array(digest))
    for split in splits:
        arrays[split["name"] + "_train"] = split["train"]
        arrays[split["name"] + "_valid"] = split["valid"]
    np.savez_compressed(OUT / "evaluation_indices_v1.npz", **arrays)
    frame = pd.DataFrame(meta, columns=["client_id", "target_product", "prefix_end", "label_time"])
    frame.insert(0, "history_index", indices)
    for split in splits:
        frame[split["name"]] = "excluded"
        frame.loc[split["train"], split["name"]] = "train"
        frame.loc[split["valid"], split["name"]] = "valid"
    frame.to_csv(OUT / "evaluation_rows_v1.csv", index=False)
    print("FROZEN_EVALUATION", protocol["folds"], flush=True)
    return meta, products, splits


def audit_saved_oof(meta, products):
    """Diagnostics only: these models were not fitted with the new forward folds."""
    with np.load(OUT / "behavior_fixed_recent_oof.npz") as source:
        assert np.array_equal(source["clients"], meta[:, 0])
        assert np.array_equal(source["labels"], np.searchsorted(products, meta[:, 1]))
        assert np.array_equal(source["products"], products)
        predictions = {name: source[name] for name in ["baseline", "behavior", "conservative"]}
    labels, clients = np.searchsorted(products, meta[:, 1]), meta[:, 0]
    reports = []
    days = pd.to_datetime(meta[:, 2], unit="ms").strftime("%Y-%m-%d").to_numpy()
    slices = [("all", np.ones(len(meta), dtype=bool)),
              ("prefix_before_policy_change", meta[:, 2] < RECENT),
              ("prefix_April8_onwards", meta[:, 2] >= pd.Timestamp("2026-04-08").value // 1_000_000)]
    slices += [(day, days == day) for day in sorted(set(days))]
    for name, p in predictions.items():
        for label, mask in slices:
            if not mask.any():
                continue
            result = metrics(p[mask], labels[mask], clients[mask], predictions["baseline"][mask])
            reports.append(dict(model=name, slice=label, role="saved grouped OOF; NOT forward evaluation", **result))
    pd.DataFrame(reports).to_csv(OUT / "evaluation_saved_oof_diagnostics.csv", index=False)
    forward_accuracy = None
    reason = "Forward fits have not been run for this protocol"
    report_path = OUT / "validation_report_v1.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['protocol']['evaluation_metadata_sha256'] != evaluation_data()[3]:
            raise ValueError("Forward results do not match the frozen evaluation data")
        forward_accuracy = {r['model']: r['accuracy'] for r in report['results']
                            if r['phase'] == 'forward' and r['scope'] == 'all_159'
                            and r['model'] in ['baseline', 'behavior', 'conservative']}
        reason = "Fixed-recipe forward fits completed; historical labels were inspected previously"
    audit = dict(examples=len(meta), prefixes_before_policy_change=int((meta[:, 2] < RECENT).sum()),
                 label_prefix_gap_seconds_quantiles=np.quantile((meta[:, 3] - meta[:, 2]) / 1000,
                                                                 [0, .5, .9, .99, 1]).tolist(),
                 forward_accuracy=forward_accuracy, reason=reason)
    (OUT / "evaluation_audit_v1.json").write_text(json.dumps(audit, indent=2))
    print("SAVED_OOF_AUDIT", audit, flush=True)


def self_check():
    p, y, clients = np.array([[.8, .2], [.3, .7]]), np.array([0, 1]), np.array([1, 2])
    result = metrics(p, y, clients, p[:, ::-1])
    assert result["accuracy"] == 1 and result["paired_ci95"] == [1, 1]
    for invalid in [p * 2, np.full_like(p, np.nan), -p]:
        try:
            metrics(invalid, y, clients)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid probabilities were accepted")
    def stamp(day):
        return pd.Timestamp(f"2026-04-{day:02d}").value // 1_000_000
    meta = np.array([[1, 0, stamp(7), stamp(7)], [1, 1, stamp(8), stamp(9)],
                     [2, 0, stamp(7), stamp(8)], [3, 1, stamp(9), stamp(9)],
                     [4, 1, stamp(10), stamp(11)], [5, 0, stamp(12), stamp(13)],
                     [6, 0, stamp(7), stamp(7)]])
    first = next(s for s in partitions(meta) if s["name"] == "forward_0")
    assert 0 not in first["train"]  # Same client as validation.
    assert 2 not in first["train"]  # Label not yet available at cutoff.
    assert 2 not in first["valid"]  # Prediction prefix belongs to the earlier period.
    print("EVALUATION_SELF_CHECK_OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    self_check()
    if not args.check:
        meta, products, _ = prepare()
        audit_saved_oof(meta, products)

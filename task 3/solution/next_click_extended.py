"""Кэши для всех известных целевых продуктов, включая классы только из прошлых кликов"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from next_click_features import extract_features, PRODUCT_NAMES

from next_click_gru import OUT, ROOT, CUTOFF, encode_sequence, take, concatenate, fit


def build_extended():
    """Собрать полные входы и обрезанные исторические примеры для общего каталога"""
    audit = json.loads((OUT / "product_audit.json").read_text())
    labels = pq.read_table(ROOT / "data/train.parquet", columns=["target_product"]).column(0).to_numpy()
    products = np.array(sorted(set(labels) | set(audit["train"]["target_click_products"]) |
                               set(audit["test"]["target_click_products"])), dtype=np.int64)
    dictionary = pd.read_parquet(ROOT / "data/event_dictionary.parquet")
    kinds = dictionary.set_index("encoded_event")["event_type"].to_dict()

    targets = set(dictionary.loc[dictionary.is_target_event, "encoded_event"])

    for split in ("train", "test"):
        path = OUT / f"{split}_extended.npz"

        if path.exists():
            continue

        data = {prefix + key: [] for prefix in ("full_", "history_")
                for key in ("c", "t", "x", "context", "m")}

        def append(prefix, events, client, label, click_time):
            ev = np.array([e["encoded_event"] for e in events], dtype=np.int32)
            pr = np.array([e["encoded_product"] if e["encoded_product"] is not None else -1
                           for e in events], dtype=np.int32)
            ts = np.array([e["timestamp"] for e in events], dtype=np.int64)
            c, t = encode_sequence(ev, pr, ts)
            context, matrix = extract_features(events, products, kinds, targets)

            for key, value in (("c", c), ("t", t), ("x", matrix.astype(np.float16)),
                               ("context", context[:11]),
                               ("m", (client, label, ts[-1], click_time))):
                data[prefix + key].append(value)

        count = 0

        for batch in pq.ParquetFile(ROOT / f"data/{split}.parquet").iter_batches(batch_size=32):
            for row in batch.to_pylist():
                events = row["events"]
                append("full_", events, row["client_id"], row.get("target_product", -1), events[-1]["timestamp"])
                observed = [i for i, e in enumerate(events) if i >= 25
                            and e["encoded_event"] in targets and e["encoded_product"] in products]

                for i in observed[-2:]:
                    if any(e["encoded_event"] in targets for e in events[i - 5:i]):
                        continue

                    append("history_", events[:i - 5], row["client_id"],
                           events[i]["encoded_product"], events[i]["timestamp"])

                count += 1

            if count % 10000 < 32:
                print("extended cache", split, count, flush=True)

        np.savez_compressed(path, products=products, **{k: np.asarray(v) for k, v in data.items()})
        print("extended cached", split, "products", len(products),
              "histories", len(data["history_m"]), flush=True)


def load_extended(split, history=False):
    """Загрузить последовательности, агрегаты и metadata в общий формат данных GRU.

    Ключи c/t/x/context/y: категории, время, продуктовые признаки, общий контекст
    и индекс правильного кандидата. meta хранит ID клиента и времена префикса/метки.
    """
    prefix = "history_" if history else "full_"

    with np.load(OUT / f"{split}_extended.npz") as source:
        categorical, timing, product_features, context, metadata_matrix = (source[prefix + key] for key in ("c", "t", "x", "context", "m"))
        products = source["products"]

    metadata = pd.DataFrame(metadata_matrix, columns=["client_id", "target_product", "end", "click_time"])
    product_features = product_features.astype(np.float32)
    product_features = np.concatenate([np.nan_to_num(product_features, nan=0), np.isnan(product_features)], axis=-1)

    return dict(c=categorical, t=timing, x=product_features, context=context, y=np.searchsorted(products, metadata_matrix[:, 1]),
                meta=metadata, products=products)


def temporal_data(loader=load_extended):
    """Подготовить ранний source и официальный временной контроль после CUTOFF."""
    full = loader("train")
    a = np.flatnonzero(full["meta"].end.to_numpy() < CUTOFF)
    b = np.flatnonzero(full["meta"].end.to_numpy() >= CUTOFF)
    history = loader("train", history=True)
    history = take(history, np.flatnonzero((history["meta"].click_time.to_numpy() < CUTOFF) &
                                         history["meta"].client_id.isin(full["meta"].iloc[a].client_id)))

    train = concatenate(take(full, a), history)
    valid = take(full, b)

    assert not set(train["meta"].client_id) & set(valid["meta"].client_id)

    return train, valid


def build_product_sequences(mode="product"):
    dictionary = pd.read_parquet(ROOT / "data/event_dictionary.parquet")
    view_events = dictionary.loc[dictionary.event_type == "view", "encoded_event"].to_numpy()

    for split in ("train", "test"):
        path = OUT / f"{split}_{mode}_sequences.npz"

        if path.exists():
            continue

        with np.load(OUT / f"{split}_extended.npz") as original:
            products = original["products"]
            expected_full, expected_history = original["full_m"], original["history_m"]

        full_c, full_t, history_c, history_t, full_m, history_m = [], [], [], [], [], []

        for batch in pq.ParquetFile(ROOT / f"data/{split}.parquet").iter_batches(batch_size=64):
            arrays = batch.column(batch.schema.get_field_index("events"))
            offsets, flat = arrays.offsets.to_numpy(), arrays.values
            ev = flat.field("encoded_event").to_numpy()
            pr = flat.field("encoded_product").fill_null(-1).to_numpy()
            ts = flat.field("timestamp").to_numpy()
            ids = batch.column(batch.schema.get_field_index("client_id")).to_numpy()

            labels = (batch.column(batch.schema.get_field_index("target_product")).to_numpy()
                      if split == "train" else np.full(len(ids), -1))

            for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
                e, p, t = ev[lo:hi], pr[lo:hi], ts[lo:hi]
                keep = p >= 0 if mode == "product" else ~np.isin(e, view_events)
                c, z = encode_sequence(e[keep], p[keep], t[keep],
                                       reference_time=t[-1] if mode == "intent" else None)
                full_c.append(c)
                full_t.append(z)
                full_m.append((ids[j], labels[j], t[-1], t[-1]))

                observed = np.flatnonzero((e == 469) & np.isin(p, products))

                for i in observed[observed >= 25][-2:]:
                    if np.any(e[i - 5:i] == 469):
                        continue

                    keep = p[:i - 5] >= 0 if mode == "product" else ~np.isin(e[:i - 5], view_events)
                    c, z = encode_sequence(e[:i - 5][keep], p[:i - 5][keep], t[:i - 5][keep],
                                           reference_time=t[i - 6] if mode == "intent" else None)
                    history_c.append(c)
                    history_t.append(z)
                    history_m.append((ids[j], p[i], t[i - 6], t[i]))

        assert np.array_equal(np.asarray(full_m), expected_full)
        assert np.array_equal(np.asarray(history_m), expected_history)

        np.savez_compressed(path, products=products, full_c=np.asarray(full_c), full_t=np.asarray(full_t),
                            history_c=np.asarray(history_c), history_t=np.asarray(history_t))
        print("FILTERED_SEQUENCE_CACHE_OK", mode, split, len(full_c), len(history_c), flush=True)


def load_product_data(split, history=False, mode="product"):
    data = load_extended(split, history)
    prefix = "history_" if history else "full_"

    with np.load(OUT / f"{split}_{mode}_sequences.npz") as cache:
        assert np.array_equal(data["products"], cache["products"])

        data["c"], data["t"] = cache[prefix + "c"], cache[prefix + "t"]

    assert len(data["c"]) == len(data["y"])

    return data


def load_intent_data(split, history=False):
    """Исторический эксперимент: убрать просмотры и оставить признаки действий."""
    data = load_product_data(split, history, mode="intent")
    keep = {"clicks", "targets", "clicks_session", "targets_day", "target_decay_day",
            "click_age_seconds", "target_age_seconds", "match_last_target_1",
            "match_last_target_2", "match_last_target_3"}
    remove = [i for i, name in enumerate(PRODUCT_NAMES) if name not in keep]
    data["x"][:, :, remove + [i + len(PRODUCT_NAMES) for i in remove]] = 0
    data["context"][:, 7] = 0

    return data


def build_exact_test():
    """Сохранить для LightGBM полный категориальный контекст и признаки float32."""
    path = OUT / "test_extended_exact.npz"

    if path.exists():
        return

    with np.load(OUT / "test_extended.npz") as cached:
        products = cached["products"]
        expected = cached["history_m"]

    dictionary = pd.read_parquet(ROOT / "data/event_dictionary.parquet")
    kinds = dictionary.set_index("encoded_event")["event_type"].to_dict()
    targets = set(dictionary.loc[dictionary.is_target_event, "encoded_event"])
    full_context, full_product, past_context, past_product, meta = [], [], [], [], []

    for batch in pq.ParquetFile(ROOT / "data/test.parquet").iter_batches(batch_size=32):
        for row in batch.to_pylist():
            events = row["events"]
            context, matrix = extract_features(events, products, kinds, targets)
            full_context.append(context)
            full_product.append(matrix)
            observed = [i for i, e in enumerate(events) if i >= 25
                        and e["encoded_event"] in targets and e["encoded_product"] in products]

            for i in observed[-2:]:
                if any(e["encoded_event"] in targets for e in events[i - 5:i]):
                    continue

                prefix = events[:i - 5]
                context, matrix = extract_features(prefix, products, kinds, targets)
                past_context.append(context)
                past_product.append(matrix)
                meta.append((row["client_id"], events[i]["encoded_product"],
                             prefix[-1]["timestamp"], events[i]["timestamp"]))

    assert np.array_equal(np.asarray(meta), expected)

    np.savez_compressed(path, products=products, full_context=np.asarray(full_context),
                        full_product=np.asarray(full_product), history_context=np.asarray(past_context),
                        history_product=np.asarray(past_product), history_m=np.asarray(meta))
    print("EXACT_TEST_FEATURES_OK", len(full_context), len(past_context), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exact-test", action="store_true")
    parser.add_argument("--product-cache", action="store_true")
    parser.add_argument("--intent-cache", action="store_true")

    parser.add_argument("--variant", choices=["extended", "extended_mlp", "extended_anonymous", "attention", "attention2", "wide", "products", "intent"], default="extended")
    args = parser.parse_args()

    if args.intent_cache:
        build_product_sequences("intent")
    elif args.product_cache:
        build_product_sequences()
    elif args.exact_test:
        build_exact_test()
    elif not args.cache:
        build_extended()
        loader = load_intent_data if args.variant == "intent" else load_product_data if args.variant == "products" else load_extended
        train, valid = temporal_data(loader)
        fit(train, valid, args.variant + "_time", epochs=25, patience=6,
            sequence=args.variant != "extended_mlp", identity=args.variant != "extended_anonymous",
            attention=args.variant in ("attention", "attention2", "products", "intent"),
            hidden=128 if args.variant == "wide" else 64, seed=2026 if args.variant == "attention2" else 42)
    else:
        build_extended()

"""GRU с вниманием к кандидатам: кодирование истории, обучение и прогноз"""

import argparse
import copy
import json
import os
import time

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
CACHE = ROOT.parents[1] / "artifacts"
OUT = Path(os.environ.get("RAIF_GRU_ARTIFACTS", "/mnt/d/raif-gru/artifacts"))
LENGTH = 160
CUTOFF = pd.Timestamp("2026-04-04").value // 1_000_000

RECENT = pd.Timestamp("2026-04-07").value // 1_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_sequence(event_codes, product_codes, timestamps, length=LENGTH, reference_time=None):
    """Закодировать последние length событий и дополнить короткую историю слева.

    Ноль зарезервирован под padding. Код события сдвигается на 1, ID продукта
    на 2: пропущенный продукт (-1 во входе) превращается в 1. Время измеряется
    в миллисекундах; выход содержит логарифмы паузы и давности в секундах /10.
    """
    event_codes, product_codes, timestamps = event_codes[-length:], product_codes[-length:], timestamps[-length:]
    categorical = np.zeros((length, 2), dtype=np.int16)
    timing = np.zeros((length, 2), dtype=np.float16)
    event_count = len(event_codes)

    if event_count:
        categorical[-event_count:, 0] = event_codes + 1
        categorical[-event_count:, 1] = product_codes + 2
        timing[-event_count:, 0] = np.log1p(np.maximum(0, np.diff(timestamps, prepend=timestamps[0])) / 1000) / 10
        reference_timestamp = timestamps[-1] if reference_time is None else reference_time

        assert reference_timestamp >= timestamps[-1]

        timing[-event_count:, 1] = np.log1p(np.maximum(0, reference_timestamp - timestamps) / 1000) / 10

    return categorical, timing


def build_sequences():
    OUT.mkdir(exist_ok=True)
    products = np.sort(pq.read_table(ROOT / "data/train.parquet", columns=["target_product"])
                       .column(0).unique().to_numpy())
    dictionary = pd.read_parquet(ROOT / "data/event_dictionary.parquet")
    target_events = dictionary.loc[dictionary.is_target_event, "encoded_event"].to_numpy()
    audit = {}

    for split in ("train", "test"):
        path = OUT / f"{split}_sequences.npz"

        if path.exists():
            continue

        history_c, history_t, history_m = [], [], []
        full_c, full_t, full_m = [], [], []
        seen, target_seen = set(), set()
        recent_clicks = 0
        n = 0

        for batch in pq.ParquetFile(ROOT / f"data/{split}.parquet").iter_batches(batch_size=64):
            arrays = batch.column(batch.schema.get_field_index("events"))
            offsets = arrays.offsets.to_numpy()
            flat = arrays.values
            ev = flat.field("encoded_event").to_numpy()
            pr = flat.field("encoded_product").fill_null(-1).to_numpy()
            ts = flat.field("timestamp").to_numpy()

            clients = batch.column(batch.schema.get_field_index("client_id")).to_numpy()
            labels = (batch.column(batch.schema.get_field_index("target_product")).to_numpy()
                      if split == "train" else np.full(len(clients), -1))
            seen.update(pr[pr >= 0].tolist())
            target_seen.update(pr[np.isin(ev, target_events) & (pr >= 0)].tolist())

            for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
                e, p, t = ev[lo:hi], pr[lo:hi], ts[lo:hi]

                assert len(e) > 0 and np.all(np.diff(t) >= 0)

                c, z = encode_sequence(e, p, t)
                full_c.append(c)
                full_t.append(z)
                full_m.append((clients[j], labels[j], t[-1]))
                observed = np.flatnonzero(np.isin(e, target_events) & np.isin(p, products))
                recent_clicks += int(np.sum(t[observed] >= RECENT))

                for i in observed[observed >= 25][-2:]:
                    if np.isin(e[i - 5:i], target_events).any():
                        continue

                    c, z = encode_sequence(e[:i - 5], p[:i - 5], t[:i - 5])
                    history_c.append(c)
                    history_t.append(z)
                    history_m.append((clients[j], p[i], t[i - 6], t[i]))

            n += len(clients)

            if n % 10000 < 64:
                print("sequence cache", split, n, flush=True)

        np.savez_compressed(path, full_c=np.asarray(full_c), full_t=np.asarray(full_t),
                            history_c=np.asarray(history_c), history_t=np.asarray(history_t),
                            full_m=np.asarray(full_m), history_m=np.asarray(history_m), products=products)
        audit[split] = {"all_products": sorted(map(int, seen)),
                        "target_click_products": sorted(map(int, target_seen)),
                        "recent_target_clicks": recent_clicks,
                        "target_click_outside_train_labels": sorted(map(int, target_seen - set(products)))}
        print("product audit", split, audit[split], flush=True)

    if audit:
        (OUT / "product_audit.json").write_text(json.dumps(audit, indent=2))


def load_data(split, history=False):
    key = "history" if history else "full"
    kind = "masked_history" if history else "candidate"

    with np.load(OUT / f"{split}_sequences.npz") as source:
        c, t, m, products = (source[f"{key}_c"], source[f"{key}_t"],
                             source[f"{key}_m"], source["products"])

    meta = pd.read_parquet(CACHE / f"{split}_{kind}_meta.parquet")

    if history:
        # The older tabular cache has three extra prefixes around duplicate clicks.
        keys = ["client_id", "target_product", "end", "click_time"]
        rows = pd.MultiIndex.from_frame(meta[keys]).get_indexer(pd.MultiIndex.from_arrays(m.T))

        assert (rows >= 0).all()

        meta = meta.iloc[rows].reset_index(drop=True)
    else:
        rows = np.arange(len(meta))

    assert np.array_equal(m[:, 0], meta.client_id)
    assert np.array_equal(m[:, 2], meta.end)

    if history or split == "train":
        assert np.array_equal(m[:, 1], meta.target_product)

    with np.load(CACHE / f"{split}_{kind}_features.npz") as source:
        assert np.array_equal(products, source["products"])

        x = source["product"][rows]
        context = source["context"][rows, :11]

    # Missing recency is a separate signal, never filled with a future observation.
    x = np.concatenate([np.nan_to_num(x, nan=0, posinf=0, neginf=0), np.isnan(x)], axis=-1)
    return {"c": c, "t": t, "x": x.astype(np.float32), "context": context,
            "y": np.searchsorted(products, m[:, 1]), "meta": meta, "products": products}


def take(data, indices):
    """Выбрать одинаковые строки массивов и metadata; каталог продуктов общий."""
    selected = {}

    for name, value in data.items():
        if name in ("c", "t", "x", "context", "y"):
            selected[name] = value[indices]
        elif name == "meta":
            selected[name] = value.iloc[indices].reset_index(drop=True)
        else:
            selected[name] = value

    return selected


def concatenate(a, b):
    """Объединить примеры двух наборов с одинаковым порядком продуктов."""

    assert np.array_equal(a["products"], b["products"])

    combined = {}

    for name in a:
        if name in ("c", "t", "x", "context", "y"):
            combined[name] = np.concatenate([a[name], b[name]])
        elif name == "meta":
            combined[name] = pd.concat([a[name], b[name]], ignore_index=True)
        else:
            combined[name] = a[name]

    return combined


def normalization(data):
    """Оценить средние и масштабы только по переданной обучающей выборке."""
    x = data["x"]
    context = data["context"]

    return (x.mean((0, 1)), np.maximum(x.std((0, 1)), .1),
            context.mean(0), np.maximum(context.std(0), .1))


class ProductGRU(nn.Module):
    """Общий GRU читает историю, затем одна и та же голова оценивает каждый продукт.

    Имена слоёв и buffers сохраняются для совместимости с готовыми .pt-весами.
    """

    def __init__(self, feature_count=82, hidden=64, dropout=.25, sequence=True, identity=True,
                 product_count=32, attention=False):
        super().__init__()
        self.sequence, self.identity = sequence, identity
        self.attention = attention
        self.event = nn.Embedding(1200, 24, padding_idx=0)
        self.product = nn.Embedding(128, 16, padding_idx=0)
        self.gru = nn.GRU(42, hidden, batch_first=True)

        self.query = nn.Linear(hidden, 16)

        if attention:
            self.attention_key = nn.Linear(hidden, 16)
            self.match_bias = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + 16 + feature_count + 11, 128),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64),
                                  nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("product_codes", torch.zeros(product_count, dtype=torch.long))
        self.register_buffer("x_mean", torch.zeros(feature_count))
        self.register_buffer("x_std", torch.ones(feature_count))
        self.register_buffer("context_mean", torch.zeros(11))

        self.register_buffer("context_std", torch.ones(11))

    def forward(self, categorical, timing, product_features, context):
        """Вернуть оценки [batch, продукты]; softmax применяется в predict()."""

        if self.sequence:
            embeddings = torch.cat(
                [self.event(categorical[:, :, 0]), self.product(categorical[:, :, 1]), timing], -1
            )
            history_states, hidden = self.gru(self.dropout(embeddings))
            hidden = self.dropout(hidden[-1])
        else:
            hidden = torch.zeros((len(categorical), self.gru.hidden_size), device=categorical.device)

        candidate_embeddings = self.product(self.product_codes)

        if not self.identity:
            candidate_embeddings = candidate_embeddings * 0

        candidate_embeddings = self.dropout(candidate_embeddings).unsqueeze(0).expand(len(categorical), -1, -1)
        product_features = torch.clamp((product_features - self.x_mean) / self.x_std, -10, 10)
        context = torch.clamp((context - self.context_mean) / self.context_std, -10, 10)
        candidate_count = len(self.product_codes)
        product_context = hidden[:, None].expand(-1, candidate_count, -1)

        if self.attention:
            # Каждый кандидат получает свои веса внимания к событиям истории.
            attention_keys = self.attention_key(history_states)
            attention_scores = torch.einsum("bsh,bch->bcs", attention_keys, candidate_embeddings) / 4
            same_product_mask = categorical[:, None, :, 1] == self.product_codes[None, :, None]
            attention_scores = attention_scores + self.match_bias * same_product_mask.float()
            attention_scores = attention_scores.masked_fill(categorical[:, None, :, 0] == 0, -1e4)
            product_context = (product_context + attention_scores.softmax(-1) @ history_states) / 2

        combined_features = torch.cat(
            [product_context, candidate_embeddings, product_features,
             context[:, None].expand(-1, candidate_count, -1)], -1
        )
        candidate_similarity = (self.query(hidden)[:, None] * candidate_embeddings).sum(-1) / 4

        return self.head(combined_features).squeeze(-1) + candidate_similarity


def tensors(data):
    """Перенести входы на DEVICE, сохранив целые категории и индексы классов."""
    return [torch.from_numpy(data[k]).to(DEVICE, dtype=torch.long if k in ("c", "y") else torch.float32)
            for k in ("c", "t", "x", "context", "y")]


@torch.no_grad()
def predict(model, data, batch_size=512):
    """Получить softmax-вероятности в исходном порядке строк и кандидатов."""
    model.eval()
    result = []

    for start in range(0, len(data["y"]), batch_size):
        end = start + batch_size
        batch = [torch.as_tensor(data[k][start:end], device=DEVICE,
                                 dtype=torch.long if k == "c" else torch.float32)
                 for k in ("c", "t", "x", "context")]
        result.append(model(*batch).softmax(-1).cpu().numpy())

    return np.concatenate(result)


def fit(train, valid, name, epochs=15, lr=.001, seed=42, sequence=True, identity=True,
        base=None, fixed=False, patience=4, attention=False, hidden=64, monitor_role=None):
    """Обучить новую модель или копию base и сохранить веса с журналом.

    При fixed=True всегда сохраняется последняя из заданных эпох. Тогда valid
    служит только для отчёта и не определяет момент остановки.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    model = (copy.deepcopy(base) if base is not None else
             ProductGRU(sequence=sequence, identity=identity, product_count=len(train["products"]),
                        attention=attention, hidden=hidden, feature_count=train["x"].shape[-1]))

    if base is None:
        stats = normalization(train)

        for key, value in zip(("x_mean", "x_std", "context_mean", "context_std"), stats):
            getattr(model, key).copy_(torch.from_numpy(value))

        model.product_codes.copy_(torch.from_numpy(train["products"] + 2))

    model.to(DEVICE)
    batch = tensors(train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    best, best_epoch, best_state = -1, 0, None
    log = []

    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        model.train()
        order = torch.randperm(len(batch[-1]), device=DEVICE)
        loss_sum = 0

        for indices in order.split(512):
            logits = model(*(v[indices] for v in batch[:-1]))
            loss = nn.functional.cross_entropy(logits, batch[-1][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            loss_sum += loss.item() * len(indices)

        predictions = predict(model, valid)
        accuracy = float(np.mean(predictions.argmax(1) == valid["y"]))
        row = dict(epoch=epoch, loss=loss_sum / len(train["y"]), accuracy=accuracy,
                   seconds=round(time.monotonic() - start, 2),
                   metric_role=monitor_role or ("training_monitor" if fixed else "selection_validation"))
        log.append(row)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(name, row, flush=True)

        if accuracy > best or fixed:
            best, best_epoch = accuracy, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if not fixed and epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "sequence": sequence, "identity": identity,
                "epoch": best_epoch, "accuracy": best,
                "product_count": len(train["products"]), "attention": model.attention,
                "hidden": model.gru.hidden_size, "seed": seed,
                "feature_count": len(model.x_mean)}, OUT / f"{name}.pt")
    pd.DataFrame(log).to_csv(OUT / f"{name}_training.csv", index=False)
    np.savez_compressed(OUT / f"{name}_valid.npz", predictions=predict(model, valid),
                        labels=valid["y"], clients=valid["meta"].client_id.to_numpy(),
                        products=valid["products"])
    print("selected", name, "epoch", best_epoch, "accuracy", best, flush=True)
    del batch

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return model, best_epoch


def self_check():
    c, t = encode_sequence(np.array([2, 469, 3]), np.array([6, 6, -1]), np.array([0, 1000, 3000]), 4)

    assert c.tolist() == [[0, 0], [3, 8], [470, 8], [4, 1]]
    assert t[-1, 1] == 0 and np.isfinite(t).all()

    torch.manual_seed(42)

    for count, attention in [(32, False), (42, True)]:
        model = ProductGRU(product_count=count, attention=attention).eval()
        model.product_codes.copy_(torch.arange(count) + 2)
        out = model(torch.from_numpy(c[None].astype(np.int64)),
                    torch.from_numpy(t[None].astype(np.float32)),
                    torch.zeros(1, count, 82), torch.zeros(1, 11))

        assert out.shape == (1, count) and torch.isfinite(out).all()

        out.sum().backward()

        assert model.gru.weight_ih_l0.grad is not None

        if attention:
            assert torch.isfinite(model.attention_key.weight.grad).all()

            empty = model(torch.zeros(1, 4, 2, dtype=torch.long), torch.zeros(1, 4, 2),
                          torch.zeros(1, count, 82), torch.zeros(1, 11))

            assert torch.isfinite(empty).all()

    print("GRU_SELF_CHECK_OK", flush=True)


def run_experiment(variant="gru"):
    build_sequences()
    full = load_data("train")
    a = np.flatnonzero(full["meta"].end.to_numpy() < CUTOFF)
    b = np.flatnonzero(full["meta"].end.to_numpy() >= CUTOFF)
    history = load_data("train", history=True)
    history = take(history, np.flatnonzero((history["meta"].click_time.to_numpy() < CUTOFF) &
                                         history["meta"].client_id.isin(full["meta"].iloc[a].client_id)))

    train = concatenate(take(full, a), history)
    valid = take(full, b)

    if variant == "short":
        for data in (train, valid):
            data["c"], data["t"] = data["c"][:, -48:], data["t"][:, -48:]

    assert not set(train["meta"].client_id) & set(valid["meta"].client_id)

    print("device", DEVICE, "train", len(train["y"]), "validation", len(valid["y"]), flush=True)
    fit(train, valid, f"{variant}_time", epochs=18,
        sequence=variant != "mlp", identity=variant != "anonymous")


def load_model(name):
    """Восстановить архитектуру и веса из research-каталога OUT."""
    checkpoint = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True)
    model = ProductGRU(sequence=checkpoint["sequence"], identity=checkpoint["identity"],
                       product_count=checkpoint.get("product_count", 32),
                       attention=checkpoint.get("attention", False), hidden=checkpoint.get("hidden", 64),
                       feature_count=checkpoint.get("feature_count", 82))
    model.load_state_dict(checkpoint["state"])
    model.training_seed = checkpoint.get("seed", 42)

    return model.to(DEVICE), checkpoint["epoch"]


def recent_split(data):
    recent = take(data, np.flatnonzero(data["meta"].click_time.to_numpy() >= RECENT))
    train, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=2026)
                          .split(recent["y"], groups=recent["meta"].client_id))

    assert not set(recent["meta"].iloc[train].client_id) & set(recent["meta"].iloc[holdout].client_id)

    return recent, train, holdout


def write_submission(probabilities, products, filename):
    meta = pd.read_parquet(CACHE / "test_candidate_meta.parquet")
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")

    assert probabilities.shape == (len(meta), len(products)) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)

    predictions = pd.Series(products[probabilities.argmax(1)], index=meta.example_id)

    assert predictions.index.is_unique and sample.example_id.is_unique
    assert set(predictions.index) == set(sample.example_id)

    sample["encoded_product"] = sample.example_id.map(predictions)

    assert sample.encoded_product.notna().all() and sample.encoded_product.isin(products).all()

    sample.to_csv(OUT / filename, index=False)
    print("saved", OUT / filename, len(sample), flush=True)


def distribution(probabilities):
    """Нормировать бинарные odds LightGBM в распределение по кандидатам."""
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    odds = probabilities / (1 - probabilities)

    return odds / odds.sum(1, keepdims=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--variant", choices=["gru", "mlp", "anonymous", "short"], default="gru")
    args = parser.parse_args()

    if args.check:
        self_check()
    elif args.cache:
        build_sequences()
    else:
        run_experiment(args.variant)

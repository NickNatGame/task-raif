"""Time, session, exposure and candidate-matching features from observed history only."""
import numpy as np

CONTEXT_NAMES = ["history_length", "span_days", "session_length", "session_seconds", "hour", "weekday", "last_gap_seconds", "view_total", "click_total", "target_total", "product_diversity"]
CONTEXT_NAMES += [f"last_event_{i}" for i in range(1, 11)]
CONTEXT_NAMES += [f"last_product_{i}" for i in range(1, 11)]
CONTEXT_NAMES += [f"last_view_product_{i}" for i in range(1, 6)]
CONTEXT_NAMES += [f"last_target_product_{i}" for i in range(1, 4)]
PRODUCT_NAMES = ["views", "clicks", "targets", "views_10", "views_50", "views_session", "clicks_session", "views_hour", "views_day", "targets_day", "view_decay_hour", "view_decay_day", "target_decay_day", "view_age_seconds", "click_age_seconds", "target_age_seconds", "view_distance", "view_share", "click_per_view", "target_per_view", "dwell_proxy_seconds", "fresh_view", "session_view_share"]
PRODUCT_NAMES += [f"match_last_product_{i}" for i in range(1, 11)]
PRODUCT_NAMES += [f"match_last_view_{i}" for i in range(1, 6)]
PRODUCT_NAMES += [f"match_last_target_{i}" for i in range(1, 4)]
CATEGORICAL_NAMES = [n for n in CONTEXT_NAMES if n.startswith("last_") and n != "last_gap_seconds"]


def extract_features(events, products, event_types, target_events):
    """Return context and one row per candidate; never receives the hidden label."""
    products = np.asarray(products, dtype=np.int64)
    if not events:
        return np.zeros(len(CONTEXT_NAMES), dtype=np.float32), np.zeros((len(products), len(PRODUCT_NAMES)), dtype=np.float32)
    event = np.array([e["encoded_event"] for e in events], dtype=np.int32)
    product = np.array([e["encoded_product"] if e["encoded_product"] is not None else -1 for e in events], dtype=np.int32)
    timestamp = np.array([e["timestamp"] for e in events], dtype=np.int64)
    age = np.maximum(0, (timestamp[-1] - timestamp) / 1000)
    position = np.arange(len(events))
    known = product >= 0
    size = max(int(products.max()), int(product.max())) + 1
    view = np.array([event_types.get(int(e)) == "view" for e in event]) & known
    click = np.array([event_types.get(int(e)) == "click" for e in event]) & known
    target = np.isin(event, list(target_events)) & known
    boundaries = np.flatnonzero(np.r_[True, np.diff(timestamp) > 1800000] | np.isin(event, [99, 687]))
    session_start = int(boundaries[-1])
    session = position >= session_start

    def count(mask, weights=None):
        return np.bincount(product[mask], weights=None if weights is None else weights[mask], minlength=size)[products]

    def last(mask):
        indices = np.full(size, -1, dtype=np.int32)
        np.maximum.at(indices, product[mask], position[mask])
        indices = indices[products]
        values = np.full(len(products), np.nan)
        exists = indices >= 0
        values[exists] = age[indices[exists]]
        return values, indices

    views, clicks, targets = count(view), count(click), count(target)
    session_views = count(view & session)
    view_age, view_index = last(view)
    click_age, _ = last(click)
    target_age, target_index = last(target)
    distance = np.where(view_index >= 0, len(events) - 1 - view_index, np.nan)
    dwell = np.r_[np.minimum(np.maximum(np.diff(timestamp) / 1000, 0), 60), 0]
    matrices = [
        np.log1p(views), np.log1p(clicks), np.log1p(targets),
        np.log1p(count(view & (position >= len(events) - 10))),
        np.log1p(count(view & (position >= len(events) - 50))),
        np.log1p(session_views), np.log1p(count(click & session)),
        np.log1p(count(view & (age <= 3600))), np.log1p(count(view & (age <= 86400))),
        np.log1p(count(target & (age <= 86400))),
        count(view, np.exp(-age / 3600)), count(view, np.exp(-age / 86400)),
        count(target, np.exp(-age / 86400)),
        np.log1p(view_age), np.log1p(click_age), np.log1p(target_age), np.log1p(distance),
        views / max(1, views.sum()), clicks / (views + 1), targets / (views + 1),
        np.log1p(count(view, dwell)), (view_index > target_index).astype(float),
        session_views / max(1, session_views.sum()),
    ]
    context = [
        np.log1p(len(events)), np.log1p((timestamp[-1] - timestamp[0]) / 86400000),
        np.log1p(len(events) - session_start), np.log1p(age[session_start]),
        (timestamp[-1] // 3600000) % 24, (timestamp[-1] // 86400000 + 3) % 7,
        np.log1p(age[-2]) if len(events) > 1 else 0,
        np.log1p(view.sum()), np.log1p(click.sum()), np.log1p(target.sum()),
        len(np.unique(product[known])),
    ]
    for i in range(1, 11):
        context.append(event[-i] if len(event) >= i else -1)
    last_products = [product[-i] if len(product) >= i else -1 for i in range(1, 11)]
    context.extend(last_products)
    view_products = product[view][::-1][:5].tolist()
    target_products = product[target][::-1][:3].tolist()
    view_products += [-1] * (5 - len(view_products))
    target_products += [-1] * (3 - len(target_products))
    context.extend(view_products + target_products)
    for p in last_products + view_products + target_products:
        matrices.append((products == p).astype(float))
    return np.asarray(context, dtype=np.float32), np.column_stack(matrices).astype(np.float32)


def self_check():
    kinds = {1: "view", 2: "click", 3: "click", 9: "other"}
    events = [
        {"timestamp": 0, "encoded_event": 1, "encoded_product": 3},
        {"timestamp": 1000, "encoded_event": 3, "encoded_product": 3},
        {"timestamp": 2000000, "encoded_event": 1, "encoded_product": 6},
        {"timestamp": 2001000, "encoded_event": 9, "encoded_product": None},
    ]
    context, matrix = extract_features(events, [3, 6, 8], kinds, {3})
    assert context.shape == (len(CONTEXT_NAMES),)
    assert matrix.shape == (3, len(PRODUCT_NAMES))
    c = {n: i for i, n in enumerate(PRODUCT_NAMES)}
    assert matrix[0, c["targets"]] == np.float32(np.log(2))
    assert matrix[1, c["views_session"]] == np.float32(np.log(2))
    assert matrix[0, c["views_session"]] == 0
    assert matrix[1, c["view_age_seconds"]] == np.float32(np.log(2))
    assert np.isnan(matrix[2, c["view_age_seconds"]])
    assert matrix[1, c["match_last_view_1"]] == 1
    assert not np.isinf(matrix).any()
    assert extract_features([], [3, 6, 8], kinds, {3})[1].shape == matrix.shape


"""Агрегаты по истории клиента: просмотры, клики, давность и текущая сессия"""

import numpy as np

# Порядок столбцов входит в формат обученных моделей.
CONTEXT_NAMES = [
    "history_length", "span_days", "session_length", "session_seconds", "hour", "weekday",
    "last_gap_seconds", "view_total", "click_total", "target_total", "product_diversity",
]
CONTEXT_NAMES += [f"last_event_{i}" for i in range(1, 11)]
CONTEXT_NAMES += [f"last_product_{i}" for i in range(1, 11)]
CONTEXT_NAMES += [f"last_view_product_{i}" for i in range(1, 6)]

CONTEXT_NAMES += [f"last_target_product_{i}" for i in range(1, 4)]
PRODUCT_NAMES = [
    "views", "clicks", "targets", "views_10", "views_50", "views_session", "clicks_session",
    "views_hour", "views_day", "targets_day", "view_decay_hour", "view_decay_day",
    "target_decay_day", "view_age_seconds", "click_age_seconds", "target_age_seconds",
    "view_distance", "view_share", "click_per_view", "target_per_view", "dwell_proxy_seconds",
    "fresh_view", "session_view_share",
]
PRODUCT_NAMES += [f"match_last_product_{i}" for i in range(1, 11)]
PRODUCT_NAMES += [f"match_last_view_{i}" for i in range(1, 6)]
PRODUCT_NAMES += [f"match_last_target_{i}" for i in range(1, 4)]
CATEGORICAL_NAMES = [n for n in CONTEXT_NAMES if n.startswith("last_") and n != "last_gap_seconds"]


def extract_features(events, products, event_types, target_events):
    """Вернуть общий контекст и строку признаков для каждого продукта-кандидата.

    Столбцы следуют CONTEXT_NAMES и PRODUCT_NAMES. Пропущенная давность
    обозначается NaN: отсутствие события отличается от события только что.
    """
    products = np.asarray(products, dtype=np.int64)

    if not events:
        return (
            np.zeros(len(CONTEXT_NAMES), dtype=np.float32),
            np.zeros((len(products), len(PRODUCT_NAMES)), dtype=np.float32),
        )

    event_codes = np.array([e["encoded_event"] for e in events], dtype=np.int32)
    product_codes = np.array(
        [e["encoded_product"] if e["encoded_product"] is not None else -1 for e in events],
        dtype=np.int32,
    )
    timestamps = np.array([e["timestamp"] for e in events], dtype=np.int64)
    age_seconds = np.maximum(0, (timestamps[-1] - timestamps) / 1000)
    event_positions = np.arange(len(events))
    has_product = product_codes >= 0

    product_capacity = max(int(products.max()), int(product_codes.max())) + 1
    view_mask = np.array([event_types.get(int(e)) == "view" for e in event_codes]) & has_product
    click_mask = np.array([event_types.get(int(e)) == "click" for e in event_codes]) & has_product
    target_mask = np.isin(event_codes, list(target_events)) & has_product
    # Новая сессия начинается после паузы >30 минут или явного события 99/687.
    boundaries = np.flatnonzero(
        np.r_[True, np.diff(timestamps) > 1800000] | np.isin(event_codes, [99, 687])
    )
    session_start = int(boundaries[-1])

    current_session_mask = event_positions >= session_start

    def count(mask, weights=None):
        return np.bincount(
            product_codes[mask],
            weights=None if weights is None else weights[mask],
            minlength=product_capacity,
        )[products]

    def last(mask):
        indices = np.full(product_capacity, -1, dtype=np.int32)
        np.maximum.at(indices, product_codes[mask], event_positions[mask])
        indices = indices[products]
        values = np.full(len(products), np.nan)
        exists = indices >= 0
        values[exists] = age_seconds[indices[exists]]

        return values, indices

    views, clicks, targets = count(view_mask), count(click_mask), count(target_mask)
    session_views = count(view_mask & current_session_mask)
    view_age, view_index = last(view_mask)
    click_age, _ = last(click_mask)
    target_age, target_index = last(target_mask)
    distance = np.where(view_index >= 0, len(events) - 1 - view_index, np.nan)

    dwell_seconds = np.r_[np.minimum(np.maximum(np.diff(timestamps) / 1000, 0), 60), 0]
    product_columns = [
        np.log1p(views), np.log1p(clicks), np.log1p(targets),
        np.log1p(count(view_mask & (event_positions >= len(events) - 10))),
        np.log1p(count(view_mask & (event_positions >= len(events) - 50))),
        np.log1p(session_views), np.log1p(count(click_mask & current_session_mask)),
        np.log1p(count(view_mask & (age_seconds <= 3600))), np.log1p(count(view_mask & (age_seconds <= 86400))),
        np.log1p(count(target_mask & (age_seconds <= 86400))),
        count(view_mask, np.exp(-age_seconds / 3600)), count(view_mask, np.exp(-age_seconds / 86400)),
        count(target_mask, np.exp(-age_seconds / 86400)),
        np.log1p(view_age), np.log1p(click_age), np.log1p(target_age), np.log1p(distance),
        views / max(1, views.sum()), clicks / (views + 1), targets / (views + 1),
        np.log1p(count(view_mask, dwell_seconds)), (view_index > target_index).astype(float),
        session_views / max(1, session_views.sum()),
    ]
    context = [
        np.log1p(len(events)), np.log1p((timestamps[-1] - timestamps[0]) / 86400000),
        np.log1p(len(events) - session_start), np.log1p(age_seconds[session_start]),
        (timestamps[-1] // 3600000) % 24, (timestamps[-1] // 86400000 + 3) % 7,
        np.log1p(age_seconds[-2]) if len(events) > 1 else 0,
        np.log1p(view_mask.sum()), np.log1p(click_mask.sum()), np.log1p(target_mask.sum()),
        len(np.unique(product_codes[has_product])),
    ]

    for i in range(1, 11):
        context.append(event_codes[-i] if len(event_codes) >= i else -1)

    last_products = [product_codes[-i] if len(product_codes) >= i else -1 for i in range(1, 11)]
    context.extend(last_products)
    view_products = product_codes[view_mask][::-1][:5].tolist()
    target_products = product_codes[target_mask][::-1][:3].tolist()
    view_products += [-1] * (5 - len(view_products))
    target_products += [-1] * (3 - len(target_products))

    context.extend(view_products + target_products)

    for p in last_products + view_products + target_products:
        product_columns.append((products == p).astype(float))

    return np.asarray(context, dtype=np.float32), np.column_stack(product_columns).astype(np.float32)


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


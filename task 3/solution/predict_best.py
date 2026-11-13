"""Воспроизвести лучший CSV по исходным parquet и сохранённым весам без обучения"""

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from next_click_behavior import behavior_features
from next_click_features import extract_features

from next_click_gru import DEVICE, ProductGRU, encode_sequence, predict, distribution
from next_click_model import candidate_table
from next_click_view_codes import features as view_code_features

ROOT = Path(__file__).resolve().parent


def run(output):
    """Записать отдельный CSV, проверив совпадение с сохранённым лучшим решением."""
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())

    for name, expected_sha256 in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_sha256, name

    assert output.resolve() != (ROOT.parent / 'submission_view_codes.csv').resolve(), 'Выбери отдельный путь для нового CSV'

    checkpoint = torch.load(ROOT / 'models/view_codes_gru.pt', map_location='cpu', weights_only=True)
    gru_model = ProductGRU(
        feature_count=checkpoint['feature_count'], hidden=checkpoint['hidden'],
        product_count=checkpoint['product_count'], attention=checkpoint['attention'],
        sequence=checkpoint['sequence'], identity=checkpoint['identity'],
    )
    gru_model.load_state_dict(checkpoint['state'])
    gru_model.to(DEVICE).eval()
    torch.set_num_threads(4)
    candidate_products = gru_model.product_codes.cpu().numpy() - 2

    assert candidate_products.tolist() == manifest['candidate_products']

    tree_model = lgb.Booster(model_file=str(ROOT / 'models/lightgbm.txt'))
    event_dictionary = pd.read_parquet(ROOT / 'data/event_dictionary.parquet')
    event_types = event_dictionary.set_index('encoded_event').event_type.to_dict()
    target_event_codes = set(event_dictionary.loc[event_dictionary.is_target_event, 'encoded_event'])
    event_kinds = np.zeros(1200, np.int8)
    event_kinds[event_dictionary.encoded_event] = np.where(event_dictionary.event_type == 'view', 1, 2)

    original_catalog_mask = np.isin(candidate_products, manifest['original_train_products'])
    example_ids, probability_batches = [], []

    for batch in pq.ParquetFile(ROOT / 'data/test.parquet').iter_batches(batch_size=64):
        categorical, timing = [], []
        neural_features, neural_context = [], []
        tree_context, tree_features = [], []

        for row in batch.to_pylist():
            events = row['events']
            event_codes = np.array([r['encoded_event'] for r in events], np.int32)
            product_codes = np.array(
                [-1 if r['encoded_product'] is None else r['encoded_product'] for r in events],
                np.int32,
            )
            timestamps = np.array([r['timestamp'] for r in events], np.int64)

            assert len(event_codes) and np.all(np.diff(timestamps) >= 0)

            context_features, product_features = extract_features(
                events, candidate_products, event_types, target_event_codes
            )
            encoded_events, time_features = encode_sequence(event_codes, product_codes, timestamps)
            # При обучении признаки хранились в float16; округление нужно сохранить.
            quantized_product_features = product_features.astype(np.float16).astype(np.float32)
            behavior = behavior_features(
                event_codes, product_codes, timestamps, candidate_products, event_kinds
            ).astype(np.float16).astype(np.float32)
            codes = view_code_features(
                encoded_events[None], time_features[None], candidate_products
            )[0].astype(np.float32)
            neural_features.append(np.concatenate(
                [np.nan_to_num(quantized_product_features, nan=0),
                 np.isnan(quantized_product_features), behavior, codes], axis=-1
            ))

            categorical.append(encoded_events)
            timing.append(time_features)
            neural_context.append(context_features[:11])
            tree_context.append(context_features)

            # Старый LightGBM ожидает доли от каталога 32 исходных train-продуктов.
            for count_column, share_column in [(0, 17), (5, 22)]:
                counts = np.expm1(product_features[:, count_column])
                product_features[:, share_column] = counts / max(counts[original_catalog_mask].sum(), 1)

            tree_features.append(product_features)
            example_ids.append(row['example_id'])

        data = dict(
            c=np.asarray(categorical), t=np.asarray(timing), x=np.asarray(neural_features),
            context=np.asarray(neural_context), y=np.zeros(len(categorical), np.int64),
        )
        neural_probabilities = predict(gru_model, data)
        candidate_rows, _ = candidate_table(
            np.asarray(tree_context), np.asarray(tree_features), candidate_products
        )
        binary_tree_probabilities = tree_model.predict(candidate_rows, num_threads=4).reshape(
            len(categorical), len(candidate_products)
        )
        tree_probabilities = distribution(binary_tree_probabilities)
        probability_batches.append(.75 * neural_probabilities + .25 * tree_probabilities)

    probabilities = np.concatenate(probability_batches)

    assert probabilities.shape == (manifest['test_rows'], len(candidate_products))
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)

    predictions = pd.Series(candidate_products[probabilities.argmax(1)], index=example_ids)
    # Строки parquet и sample_submission могут идти в разном порядке.
    submission = pd.read_csv(ROOT / 'data/sample_submission.csv')

    assert predictions.index.is_unique and submission.example_id.is_unique
    assert set(predictions.index) == set(submission.example_id)

    submission['encoded_product'] = submission.example_id.map(predictions)

    assert submission.encoded_product.notna().all() and submission.encoded_product.isin(candidate_products).all()

    reference = pd.read_csv(ROOT.parent / 'submission_view_codes.csv')

    assert submission.equals(reference), 'Прогноз отличается от лучшего; новый CSV не записан'

    submission.to_csv(output, index=False)
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    assert output_sha256 == manifest['sha256']['../submission_view_codes.csv']

    print(f'BEST_REPRODUCED: {len(submission)} rows, device={DEVICE}, SHA256={output_sha256}', flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, default=ROOT / 'submission_reproduced.csv',
        help='Путь для нового CSV; сохранённый submission_view_codes.csv защищён от перезаписи',
    )
    run(parser.parse_args().output)

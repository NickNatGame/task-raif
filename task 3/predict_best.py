"""Reproduce the confirmed best submission from raw parquet and bundled weights."""
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
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert output.resolve() != (ROOT / 'submission_view_codes.csv').resolve(), 'Keep the confirmed best CSV'
    checkpoint = torch.load(ROOT / 'models/view_codes_gru.pt', map_location='cpu', weights_only=True)
    model = ProductGRU(feature_count=checkpoint['feature_count'], hidden=checkpoint['hidden'],
                       product_count=checkpoint['product_count'], attention=checkpoint['attention'],
                       sequence=checkpoint['sequence'], identity=checkpoint['identity'])
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE).eval()
    torch.set_num_threads(4)
    products = model.product_codes.cpu().numpy() - 2
    assert products.tolist() == manifest['candidate_products']
    booster = lgb.Booster(model_file=str(ROOT / 'models/lightgbm.txt'))
    dictionary = pd.read_parquet(ROOT / 'data/event_dictionary.parquet')
    event_types = dictionary.set_index('encoded_event').event_type.to_dict()
    targets = set(dictionary.loc[dictionary.is_target_event, 'encoded_event'])
    kinds = np.zeros(1200, np.int8)
    kinds[dictionary.encoded_event] = np.where(dictionary.event_type == 'view', 1, 2)
    known = np.isin(products, manifest['original_train_products'])
    example_ids, results = [], []
    for batch in pq.ParquetFile(ROOT / 'data/test.parquet').iter_batches(batch_size=64):
        categorical, timing, neural_x, neural_context, tree_context, tree_x = [], [], [], [], [], []
        for row in batch.to_pylist():
            events = row['events']
            e = np.array([r['encoded_event'] for r in events], np.int32)
            p = np.array([-1 if r['encoded_product'] is None else r['encoded_product'] for r in events], np.int32)
            t = np.array([r['timestamp'] for r in events], np.int64)
            assert len(e) and np.all(np.diff(t) >= 0)
            context, matrix = extract_features(events, products, event_types, targets)
            c, tm = encode_sequence(e, p, t)
            # Preserve the float16 feature quantization used by the winning checkpoint.
            raw = matrix.astype(np.float16).astype(np.float32)
            behavior = behavior_features(e, p, t, products, kinds).astype(np.float16).astype(np.float32)
            codes = view_code_features(c[None], tm[None], products)[0].astype(np.float32)
            neural_x.append(np.concatenate([np.nan_to_num(raw, nan=0), np.isnan(raw), behavior, codes], axis=-1))
            categorical.append(c)
            timing.append(tm)
            neural_context.append(context[:11])
            tree_context.append(context)
            for count_column, share_column in [(0, 17), (5, 22)]:
                counts = np.expm1(matrix[:, count_column])
                matrix[:, share_column] = counts / max(counts[known].sum(), 1)
            tree_x.append(matrix)
            example_ids.append(row['example_id'])
        data = dict(c=np.asarray(categorical), t=np.asarray(timing), x=np.asarray(neural_x),
                    context=np.asarray(neural_context), y=np.zeros(len(categorical), np.int64))
        nn = predict(model, data)
        table, _ = candidate_table(np.asarray(tree_context), np.asarray(tree_x), products)
        raw_tree = booster.predict(table, num_threads=4).reshape(len(categorical), len(products))
        tree = distribution(raw_tree)
        results.append(.75 * nn + .25 * tree)
    probabilities = np.concatenate(results)
    assert probabilities.shape == (manifest['test_rows'], len(products))
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)
    predictions = pd.Series(products[probabilities.argmax(1)], index=example_ids)
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
    assert predictions.index.is_unique and sample.example_id.is_unique
    assert set(predictions.index) == set(sample.example_id)
    sample['encoded_product'] = sample.example_id.map(predictions)
    assert sample.encoded_product.notna().all() and sample.encoded_product.isin(products).all()
    reference = pd.read_csv(ROOT / 'submission_view_codes.csv')
    assert sample.equals(reference), 'Prediction differs from the preserved best; output was not written'
    sample.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert digest == manifest['sha256']['submission_view_codes.csv']
    print(f'BEST_REPRODUCED: {len(sample)} rows, device={DEVICE}, SHA256={digest}', flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'submission_reproduced.csv')
    run(parser.parse_args().output)

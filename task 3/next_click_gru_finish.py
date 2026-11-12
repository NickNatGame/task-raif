"""Final fits and client-held-out adaptation, with all new outputs on D:."""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from next_click_gru import (OUT, CACHE, RECENT, take, concatenate, fit, load_data,
                            load_model, predict, recent_split, write_submission, distribution)
from next_click_extended import load_extended, load_product_data, load_intent_data, build_exact_test
from next_click_model import candidate_table, CATEGORICAL, ADAPT_PARAMETERS


def legacy_table(history=True):
    build_exact_test()
    prefix = "history" if history else "full"
    with np.load(OUT / "test_extended_exact.npz") as data:
        context = data[prefix + "_context"]
        matrix = data[prefix + "_product"]
        products = data["products"]
    with np.load(CACHE / "train_candidate_features.npz") as old:
        original = old["products"]
    known = np.isin(products, original)
    views = np.expm1(matrix[:, :, 0])
    session = np.expm1(matrix[:, :, 5])
    matrix[:, :, 17] = views / np.maximum(views[:, known].sum(1, keepdims=True), 1)
    matrix[:, :, 22] = session / np.maximum(session[:, known].sum(1, keepdims=True), 1)
    table, _ = candidate_table(context, matrix, products)
    return table.reshape(len(context), len(products), -1), products


def fit_legacy(base, x, labels, count):
    y = np.eye(count, dtype=np.int8)[labels].ravel()
    model = lgb.LGBMClassifier(**ADAPT_PARAMETERS)
    model.fit(x.reshape(-1, x.shape[-1]), y, init_model=base, categorical_feature=CATEGORICAL)
    return model.booster_


def predict_legacy(model, x):
    p = model.predict(x.reshape(-1, x.shape[-1]), num_threads=4).reshape(x.shape[:2])
    return distribution(p)


def source_fits(tags=None):
    for tag in tags or ["gru", "extended"]:
        loader = {"gru": load_data, "products": load_product_data, "intent": load_intent_data}.get(tag, load_extended)
        reference, epochs = load_model(tag + "_time")
        full, history = loader("train"), loader("train", history=True)
        train = concatenate(full, history)
        monitor = take(full, np.arange(1024))
        model, _ = fit(train, monitor, tag + "_source", epochs=epochs, fixed=True,
                       sequence=reference.sequence, identity=reference.identity,
                       attention=reference.attention, hidden=reference.gru.hidden_size, seed=reference.training_seed)
        del reference
        del full, history, train, monitor
        gc.collect()
        for is_history in (True, False):
            data = loader("test", history=is_history)
            probabilities = predict(model, data)
            kind = "history" if is_history else "test"
            np.savez_compressed(OUT / f"{tag}_source_{kind}.npz", predictions=probabilities,
                                products=data["products"], labels=data["y"],
                                clients=data["meta"].client_id.to_numpy())
            if not is_history:
                write_submission(probabilities, data["products"], f"submission_{tag}_source.csv")
            del data
        del model
        gc.collect()
        torch.cuda.empty_cache()


def adaptation(tag="extended"):
    stem = "" if tag == "extended" else tag + "_"
    base, _ = load_model(tag + "_source")
    loader = {"products": load_product_data, "intent": load_intent_data}.get(tag, load_extended)
    data = loader("test", history=True)
    recent, tuning, holdout = recent_split(data)
    tune = take(recent, tuning)
    # The holdout labels must not participate in early stopping or blending selection.
    split_path = OUT / "recent_split.npz"
    np.savez_compressed(split_path, tuning=tuning, holdout=holdout,
                        clients=recent["meta"].client_id.to_numpy(), labels=recent["y"],
                        products=recent["products"])
    source = predict(base, tune)
    out_of_fold = np.zeros_like(source)
    x, products = legacy_table()
    assert np.array_equal(products, recent["products"])
    recent_indices = np.flatnonzero(data["meta"].click_time.to_numpy() >= RECENT)
    x = x[recent_indices]
    tune_x = x[tuning]
    legacy_base = lgb.Booster(model_file=str(CACHE / "candidate_full.txt"))
    legacy_oof = np.zeros_like(source)
    rounds = []
    for fold, (a, b) in enumerate(GroupKFold(3).split(tune["y"], groups=tune["meta"].client_id)):
        assert not set(tune["meta"].iloc[a].client_id) & set(tune["meta"].iloc[b].client_id)
        model, epoch = fit(take(tune, a), take(tune, b), f"{tag}_adapt_fold{fold}",
                           epochs=60, lr=.0003, base=base, patience=12, seed=42 + fold)
        out_of_fold[b] = predict(model, take(tune, b))
        legacy = fit_legacy(legacy_base, tune_x[a], tune["y"][a], len(products))
        legacy_oof[b] = predict_legacy(legacy, tune_x[b])
        rounds.append(epoch)
        del model
    choices = []
    for weight in [0, .5, 1]:
        neural = (1 - weight) * source + weight * out_of_fold
        for neural_weight in [0, .25, .5, .75, 1]:
            p = neural_weight * neural + (1 - neural_weight) * legacy_oof
            choices.append(dict(adaptation_weight=weight, neural_weight=neural_weight,
                                accuracy=float(np.mean(p.argmax(1) == tune["y"]))))
    pd.DataFrame(choices).to_csv(OUT / f"{stem}adaptation_selection.csv", index=False)
    chosen = max(choices, key=lambda row: row["accuracy"])
    chosen["rounds"] = int(np.median(rounds))
    (OUT / f"{stem}adaptation_choice.json").write_text(json.dumps(chosen, indent=2))
    np.savez_compressed(OUT / f"{stem}adaptation_tuning_oof.npz", source=source, adapted=out_of_fold,
                        legacy=legacy_oof, labels=tune["y"], products=tune["products"])
    # Fit on tuning clients only, then evaluate the preselected recipe once.
    adapted, _ = fit(tune, take(tune, np.arange(min(128, len(tune["y"])))), tag + "_adapt_tuning",
                     epochs=chosen["rounds"], lr=.0003, base=base, fixed=True)
    check = take(recent, holdout)
    source_check = predict(base, check)
    adapted_check = predict(adapted, check)
    w = chosen["adaptation_weight"]
    legacy_tuned = fit_legacy(legacy_base, tune_x, tune["y"], len(products))
    legacy_check = predict_legacy(legacy_tuned, x[holdout])
    nw = chosen["neural_weight"]
    prediction = nw * ((1 - w) * source_check + w * adapted_check) + (1 - nw) * legacy_check
    result = {"examples": len(check["y"]), "clients": int(check["meta"].client_id.nunique()),
              "source_accuracy": float(np.mean(source_check.argmax(1) == check["y"])),
              "boosted_adapted_accuracy": float(np.mean(legacy_check.argmax(1) == check["y"])),
              "selected_accuracy": float(np.mean(prediction.argmax(1) == check["y"])), **chosen}
    result["role"] = "first holdout evaluation" if tag == "extended" else "secondary holdout diagnostic"
    (OUT / f"{stem}recent_holdout.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / f"{stem}recent_holdout.npz", source=source_check, selected=prediction,
                        adapted=adapted_check,
                        legacy=legacy_check, labels=check["y"], products=check["products"],
                        clients=check["meta"].client_id.to_numpy())
    print("RECENT_HOLDOUT", result, flush=True)
    del adapted
    final, _ = fit(recent, take(recent, np.arange(min(128, len(recent["y"])))), tag + "_adapt_full",
                   epochs=chosen["rounds"], lr=.0003, base=base, fixed=True)
    test = loader("test")
    raw_adapted = predict(final, test)
    neural_predictions = (1 - w) * predict(base, test) + w * raw_adapted
    np.savez_compressed(OUT / f"{stem}gru_raw_test.npz", predictions=raw_adapted, products=products)
    legacy_full = fit_legacy(legacy_base, x, recent["y"], len(products))
    legacy_full.save_model(str(OUT / "candidate_extended_adapted.txt"))
    test_x, test_products = legacy_table(history=False)
    assert np.array_equal(test_products, products)
    legacy_predictions = predict_legacy(legacy_full, test_x)
    final_predictions = nw * neural_predictions + (1 - nw) * legacy_predictions
    np.savez_compressed(OUT / f"{tag}_adapted_test.npz", predictions=final_predictions,
                        products=test["products"])
    write_submission(neural_predictions, test["products"], f"submission_{stem}gru_adapted.csv")
    write_submission(legacy_predictions, test["products"], f"submission_{stem}extended_boosted.csv")
    write_submission(final_predictions, test["products"], f"submission_{stem}gru_blend.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["source", "adapt"], required=True)
    parser.add_argument("--model", choices=["all", "gru", "extended", "attention", "attention2", "wide", "products", "intent"], default="all")
    args = parser.parse_args()
    if args.stage == "adapt" and args.model == "gru":
        parser.error("The 32-product GRU is a source-only control; adapt an extended model.")
    if args.stage == "source":
        source_fits(None if args.model == "all" else [args.model])
    else:
        adaptation("extended" if args.model == "all" else args.model)

"""Train boosted W-BCE and raw conditional-mean models on training rows only."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

PARAMETERS = {
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 6,
    "eta": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1,
}
MAX_ROUNDS = 600
PATIENCE = 40


def inner_split(label, accounts, seed):
    rng = np.random.default_rng(seed)
    if accounts is not None:
        levels = np.unique(accounts)
        rng.shuffle(levels)
        held = levels[: max(1, round(0.2 * len(levels)))]
        mask = np.isin(accounts, held)
        return np.flatnonzero(~mask), np.flatnonzero(mask)
    train, validation = [], []
    for cls in np.unique(label):
        ids = np.flatnonzero(label == cls)
        rng.shuffle(ids)
        k = min(max(1, round(0.2 * len(ids))), len(ids) - 1)
        validation.extend(ids[:k])
        train.extend(ids[k:])
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def fit_one(x, target, weights, inner, objective, seed, threads, output):
    params = {
        **PARAMETERS,
        "objective": objective,
        "seed": seed,
        "nthread": threads,
        "eval_metric": "logloss" if objective == "binary:logistic" else "rmse",
    }
    left, right = inner

    def matrix(ids):
        selected = ids if weights is None else ids[weights[ids] > 0]
        if not len(selected):
            raise ValueError("Empty weighted training/inner-validation split")
        return xgb.DMatrix(
            x[selected],
            label=target[selected],
            weight=None if weights is None else weights[selected],
            nthread=threads,
        )

    train, validation = matrix(left), matrix(right)
    history = {}
    tuned = xgb.train(
        params,
        train,
        MAX_ROUNDS,
        evals=[(validation, "inner")],
        early_stopping_rounds=PATIENCE,
        evals_result=history,
        verbose_eval=False,
    )
    rounds = tuned.best_iteration + 1
    full = matrix(np.arange(len(x)))
    model = xgb.train(params, full, rounds, verbose_eval=False)
    model.save_model(output)
    return model, {
        "rounds": rounds,
        "best_inner_metric": float(tuned.best_score),
        "inner_history": history,
        "parameters": params,
        "inner_train_n": len(left),
        "inner_validation_n": len(right),
    }


def fit_models(x, value, loss, label, accounts, seed, threads, directory):
    inner = inner_split(label, accounts, 730000 + seed)
    total = value + loss
    if np.sum(value) <= 0 or np.sum(loss) <= 0:
        raise ValueError("W-BCE training requires positive value and loss support")
    weights = total / np.mean(total[total > 0])
    target = np.divide(
        loss, total, out=np.zeros_like(loss, dtype=float), where=total > 0
    )
    models, records = {}, {}
    models["wbce"], records["wbce"] = fit_one(
        x,
        target,
        weights,
        inner,
        "binary:logistic",
        500000 + seed,
        threads,
        directory / "wbce.ubj",
    )
    for name, amount, offset in (("value", value, 1), ("loss", loss, 2)):
        location, scale = float(np.mean(amount)), float(np.std(amount))
        if scale <= 1e-12:
            scale = 1.0
        models[name], records[name] = fit_one(
            x,
            (amount - location) / scale,
            None,
            inner,
            "reg:squarederror",
            seed + offset,
            threads,
            directory / (name + ".ubj"),
        )
        records[name].update(location=location, scale=scale)
    records["inner_train_ids"] = inner[0].tolist()
    records["inner_validation_ids"] = inner[1].tolist()
    return models, records


def predict(models, records, x, threads):
    matrix = xgb.DMatrix(x, nthread=threads)
    logit = models["wbce"].predict(matrix, output_margin=True).astype(float)
    amounts = {}
    for name in ("value", "loss"):
        info = records[name]
        raw = (
            models[name].predict(matrix).astype(float) * info["scale"]
            + info["location"]
        )
        amounts[name] = np.maximum(raw, 1e-9 if name == "value" else 0)
    if not all(np.isfinite(a).all() for a in (logit, *amounts.values())):
        raise ValueError("Nonfinite boosted prediction")
    return logit, amounts["value"], amounts["loss"]

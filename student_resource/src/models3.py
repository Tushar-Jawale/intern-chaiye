"""
Model zoo for the v3 matcher: LightGBM, XGBoost, CatBoost, random forest.
XGBoost and CatBoost train on the GPU when one is present. Blend weights are
fitted on a held-out split (grid over the simplex, lowest log loss).
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import time

import numpy as np


def gpu_available() -> bool:
    if os.environ.get("ER_NO_GPU"):
        return False
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(kind: str, x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, gpu: bool, seed: int = 0,
        rounds: int = 3000):
    t0 = time.time()
    if kind == "lgb":
        import lightgbm as lgb

        params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      max_bin=255, seed=seed, verbose=-1, num_threads=os.cpu_count() or 4)
        dtr = lgb.Dataset(x, y, free_raw_data=True)
        dva = lgb.Dataset(xv, yv, reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        info = model.best_iteration
    elif kind == "xgb":
        import xgboost as xgb

        params = dict(objective="binary:logistic", eval_metric="logloss", eta=0.05, max_depth=9,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      tree_method="hist", max_bin=256, seed=seed,
                      device="cuda" if gpu else "cpu", nthread=os.cpu_count() or 4)
        dtr = xgb.DMatrix(x, label=y)
        dva = xgb.DMatrix(xv, label=yv)
        model = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "valid")],
                          early_stopping_rounds=100, verbose_eval=False)
        info = model.best_iteration
    elif kind == "cat":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(iterations=rounds, learning_rate=0.08, depth=8, loss_function="Logloss",
                                   task_type="GPU" if gpu else "CPU", random_seed=seed, od_type="Iter",
                                   od_wait=100, verbose=0, thread_count=os.cpu_count() or 4,
                                   border_count=254 if gpu else 254)
        model.fit(x, y, eval_set=(xv, yv), use_best_model=True)
        info = model.get_best_iteration()
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        max_samples = min(1.0, 1_000_000 / max(len(y), 1))
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=20, max_features="sqrt",
                                       max_samples=max_samples, n_jobs=-1, random_state=seed)
        model.fit(x, y)
        info = model.n_estimators
    else:
        raise ValueError(kind)
    pv = predict(kind, model, xv)
    print(f"      {kind}: iters={info} valid logloss={_logloss(yv, pv):.5f} in {time.time() - t0:.0f}s", flush=True)
    return model


def predict(kind: str, model, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, np.float32)
    if kind == "lgb":
        return model.predict(x, num_iteration=model.best_iteration or None).astype(np.float32)
    if kind == "xgb":
        best = getattr(model, "best_iteration", None)
        rng = (0, best + 1) if best is not None else (0, 0)
        import xgboost as xgb

        return model.predict(xgb.DMatrix(x), iteration_range=rng).astype(np.float32)
    if kind == "cat":
        return model.predict_proba(x)[:, 1].astype(np.float32)
    if kind == "rf":
        out = np.empty(len(x), np.float32)
        step = 500_000
        for s in range(0, len(x), step):
            out[s:s + step] = model.predict_proba(x[s:s + step])[:, 1]
        return out
    raise ValueError(kind)


def save(kind: str, model, path: str) -> str:
    if kind == "lgb":
        model.save_model(path + ".lgb.txt", num_iteration=model.best_iteration or None)
        return path + ".lgb.txt"
    if kind == "xgb":
        best = getattr(model, "best_iteration", None)
        if best is not None:
            model = model[: best + 1]
        model.save_model(path + ".xgb.json")
        return path + ".xgb.json"
    if kind == "cat":
        model.save_model(path + ".cbm")
        return path + ".cbm"
    import joblib

    joblib.dump(model, path + ".rf.joblib", compress=3)
    return path + ".rf.joblib"


def load(kind: str, path: str, gpu: bool = False):
    if kind == "lgb":
        import lightgbm as lgb

        return lgb.Booster(model_file=path + ".lgb.txt")
    if kind == "xgb":
        import xgboost as xgb

        b = xgb.Booster()
        b.load_model(path + ".xgb.json")
        b.set_param({"device": "cuda" if gpu else "cpu"})
        return b
    if kind == "cat":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier()
        m.load_model(path + ".cbm")
        return m
    import joblib

    return joblib.load(path + ".rf.joblib")


class Ensemble:
    """Weighted average of several model kinds."""

    def __init__(self, kinds: list[str], weights: list[float] | None = None):
        self.kinds = list(kinds)
        self.models: dict[str, object] = {}
        self.weights = weights or [1.0 / len(kinds)] * len(kinds)

    def fit(self, x, y, xv, yv, gpu: bool, seed: int = 0) -> None:
        for k in self.kinds:
            self.models[k] = fit(k, x, y, xv, yv, gpu, seed)

    def predict_each(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {k: predict(k, self.models[k], x) for k in self.kinds}

    def predict(self, x: np.ndarray) -> np.ndarray:
        parts = self.predict_each(x)
        return self.blend(parts)

    def blend(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(len(next(iter(parts.values()))), np.float32)
        for k, w in zip(self.kinds, self.weights):
            out += np.float32(w) * parts[k]
        return out

    def tune_weights(self, parts: dict[str, np.ndarray], y: np.ndarray, step: float = 0.1) -> None:
        if len(self.kinds) == 1:
            self.weights = [1.0]
            return
        n = len(self.kinds)
        ticks = int(round(1 / step))
        best = (None, 1e9)
        for combo in itertools.product(range(ticks + 1), repeat=n - 1):
            if sum(combo) > ticks:
                continue
            w = [c / ticks for c in combo] + [(ticks - sum(combo)) / ticks]
            p = sum(np.float32(wi) * parts[k] for wi, k in zip(w, self.kinds))
            loss = _logloss(y, p)
            if loss < best[1]:
                best = (w, loss)
        self.weights = best[0]
        solo = ", ".join(f"{k}={_logloss(y, parts[k]):.5f}" for k in self.kinds)
        print(f"      blend weights {dict(zip(self.kinds, self.weights))} logloss={best[1]:.5f} (solo: {solo})", flush=True)

    def save(self, prefix: str) -> None:
        paths = {k: save(k, self.models[k], f"{prefix}_{k}") for k in self.kinds}
        with open(prefix + "_ensemble.json", "w", encoding="utf-8") as handle:
            json.dump({"kinds": self.kinds, "weights": self.weights, "paths": paths}, handle, indent=2)

    @classmethod
    def load(cls, prefix: str, gpu: bool = False) -> "Ensemble":
        with open(prefix + "_ensemble.json", encoding="utf-8") as handle:
            meta = json.load(handle)
        ens = cls(meta["kinds"], meta["weights"])
        for k in ens.kinds:
            ens.models[k] = load(k, f"{prefix}_{k}", gpu)
        return ens

"""
End-to-end CatBoost + Optuna training pipeline for binary classification.
Optimized for Kaggle (no seller-agg in this build):
- Robust CV (StratifiedKFold or GroupKFold)
- Optuna tuning with pruning + early stopping
- Tracks ROC-AUC and PR-AUC (Average Precision)
- OOF predictions, best threshold selection (accuracy|f1)
- Optional probability calibration (platt|isotonic)
- Retrains final single model + saves k-fold ensemble
- Saves artifacts and (optionally) builds a Kaggle submission from test CSV

Usage (basic):
  python train_catboost_optuna.py --data train.csv --target is_used

Run until plateau (patience on non-improving completed trials):
  python train_catboost_optuna.py --data train.csv --target is_used \
    --patience-trials 20 --min-improve 1e-4 --progress

Resume a study:
  python train_catboost_optuna.py --data train.csv --target is_used \
    --study-storage sqlite:///study.db --study-name meli_used_new --resume

Group CV:
  python train_catboost_optuna.py --data train.csv --target is_used --group-col seller_id

Create a submission:
  python train_catboost_optuna.py --data train.csv --target is_used \
    --test-data test.csv --id-col id --submission-path artifacts/submission.csv
"""

import argparse
import json
import random
import time
import math
from pathlib import Path
import hashlib
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, accuracy_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from catboost import CatBoostClassifier, Pool

EPS = 1e-10
PLATEAU_KEY = "plateau_state_v2"

# -----------------------
# Utilities
# -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _norm_key(v):
    # Stable key for JSON + matching at inference
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "__NA__"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        # collapse 1.0 -> 1 to avoid '1' vs '1.0' mismatches
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return x
    return str(v) if not isinstance(v, (str, int, float, bool)) else v

def json_ready(obj):
    # Recursively make anything JSON-serializable (incl. NumPy scalars & NaN)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = k if isinstance(k, (str, int, float, bool, type(None))) else str(k)
            out[kk] = json_ready(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(x) for x in obj]
    return obj

def detect_cat_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    # Identify object/category/boolean + low-cardinality integers as categorical
    cats = []
    for c in df.columns:
        if c in exclude:
            continue
        dt = df[c].dtype
        if dt == "object" or str(dt).startswith("category") or dt == "bool":
            cats.append(c)
        elif np.issubdtype(dt, np.integer):
            n_unique = df[c].nunique(dropna=False)
            if 2 <= n_unique <= 50:
                cats.append(c)
    return cats

def get_cv(y: pd.Series, groups: Optional[pd.Series], n_splits: int, seed: int):
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    else:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def param_signature(params: Dict[str, Any]) -> str:
    """Stable signature for an Optuna param dict (floats normalized)."""
    def _norm(v):
        if isinstance(v, float):
            # stable float text (avoid tiny repr diffs)
            return float(np.format_float_positional(v, unique=True, precision=12, trim='-'))
        return v
    payload = {k: _norm(v) for k, v in sorted(params.items())}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _plateau_scan(trials, min_improve: float):
    """
    Scan full trial history (COMPLETE trials) and compute:
      - plateau_best_value: last value that improved by >= min_improve
      - plateau_best_trial: its trial number
      - stale: consecutive non-improving trials since then
    """
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    complete = sorted(complete, key=lambda t: t.number)
    if not complete:
        return {"plateau_best_value": -float("inf"), "plateau_best_trial": None, "stale": 0}

    plateau_best = complete[0].value
    plateau_best_trial = complete[0].number
    stale = 0

    for t in complete[1:]:
        v = t.value
        # improvement must be >= min_improve to reset the counter
        if v >= plateau_best + float(min_improve):
            plateau_best = v
            plateau_best_trial = t.number
            stale = 0
        else:
            stale += 1

    return {
        "plateau_best_value": float(plateau_best),
        "plateau_best_trial": plateau_best_trial,
        "stale": int(stale),
    }

def get_plateau_state(study: optuna.Study, min_improve: float):
    """
    Load plateau state from user_attrs. If missing/corrupt, recompute from history and persist.
    """
    ua = study.user_attrs or {}
    if PLATEAU_KEY in ua:
        try:
            state = json.loads(ua[PLATEAU_KEY])
            # guard missing keys
            if {"plateau_best_value","plateau_best_trial","stale"} <= set(state):
                return state
        except Exception:
            pass
    # recompute from DB
    state = _plateau_scan(study.get_trials(deepcopy=False), min_improve)
    study.set_user_attr(PLATEAU_KEY, json.dumps(state))
    return state

def set_plateau_state(study: optuna.Study, state: dict):
    study.set_user_attr(PLATEAU_KEY, json.dumps(state))

def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    roc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return {"roc_auc": float(roc), "pr_auc": float(ap)}

def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Return (best_threshold, best_f1) via PR sweep."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_f1 = -1.0
    best_thr = 0.5
    for i, thr in enumerate(thresholds):
        p = precision[i+1]
        r = recall[i+1]
        f1 = 0.0 if (p + r == 0) else (2 * p * r / (p + r))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)

def best_threshold_by_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Return (best_threshold, best_accuracy) by sweeping unique probs."""
    thr_candidates = np.unique(np.concatenate(([0.0, 1.0], y_prob)))
    best_acc, best_thr = -1.0, 0.5
    for thr in thr_candidates:
        pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return float(best_thr), float(best_acc)

def compute_segment_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    segments: pd.DataFrame,
    metric: str = "accuracy",
    min_segment_size: int = 200,
) -> Dict[str, Dict[Any, float]]:
    """
    Returns: {col_name: {segment_value: threshold}}
    """
    thr_map: Dict[str, Dict[Any, float]] = {}
    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)

    chooser = best_threshold_by_accuracy if metric == "accuracy" else best_threshold_by_f1
    global_thr, _ = chooser(y_true_np, y_prob_np)
    
    for col in segments.columns:
        col_map = {}
        s = segments[col].values
        for val in pd.Series(s).unique():
            mask = (s == val)
            if mask.sum() < min_segment_size:
                col_map[_norm_key(val)] = float(global_thr)
                continue
            thr, _ = chooser(y_true_np[mask], y_prob_np[mask])
            col_map[_norm_key(val)] = float(thr)
        thr_map[col] = col_map
    return thr_map

def apply_segment_thresholds(
    prob: np.ndarray,
    segments: pd.DataFrame,
    thr_global: float,
    thr_map: Optional[Dict[str, Dict[Any, float]]] = None,
) -> np.ndarray:
    """
    If multiple segment columns are provided, applies thresholds in column order (first match wins).
    Falls back to global threshold where no rule.
    """
    if not thr_map:
        return (prob >= thr_global).astype(int)

    out = np.zeros_like(prob, dtype=int)
    used = np.zeros_like(prob, dtype=bool)

    for col in segments.columns:
        raw_rules = thr_map.get(col, {})
        # normalize rule keys once
        col_rules = { _norm_key(k): v for k, v in raw_rules.items() }
        vals = segments[col].values
        uniq_vals = pd.Series(vals).unique()
        for v in uniq_vals:
            key = _norm_key(v)
            thr = col_rules.get(key)
            if thr is None:
                continue
            m = (~used) & (vals == v)
            if not np.any(m):
                continue
            out[m] = (prob[m] >= thr).astype(int)
            used[m] = True

    out[~used] = (prob[~used] >= thr_global).astype(int)
    return out

def apply_calibration(
    method: Optional[str],
    y_prob_train: np.ndarray,
    y_true_train: np.ndarray,
    y_prob_apply: np.ndarray
):
    if method is None or method.lower() == "none":
        return y_prob_apply, None

    method = method.lower()
    if method == "platt":
        eps = 1e-6
        logits = np.log(np.clip(y_prob_train, eps, 1 - eps) / np.clip(1 - y_prob_train, eps, 1 - eps))
        X = logits.reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X, y_true_train)
        logits_apply = np.log(np.clip(y_prob_apply, eps, 1 - eps) / np.clip(1 - y_prob_apply, eps, 1 - eps))
        y_prob_cal = lr.predict_proba(logits_apply.reshape(-1, 1))[:, 1]
        return y_prob_cal, {"type": "platt", "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}
    elif method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob_train, y_true_train)
        y_prob_cal = iso.transform(y_prob_apply)
        return y_prob_cal, {
            "type": "isotonic",
            "X_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
        }
    else:
        raise ValueError("Unknown calibration method. Use: none|platt|isotonic")

def rebuild_isotonic(cal_meta: Dict[str, Any]):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.X_thresholds_ = np.array(cal_meta["X_thresholds"])
    iso.y_thresholds_ = np.array(cal_meta["y_thresholds"])
    iso.f_ = None
    return iso


# -----------------------
# Training per fold
# -----------------------

def fit_one_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    cat_idx: List[int],
    params: Dict[str, Any],
    seed: int,
    thread_count: int,
    early_stopping_rounds: int,
    log_every_iter: int,
) -> Tuple[CatBoostClassifier, np.ndarray, Dict[str, float]]:
    cb = CatBoostClassifier(
        **params,
        random_seed=seed,
        thread_count=thread_count,
        verbose=log_every_iter if log_every_iter > 0 else False,
        use_best_model=True
    )

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx if len(cat_idx) > 0 else None)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx if len(cat_idx) > 0 else None)

    cb.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=early_stopping_rounds,
    )
    
    evals = cb.eval_metrics(
        valid_pool,
        metrics=["Accuracy"],
        ntree_start=1,
        ntree_end=cb.tree_count_ + 1,
        eval_period=1
    )
    acc_series = np.array(evals["Accuracy"])
    best_iter_acc = int(acc_series.argmax()) + 1  # because ntree_start=1

    # Keep only the best-Accuracy prefix of trees
    cb.shrink(ntree_start=0, ntree_end=best_iter_acc)

    y_pred_va = cb.predict_proba(valid_pool)[:, 1]
    thr_fold, acc_fold = best_threshold_by_accuracy(y_va.values, y_pred_va)
    fold_metrics = {"accuracy_at_best_thr": acc_fold, "best_thr": thr_fold, "best_iter_acc": best_iter_acc}
    
    return cb, y_pred_va, fold_metrics


# -----------------------
# Optuna Objective
# -----------------------

class Objective:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series],
        cat_cols: List[str],
        n_splits: int,
        seed: int,
        early_stopping_rounds: int,
        thread_count: int,
        eval_metric: str = "pr_auc",
        max_iterations: int = 5000,
        progress: bool = False,
        log_every_iter: int = 0,
    ):
        self.X = X
        self.y = y
        self.groups = groups
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.seed = seed
        self.early_stopping = early_stopping_rounds
        self.thread_count = thread_count
        self.eval_metric = eval_metric
        self.max_iterations = max_iterations
        self.progress = progress
        self.log_every_iter = log_every_iter

        self.cat_idx = [self.X.columns.get_loc(c) for c in self.cat_cols] if len(cat_cols) > 0 else []

        # class ratio for scale_pos_weight suggestions
        pos = max(int((y == 1).sum()), 1)
        neg = int((y == 0).sum())
        self.neg_pos_ratio = float(neg) / float(pos)

    def __call__(self, trial: optuna.Trial) -> float:
        params = self.suggest_params(trial)
        
        # --- duplicate guard ---
        sig = param_signature(params)
        trial.set_user_attr("param_sig", sig)

        # Collect signatures of already COMPLETED trials in this study
        completed = trial.study.get_trials(
            states=(optuna.trial.TrialState.COMPLETE,),
            deepcopy=False
        )
        seen_sigs = set()
        for t in completed:
            ps = t.user_attrs.get("param_sig")
            if ps is None:
                ps = param_signature(t.params)  # for older trials without the attr
            seen_sigs.add(ps)

        if sig in seen_sigs:
            if self.progress:
                print(f"[Trial {trial.number}] DUPLICATE params (sig={sig}) → pruning fast.")
            raise optuna.TrialPruned("duplicate-params")
        # --- end duplicate guard ---
        
        cv = get_cv(self.y, self.groups, self.n_splits, self.seed)

        metrics_all = []
        for fold_idx, split in enumerate(cv.split(self.X, self.y, groups=self.groups)):
            tr_idx, va_idx = split
            X_tr, y_tr = self.X.iloc[tr_idx], self.y.iloc[tr_idx]
            X_va, y_va = self.X.iloc[va_idx], self.y.iloc[va_idx]

            _, y_pred_va, m = fit_one_fold(
                X_tr, y_tr, X_va, y_va,
                cat_idx=self.cat_idx,
                params=params,
                seed=self.seed + fold_idx,
                thread_count=self.thread_count,
                early_stopping_rounds=self.early_stopping,
                log_every_iter=self.log_every_iter,
            )
            
            acc = m["accuracy_at_best_thr"]
            target_metric = acc
            trial.report(target_metric, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
            metrics_all.append(acc)
            
        mean_value = float(np.mean(metrics_all))
        trial.set_user_attr("mean_accuracy", mean_value)
        if self.progress:
            print(f"[Trial {trial.number}] DONE  mean ACCURACY={mean_value:.5f}")
        return mean_value


    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # CatBoost search space
        depth = trial.suggest_int("depth", 5, 10)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        random_strength = trial.suggest_float("random_strength", 1.0, 100.0)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 200)
        border_count = trial.suggest_int("border_count", 32, 128)
        # one_hot_max_size = trial.suggest_int("one_hot_max_size", 2, 8)
        
        # bootstrap_type = trial.suggest_categorical("bootstrap_type", ["MVS", "Bernoulli", "Bayesian"])
        bootstrap_type = "Bernoulli"
        subsample = None
        bagging_temperature = None
        if bootstrap_type == "Bernoulli":
            subsample = trial.suggest_float("subsample", 0.85, 1.0)
        elif bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 5.0)
                    
        rsm = trial.suggest_float("rsm", 0.70, 0.95)
        
        weighting_mode = trial.suggest_categorical(
            "weighting_mode", ["none", "auto_balanced", "scale_pos_weight"]
        )
        auto_class_weights = None
        scale_pos_weight = None
        if weighting_mode == "auto_balanced":
            auto_class_weights = "Balanced"
        elif weighting_mode == "scale_pos_weight":
            base = max(self.neg_pos_ratio, 1.0)
            scale_pos_weight = trial.suggest_float("scale_pos_weight", base/3.0, base*3.0, log=True)

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": self.max_iterations,
            "od_type": "Iter",
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "learning_rate": learning_rate,
            "random_strength": random_strength,
            "min_data_in_leaf": min_data_in_leaf,
            "border_count": border_count,
            "bootstrap_type": bootstrap_type,
            "task_type": "CPU",
            "boosting_type": "Plain",
            "auto_class_weights": auto_class_weights,
            "scale_pos_weight": scale_pos_weight,
            "rsm": rsm,
        }
        if subsample is not None:
            params["subsample"] = subsample
        if bagging_temperature is not None:
            params["bagging_temperature"] = bagging_temperature
        
        return params


# -----------------------
# Main training routine
# -----------------------

def train_and_save_best(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    cat_cols: List[str],
    best_params: Dict[str, Any],
    n_splits: int,
    seed: int,
    early_stopping_rounds: int,
    thread_count: int,
    artifacts_dir: Path,
    log_every_iter: int,
    calibrate: str = "none",
    threshold_metric: str = "accuracy",
    segment_threshold_cols: Optional[List[str]] = None,
    min_segment_size: int = 200,
    n_full_seed_models: int = 0,
):
    ensure_dir(artifacts_dir)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols] if len(cat_cols) > 0 else []

    best_params["loss_function"] =  "Logloss"
    best_params["od_type"] =  "LogIterloss"
    best_params["boosting_type"] =  "Plain"
    best_params["task_type"] =  "CPU"
    best_params['iterations'] = 5000
    if best_params.get('scale_pos_weight') is None :
        best_params['auto_class_weights'] = 'Balanced'
    if "subsample" in best_params and best_params.get("bootstrap_type") is None:
        best_params["bootstrap_type"] = "Bernoulli"

    # 1) OOF predictions with best params
    cv = get_cv(y, groups, n_splits, seed)
    oof = np.zeros(len(y), dtype=float)
    fold_metrics = []
    fold_models = []
    
    for fold_idx, split in enumerate(cv.split(X, y, groups=groups)):
        tr_idx, va_idx = split
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model, y_pred_va, m = fit_one_fold(
            X_tr, y_tr, X_va, y_va,
            cat_idx=cat_idx,
            params=best_params,
            seed=seed + fold_idx,
            thread_count=thread_count,
            early_stopping_rounds=early_stopping_rounds,
            log_every_iter=log_every_iter,
        )
        oof[va_idx] = y_pred_va
        fold_metrics.append(m)
        fold_models.append(model)
        
        # Save fold model
        fold_path = artifacts_dir / f"model_fold{fold_idx}.cbm"
        model.save_model(str(fold_path))
        
        print(f"Fold {fold_idx} accuracy: {m['accuracy_at_best_thr']}")

    # 2) Pick threshold on OOF
    if threshold_metric == "accuracy":
        thr, best_score = best_threshold_by_accuracy(y.values, oof)
    elif threshold_metric == "f1":
        thr, best_score = best_threshold_by_f1(y.values, oof)
    else:
        raise ValueError("threshold_metric must be 'accuracy' or 'f1'")

    base_metrics = metrics_from_probs(y.values, oof)

    # 3) Optional calibration on OOF
    cal_meta = None
    oof_cal = None
    thr_cal = None
    best_score_cal = None
    cal_metrics = None
    if calibrate.lower() in {"platt", "isotonic"}:
        oof_cal, cal_meta = apply_calibration(calibrate, oof, y.values, oof)
        if threshold_metric == "accuracy":
            thr_cal, best_score_cal = best_threshold_by_accuracy(y.values, oof_cal)
        else:
            thr_cal, best_score_cal = best_threshold_by_f1(y.values, oof_cal)
        cal_metrics = metrics_from_probs(y.values, oof_cal)

    # 4) Segmented thresholds on OOF
    seg_thr_map = None
    seg_thr_map_cal = None
    if segment_threshold_cols:
        seg_df = X[segment_threshold_cols].copy()
        seg_thr_map = compute_segment_thresholds(
            y_true=y.values, y_prob=oof, segments=seg_df,
            metric=threshold_metric, min_segment_size=min_segment_size
        )
        if oof_cal is not None:
            seg_thr_map_cal = compute_segment_thresholds(
                y_true=y.values, y_prob=oof_cal, segments=seg_df,
                metric=threshold_metric, min_segment_size=min_segment_size
            )

    # 5) Save OOF + metrics
    pd.DataFrame({"oof_prob": oof, "y": y.values}).to_parquet(artifacts_dir / "oof.parquet", index=False)
    if oof_cal is not None:
        pd.DataFrame({"oof_prob_cal": oof_cal, "y": y.values}).to_parquet(artifacts_dir / "oof_calibrated.parquet", index=False)

    out = {
        "cv_metrics_mean": {
            "accuracy": float(np.mean([m.get("accuracy_at_best_thr", np.nan) for m in fold_metrics if "accuracy_at_best_thr" in m])),
        },
        "oof_metrics": base_metrics,
        "threshold_metric": threshold_metric,
        "best_threshold": thr,
        "best_threshold_value": best_score,
        "calibration": cal_meta,
        "oof_metrics_calibrated": cal_metrics,
        "best_threshold_calibrated": thr_cal,
        "best_threshold_calibrated_value": best_score_cal,
        "segment_threshold_cols": segment_threshold_cols,
        "segment_thresholds": seg_thr_map,
        "segment_thresholds_calibrated": seg_thr_map_cal,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    # 6) Retrain final single model on 100% data
    full_model = CatBoostClassifier(
        **best_params,
        random_seed=seed,
        thread_count=thread_count,
        verbose=False,
        use_best_model=False,
    )
    full_pool = Pool(X, y, cat_features=cat_idx if len(cat_idx) > 0 else None)
    full_model.fit(full_pool)
    full_model.save_model(str(artifacts_dir / "model_final_single.cbm"))

    # 7) Extra full-data seed ensemble
    extra_models = 0
    for i in range(n_full_seed_models):
        m = CatBoostClassifier(
            **best_params,
            random_seed=seed + 1000 + i,
            thread_count=thread_count,
            verbose=False,
            use_best_model=False,
        )
        m.fit(full_pool)
        m.save_model(str(artifacts_dir / f"model_fullseed{i}.cbm"))
        extra_models += 1

    # 7) Save ensemble info
    (artifacts_dir / "ensemble_info.json").write_text(json.dumps({
        "n_fold_models": len(fold_models),
        "n_full_seed_models": extra_models,
        "note": "During inference, we average probs from model_fold*.cbm and model_fullseed*.cbm if present.",
    }, indent=2))

    # 8) Save inference config for submission
    (artifacts_dir / "inference_config.json").write_text(json.dumps({
        "best_params": best_params,
        "threshold_metric": threshold_metric,
        "threshold": thr,
        "threshold_calibrated": thr_cal,
        "segment_threshold_cols": segment_threshold_cols,
        "segment_thresholds": seg_thr_map,
        "segment_thresholds_calibrated": seg_thr_map_cal,
        "calibration": cal_meta,
        "cat_cols": cat_cols,
        "columns_order": X.columns.tolist(),
    }, indent=2))

    # 9) Feature importance
    try:
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": full_model.get_feature_importance(full_pool, type="FeatureImportance")
        }).sort_values("importance", ascending=False)
        fi.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    except Exception:
        pass

    return out


# -----------------------
# Inference helpers
# -----------------------

def predict_with_model_files(
    df: pd.DataFrame,
    model_paths: List[Path],
    cat_cols: List[str],
    calibration: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    cat_idx = [df.columns.get_loc(c) for c in cat_cols] if len(cat_cols) > 0 else []
    pool = Pool(df, cat_features=cat_idx if len(cat_idx) > 0 else None)

    preds = []
    for mp in model_paths:
        m = CatBoostClassifier()
        m.load_model(str(mp))
        preds.append(m.predict_proba(pool)[:, 1])
    prob = np.mean(np.vstack(preds), axis=0)

    # Apply calibration if provided
    if calibration is not None:
        if calibration.get("type") == "platt":
            coef = np.array(calibration["coef"]).ravel()
            intercept = np.array(calibration["intercept"]).ravel()
            eps = 1e-6
            logits = np.log(np.clip(prob, eps, 1 - eps) / np.clip(1 - prob, eps, 1 - eps))
            z = logits * coef[0] + intercept[0]
            prob = 1 / (1 + np.exp(-z))
        elif calibration.get("type") == "isotonic":
            iso = rebuild_isotonic(calibration)
            prob = iso.transform(prob)
    return prob

def make_submission(
    test_df: pd.DataFrame,
    id_col: str,
    model_dir: Path,
    calibrated: Optional[bool] = None,
    submission_path: Optional[Path] = None,
) -> Path:
    """
    Loads fold models + inference_config, predicts test probabilities,
    applies (optional) calibration and global/segment thresholds, writes CSV.
    """
    inf = json.loads((model_dir / "inference_config.json").read_text())

    # Decide whether to use calibration & which thresholds to use
    has_cal = inf.get("calibration") is not None
    use_cal = has_cal if calibrated is None else calibrated

    # Align columns
    cols_order = inf["columns_order"]
    missing = [c for c in cols_order if c not in test_df.columns]
    assert not missing, f"Test is missing required columns: {missing}"
    X_te = test_df[cols_order].copy()

    # Fill categoricals like training
    for c in inf.get("cat_cols", []):
        if c in X_te.columns:
            X_te[c] = X_te[c].astype("object").fillna("__NA__")

    # Predict with fold ensemble
    model_paths = sorted(model_dir.glob("model_fold*.cbm")) + sorted(model_dir.glob("model_fullseed*.cbm"))
    assert len(model_paths) > 0, "No fold models found under artifacts."
    prob = predict_with_model_files(
        df=X_te,
        model_paths=model_paths,
        cat_cols=inf.get("cat_cols", []),
        calibration=inf["calibration"] if use_cal else None,
    )

    # Thresholds (global + optional segmented)
    thr_global = inf["threshold_calibrated"] if (use_cal and inf.get("threshold_calibrated") is not None) else inf["threshold"]
    seg_cols = inf.get("segment_threshold_cols") or []
    thr_map = inf.get("segment_thresholds_calibrated") if (use_cal and inf.get("segment_thresholds_calibrated")) else inf.get("segment_thresholds")

    if seg_cols:
        for c in seg_cols:
            assert c in test_df.columns, f"Segment column '{c}' missing in test."
        seg_df = test_df[seg_cols].copy()
        pred = apply_segment_thresholds(prob, seg_df, thr_global, thr_map)
    else:
        pred = (prob >= thr_global).astype(int)

    sub = pd.DataFrame({id_col: test_df[id_col].values, "condition": pred})
    out_path = submission_path if submission_path else (model_dir / "submission.csv")
    out_path = Path(out_path)
    sub.to_csv(out_path, index=False)
    return out_path


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV/Parquet with training data.")
    parser.add_argument("--target", type=str, required=True, help="Target column (binary: 0/1).")
    parser.add_argument("--group-col", type=str, default=None, help="Column for GroupKFold (e.g., seller_id).")
    parser.add_argument("--id-cols", type=str, nargs="*", default=[], help="Columns to exclude from modeling (ids).")
    parser.add_argument("--categoricals", type=str, nargs="*", default=None, help="Optional explicit categorical column names.")
    parser.add_argument("--drop-cols", type=str, nargs="*", default=[], help="Columns to drop before modeling.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=int(time.time()) % 10_000_000)
    parser.add_argument("--timeout", type=int, default=None, help="Seconds to stop Optuna early (optional).")
    parser.add_argument("--study-name", type=str, default="catboost_meli")
    parser.add_argument("--study-storage", type=str, default="sqlite:///optuna_study.db", help="Optuna storage URI.")
    parser.add_argument("--resume", action="store_true", help="Resume study if exists.")
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--thread-count", type=int, default=-1, help="CatBoost thread_count (-1 uses all).")
    parser.add_argument("--eval-metric", type=str, choices=["accuracy"], default="accuracy")
    parser.add_argument("--calibrate", type=str, choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--progress", action="store_true", help="Show Optuna progress bar and per-trial logs.")
    parser.add_argument("--log-every-iter", type=int, default=0, help="If >0, CatBoost prints metrics every N iters.")

    # Tuning vs final CV
    parser.add_argument("--tune-n-splits", type=int, default=None, help="If set, folds during tuning (overrides --n-splits for tuning only).")
    parser.add_argument("--threshold-metric", type=str, choices=["accuracy", "f1"], default="accuracy", help="Metric to choose threshold on OOF.")
    parser.add_argument("--segment-threshold-cols", type=str, nargs="*", default=[], help="Columns to compute per-segment thresholds on OOF.")
    parser.add_argument("--min-segment-size", type=int, default=200, help="Minimum OOF count in a segment to learn its own threshold.")
    parser.add_argument("--patience-trials", type=int, default=None, help="Stop the study if no improvement after this many COMPLETED trials.")
    parser.add_argument("--min-improve", type=float, default=1e-4, help="Minimum gain to count as improvement (ignore CV noise).")

    # Kaggle submission
    parser.add_argument("--test-data", type=str, default=None, help="Optional path to test CSV/Parquet for submission.")
    parser.add_argument("--id-col", type=str, default=None, help="ID column in test for submission CSV.")
    parser.add_argument("--submission-path", type=str, default=None, help="Where to save submission CSV (if --test-data is given).")
    
    # Extra seed ensemble
    parser.add_argument("--n-full-seed-models", type=int, default=0, help="Train K extra full-data models with different seeds.")
    
    parser.add_argument('--just-train-save-best', action='store_true', help='Train and save with best params found in the study.')

    args = parser.parse_args()

    set_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    ensure_dir(artifacts)

    # Load training data
    p = Path(args.data)
    if p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    assert args.target in df.columns, f"Target {args.target} not found."

    # Groups for GroupKFold
    if args.group_col is not None:
        assert args.group_col in df.columns, f"Group col {args.group_col} not found."
        df[args.group_col] = df[args.group_col].astype("object")
    groups = df[args.group_col] if args.group_col else None

    # Prepare X/y
    drop_cols = set(args.drop_cols)
    y = df[args.target].eq('new').astype(int)

    exclude = set([args.target]) | set(args.id_cols) | drop_cols
    if args.group_col:
        exclude.add(args.group_col)

    X = df.drop(columns=list(exclude), errors="ignore").copy()

    # Detect categoricals
    cat_cols = detect_cat_cols(X, exclude=[])
    if args.categoricals is not None:
        for c in args.categoricals:
            assert c in X.columns, f"Categorical {c} not found in features."
        # de-dup while preserving order
        cat_cols = list(dict.fromkeys(cat_cols + list(args.categoricals)))

    # Clean NA
    X = X.loc[:, X.notna().any(axis=0)].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("object")

    # Create Optuna study
    sampler = TPESampler(
        seed=int(time.time()),
        multivariate=True,
        group=False,         
        consider_prior=True,
        n_startup_trials=10  
    )
    pruner = MedianPruner(n_warmup_steps=2)

    if args.resume:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=args.study_storage,
                sampler=sampler,
                pruner=pruner,
            )
            print(f"[Optuna] Resuming existing study '{args.study_name}'.")
        except KeyError:
            print(f"[Optuna] Study '{args.study_name}' not found. Creating a new one.")
            study = optuna.create_study(
                direction="maximize",
                study_name=args.study_name,
                storage=args.study_storage,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
        
    else:
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.study_storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

    if not args.just_train_save_best :
        # Initialize / resume plateau state from DB and show it
        state0 = get_plateau_state(study, args.min_improve)
        if args.progress:
            print(f"[Plateau] resume: stale={state0['stale']}  best={state0['plateau_best_value']:.6f}  best_trial={state0['plateau_best_trial']}")

        objective = Objective(
            X=X, y=y, groups=groups, cat_cols=cat_cols,
            n_splits=args.tune_n_splits if args.tune_n_splits else args.n_splits,
            seed=args.seed,
            early_stopping_rounds=args.early_stopping_rounds,
            thread_count=args.thread_count,
            eval_metric=args.eval_metric,
            max_iterations=args.max_iterations,
            progress=args.progress,
            log_every_iter=args.log_every_iter,
        )

        def _cb_plateau(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial):
            # Load current state (recomputed if missing)
            state = get_plateau_state(study_obj, args.min_improve)
            best = state["plateau_best_value"]
            stale = int(state["stale"])

            # Current trial value
            val = trial.value
            if val is None or not math.isfinite(val):
                return

            improved = (val >= best + float(args.min_improve))
            if improved:
                best = float(val)
                state["plateau_best_value"] = best
                state["plateau_best_trial"] = trial.number
                stale = 0
            else:
                stale += 1

            state["stale"] = stale
            set_plateau_state(study_obj, state)

            if args.progress:
                print(f"[Plateau] stale={stale}/{args.patience_trials}  best={best:.6f}  best_trial={state['plateau_best_trial']}")

            # Optional early stop on plateau
            if args.patience_trials and stale >= args.patience_trials:
                if args.progress:
                    print("[Plateau] patience reached; requesting study stop.")
                study_obj.stop()

        start_time = time.time()
        trial_times = []

        def _cb_progress(study, trial):
            if not args.progress:
                return
            dur = trial.duration.total_seconds() if trial.duration else 0.0
            trial_times.append(dur)
            avg = sum(trial_times) / max(1, len(trial_times))
            elapsed_min = (time.time() - start_time) / 60.0

            val = trial.value
            val_str = f"{val:.6f}" if isinstance(val, (int, float)) and math.isfinite(val) else "NA"

            metric_val = trial.user_attrs.get(f"mean_{args.eval_metric}")
            metric_str = (f"{metric_val:.6f}" if isinstance(metric_val, (int, float)) and
                        math.isfinite(metric_val) else "NA")

            print(f"[Trial {trial.number}] state={trial.state.name} value={val_str} "
                f"{args.eval_metric.upper()}={metric_str} dur={dur:.1f}s "
                f"avg={avg:.1f}s elapsed~{elapsed_min:.1f} min")

        
        # Build callbacks list
        callbacks = []
        if args.progress:
            callbacks.append(_cb_progress)
        callbacks.append(_cb_plateau)  # safe even if patience is 0 (no-op)
        
        try:
            # --- Run optimization ---
            study.optimize(
                objective,
                n_trials=None,                # run until timeout or plateau stop
                timeout=args.timeout,         # you can pass --timeout from GH Actions, or leave None
                gc_after_trial=True,
                show_progress_bar=args.progress,
                callbacks=callbacks,
            )
        except KeyboardInterrupt:
            print("[interrupt] KeyboardInterrupt received. Proceeding to finalize with best-so-far trial...")

    if study.best_trial is None or study.best_value is None:
        raise RuntimeError(
            "Study ended without a completed trial. "
            "Consider lowering --early-stopping-rounds, checking data, or setting a longer timeout."
        )

    best_params = study.best_trial.params
    # Normalize best params for CatBoost (explicit Nones)
    if "weighting_mode" in best_params:
        wm = best_params["weighting_mode"]
        if wm == "none":
            best_params["auto_class_weights"] = None
            best_params["scale_pos_weight"] = None
        elif wm == "auto_balanced":
            best_params["auto_class_weights"] = "Balanced"
            best_params["scale_pos_weight"] = None
        elif wm == "scale_pos_weight":
            best_params["auto_class_weights"] = None
        del best_params["weighting_mode"]

    # Persist study info + best params
    (artifacts / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (artifacts / "study_best.json").write_text(json.dumps({
        "value": study.best_value,
        "number": study.best_trial.number,
        "user_attrs": study.best_trial.user_attrs,
    }, indent=2))

    # Train finals + export artifacts
    summary = train_and_save_best(
        X=X, y=y, groups=groups, cat_cols=cat_cols,
        best_params=best_params, n_splits=args.n_splits, seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
        thread_count=args.thread_count,
        artifacts_dir=artifacts,
        calibrate=args.calibrate,
        log_every_iter=args.log_every_iter,
        threshold_metric=args.threshold_metric,
        segment_threshold_cols=args.segment_threshold_cols if args.segment_threshold_cols else None,
        min_segment_size=args.min_segment_size,
        n_full_seed_models=args.n_full_seed_models,
    )

    print("\n=== Done (Training) ===")
    print("CV mean accuracy:", summary["cv_metrics_mean"].get("accuracy"))
    print(f"Threshold metric: {summary['threshold_metric']}")
    print("Best threshold:", summary["best_threshold"], "Value:", summary["best_threshold_value"])
    if summary["oof_metrics_calibrated"] is not None:
        print("OOF metrics (calibrated):", summary["oof_metrics_calibrated"])
        print("Best threshold (calibrated):", summary["best_threshold_calibrated"], "Value:", summary["best_threshold_calibrated_value"])
    if summary.get("segment_threshold_cols"):
        print("Segment thresholds learned for:", summary["segment_threshold_cols"])
    print(f"Artifacts saved under: {artifacts.resolve()}")

    # Optional submission
    if args.test_data:
        assert args.id_col is not None, "--id-col is required when --test-data is provided."
        tp = Path(args.test_data)
        if tp.suffix.lower() in [".parquet", ".pq"]:
            test_df = pd.read_parquet(tp)
        else:
            test_df = pd.read_csv(tp)

        out_path = args.submission_path if args.submission_path else str(artifacts / "submission.csv")
        out_path = Path(out_path)
        out = make_submission(
            test_df=test_df,
            id_col=args.id_col,
            model_dir=artifacts,
            calibrated=None,  # auto: use if available
            submission_path=out_path,
        )
        print(f"Submission saved to: {out.resolve()}")

if __name__ == "__main__":
    main()

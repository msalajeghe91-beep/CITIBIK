#!/usr/bin/env python3
"""
train_predict.py

Robust station-month demand forecasting for Citi Bike.

INPUT (from existing pipeline):
  summaries/<RUN_TAG>/citibike_station_exposure.csv

EXPECTED columns (minimum):
  - mode
  - year
  - month
  - start_station_id (or station_id)
  - trips (or y_trips)

OPTIONAL columns (used if present):
  - start_station_name
  - station_lat, station_lng
  - start_trips, end_trips, touchpoints

OUTPUT (default to <in-dir>/ml):
  - metrics.json
  - model_card.md
  - predictions_demand_station_month_<mode>.csv
  - predictions_demand_station_month.csv (combined)
  - feature_importance_<mode>.csv (if LightGBM)

Key features:
- Works for both dense (all months) and sparse month selections.
- Safe time splitting by LABEL month (prevents leakage).
- Degrades split automatically if not enough label months (never crashes).
- Can write predictions for 'test' only, 'val+test', or 'all' labeled months.
- Can generate FUTURE forecasts (beyond data) for arbitrary requested months next year
  using iterative 1-month-ahead rolling predictions (no extra datasets required).

Examples:
  # Standard evaluation (default): last month as test, previous as val
  python scripts/train_predict.py --in-dir summaries/<RUN_TAG> --model lightgbm

  # Write val+test predictions (so you see multiple months)
  python scripts/train_predict.py --in-dir summaries/<RUN_TAG> --write-splits valtest

  # More months in test for presentation
  python scripts/train_predict.py --in-dir summaries/<RUN_TAG> --val-months 2 --test-months 3 --write-splits test

  # One-year-ahead seasonal months (e.g., next year Mar/Apr/May)
  python scripts/train_predict.py --in-dir summaries/<RUN_TAG> \
      --predict-future --future-years-ahead 1 --future-months 3 4 5

Notes on sparse month runs:
- If you only provide a few months, lag/rolling features become weak.
- This script automatically adds fallback signals and avoids hard failures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

# Optional LightGBM
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None


# ----------------------------
# Time helpers
# ----------------------------
def ym_to_index(year: int, month: int) -> int:
    # month is 1..12
    return int(year) * 12 + int(month)


def index_to_ym(idx: int) -> Tuple[int, int]:
    # inverse of year*12+month with month 1..12
    year = idx // 12
    month = idx % 12
    if month == 0:
        year -= 1
        month = 12
    return int(year), int(month)


def add_months(year: int, month: int, k: int) -> Tuple[int, int]:
    base = ym_to_index(year, month)
    y2, m2 = index_to_ym(base + k)
    return y2, m2


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


# ----------------------------
# Feature engineering
# ----------------------------
def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    m = df["month"].astype(int)
    df["month_sin"] = np.sin(2.0 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * m / 12.0)
    return df


def _group_sort(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return df.sort_values(group_cols + ["date_index"]).copy()


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    y_col: str,
    lags: Tuple[int, ...] = (1, 2, 3, 12),
    roll_windows: Tuple[int, ...] = (3, 6),
) -> pd.DataFrame:
    """
    Adds lag and rolling stats using calendar-month ordering (date_index).
    """
    df = _group_sort(df, group_cols)
    g = df.groupby(group_cols, group_keys=False)

    for k in lags:
        df[f"{y_col}_lag_{k}"] = g[y_col].apply(lambda s: s.shift(k))

    for w in roll_windows:
        # past-only rolling: shift(1) then rolling
        df[f"{y_col}_rollmean_{w}"] = g[y_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"{y_col}_rollstd_{w}"] = g[y_col].apply(lambda s: s.shift(1).rolling(w, min_periods=2).std())

    return df


def add_optional_component_lags(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    df = _group_sort(df, group_cols)
    g = df.groupby(group_cols, group_keys=False)
    for col in ["start_trips", "end_trips", "touchpoints"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_lag_1"] = g[col].apply(lambda s: s.shift(1))
    return df


def add_station_level_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds robust fallback signals that work even when months are sparse:
    - station_mean
    - station_month_mean (mean demand for that station for that month-of-year)
    - station_last_seen (indicator)
    """
    df["y_trips"] = pd.to_numeric(df["y_trips"], errors="coerce")

    df["station_mean"] = df.groupby(["mode", "station_id"])["y_trips"].transform("mean")
    df["station_month_mean"] = df.groupby(["mode", "station_id", "month"])["y_trips"].transform("mean")
    df["has_y_trips"] = df["y_trips"].notna().astype(int)

    # Fill station_month_mean fallback to station_mean if missing
    df["station_month_mean"] = df["station_month_mean"].where(df["station_month_mean"].notna(), df["station_mean"])
    return df


# ----------------------------
# Splitting
# ----------------------------
@dataclass
class SplitPlan:
    split_mode: str  # "normal" or "degraded"
    val_months_req: int
    test_months_req: int
    val_months_eff: int
    test_months_eff: int
    label_months_available: int
    note: str


def make_split_plan(label_months_sorted: np.ndarray, val_months: int, test_months: int) -> SplitPlan:
    """
    Normal requires at least val+test+2 label months (some buffer for training).
    If not enough, degrade gracefully:
      - keep test=1 if possible
      - reduce val to 0 if needed
      - if still not possible, test=0 (train only)
    """
    n = int(len(label_months_sorted))
    req = int(val_months + test_months + 2)

    if n >= req:
        return SplitPlan(
            split_mode="normal",
            val_months_req=val_months,
            test_months_req=test_months,
            val_months_eff=val_months,
            test_months_eff=test_months,
            label_months_available=n,
            note="normal time split",
        )

    # degraded
    if n >= 2:
        # keep test=1
        test_eff = 1
        # allow val if possible
        # we want at least 1 train month before val/test
        val_eff = 1 if n >= 3 else 0
        return SplitPlan(
            split_mode="degraded",
            val_months_req=val_months,
            test_months_req=test_months,
            val_months_eff=val_eff,
            test_months_eff=test_eff,
            label_months_available=n,
            note="degraded: train on all before last; test last; val optional",
        )

    # too small to test
    return SplitPlan(
        split_mode="degraded",
        val_months_req=val_months,
        test_months_req=test_months,
        val_months_eff=0,
        test_months_eff=0,
        label_months_available=n,
        note="degraded: not enough label months to create test; train only",
    )


def split_by_label_month(df: pd.DataFrame, label_months_sorted: np.ndarray, plan: SplitPlan) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split based on label_date_index.
    """
    if plan.test_months_eff == 0:
        # train only
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    test_start = label_months_sorted[-plan.test_months_eff]
    if plan.val_months_eff > 0:
        val_start = label_months_sorted[-(plan.test_months_eff + plan.val_months_eff)]
        train = df[df["label_date_index"] < val_start].copy()
        val = df[(df["label_date_index"] >= val_start) & (df["label_date_index"] < test_start)].copy()
        test = df[df["label_date_index"] >= test_start].copy()
        return train, val, test
    else:
        train = df[df["label_date_index"] < test_start].copy()
        val = df.iloc[0:0].copy()
        test = df[df["label_date_index"] >= test_start].copy()
        return train, val, test


# ----------------------------
# Models
# ----------------------------
def make_lightgbm():
    # Conservative defaults; stable
    return lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )


def make_histgb():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=800,
        random_state=42,
    )


def fit_model(model, X_train, y_train, X_val=None, y_val=None, model_used: str = "histgb"):
    """
    Fit with LightGBM callbacks (avoids verbose kwarg incompatibilities).
    """
    if model_used == "lightgbm" and lgb is not None and X_val is not None and y_val is not None and len(X_val) > 0:
        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train)
    return model


# ----------------------------
# Backtest
# ----------------------------
def rolling_backtest(
    df: pd.DataFrame,
    model_factory,
    feature_cols: List[str],
    target_col: str,
    label_months_sorted: np.ndarray,
    backtest_months: int,
    log1p: bool,
) -> Dict[str, float]:
    if backtest_months <= 0:
        return {}

    n = len(label_months_sorted)
    if n < backtest_months + 3:
        return {"backtest_skipped": 1.0, "backtest_reason": float(n)}

    test_months = label_months_sorted[-backtest_months:]
    maes, smapes = [], []

    for L in test_months:
        train = df[df["label_date_index"] < L].copy()
        test = df[df["label_date_index"] == L].copy()
        if train.empty or test.empty:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].astype(float).values
        X_test = test[feature_cols]
        y_test = test[target_col].astype(float).values

        y_train_t = np.log1p(np.maximum(y_train, 0)) if log1p else y_train

        model = model_factory()
        model.fit(X_train, y_train_t)
        y_pred_t = model.predict(X_test)
        y_pred = np.expm1(y_pred_t) if log1p else y_pred_t
        y_pred = np.maximum(y_pred, 0)

        maes.append(mean_absolute_error(y_test, y_pred))
        smapes.append(safe_smape(y_test, y_pred))

    if not maes:
        return {"backtest_failed": 1.0}

    return {
        "backtest_months": float(backtest_months),
        "backtest_mae_mean": float(np.mean(maes)),
        "backtest_mae_std": float(np.std(maes)),
        "backtest_smape_mean": float(np.mean(smapes)),
        "backtest_smape_std": float(np.std(smapes)),
    }


# ----------------------------
# Train + predict for one mode (evaluation)
# ----------------------------
def train_predict_one_mode_eval(
    df_mode: pd.DataFrame,
    mode: str,
    out_dir: Path,
    model_name: str,
    horizon_months: int,
    val_months: int,
    test_months: int,
    backtest_months: int,
    log1p: bool,
    write_splits: str,
) -> Dict[str, object]:
    df = df_mode.copy()

    group_cols = ["mode", "station_id"]
    df = df.sort_values(group_cols + ["date_index"]).copy()

    # label month index = feature month + horizon
    df["label_date_index"] = df["date_index"] + int(horizon_months)

    # label = y_trips shifted by -horizon within station
    g = df.groupby(group_cols, group_keys=False)
    df["y_label"] = g["y_trips"].apply(lambda s: s.shift(-horizon_months))

    # Features
    df = add_seasonality_features(df)
    df = add_station_level_fallbacks(df)
    df = add_lag_and_rolling_features(df, group_cols=group_cols, y_col="y_trips")
    df = add_optional_component_lags(df, group_cols=group_cols)

    # Feature list (robust)
    feature_cols = [
        "month_sin",
        "month_cos",
        "station_mean",
        "station_month_mean",
        "has_y_trips",
        "y_trips_lag_1",
        "y_trips_lag_2",
        "y_trips_lag_3",
        "y_trips_lag_12",
        "y_trips_rollmean_3",
        "y_trips_rollmean_6",
        "y_trips_rollstd_3",
        "y_trips_rollstd_6",
    ]

    for c in ["station_lat", "station_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            feature_cols.append(c)

    for c in ["start_trips_lag_1", "end_trips_lag_1", "touchpoints_lag_1"]:
        if c in df.columns:
            feature_cols.append(c)

    # Keep rows with label
    data = df.dropna(subset=["y_label"]).copy()

    # Baselines for evaluation
    data["baseline_lag1"] = data["y_trips_lag_1"]
    data["baseline_lag12"] = data["y_trips_lag_12"].where(~data["y_trips_lag_12"].isna(), data["baseline_lag1"])

    # Clean feature matrix
    X_all = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data.loc[:, feature_cols] = X_all

    label_months_sorted = np.sort(data["label_date_index"].unique())
    plan = make_split_plan(label_months_sorted, val_months=val_months, test_months=test_months)
    train_df, val_df, test_df = split_by_label_month(data, label_months_sorted, plan)

    # Choose model
    if model_name == "lightgbm" and lgb is not None:
        model_factory = make_lightgbm
        model_used = "lightgbm"
    elif model_name == "lightgbm" and lgb is None:
        model_factory = make_histgb
        model_used = "histgb_fallback"
    else:
        model_factory = make_histgb
        model_used = "histgb"

    # Train
    X_train = train_df[feature_cols]
    y_train = train_df["y_label"].astype(float).values
    X_val = val_df[feature_cols]
    y_val = val_df["y_label"].astype(float).values
    X_test = test_df[feature_cols] if len(test_df) > 0 else test_df
    y_test = test_df["y_label"].astype(float).values if len(test_df) > 0 else np.array([])

    y_train_t = np.log1p(np.maximum(y_train, 0)) if log1p else y_train
    y_val_t = np.log1p(np.maximum(y_val, 0)) if log1p else y_val

    model = model_factory()
    model = fit_model(model, X_train, y_train_t, X_val=X_val, y_val=y_val_t, model_used=("lightgbm" if model_used == "lightgbm" else "histgb"))

    # If no test split possible, still write metrics and stop
    if plan.test_months_eff == 0 or len(test_df) == 0:
        metrics = {
            "mode": mode,
            "model": model_used,
            "horizon_months": int(horizon_months),
            "split_mode": plan.split_mode,
            "degraded_plan": plan.note,
            "label_months_available": int(plan.label_months_available),
            "val_months_requested": int(plan.val_months_req),
            "test_months_requested": int(plan.test_months_req),
            "effective_val_months": int(plan.val_months_eff),
            "effective_test_months": int(plan.test_months_eff),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "stations_train": int(train_df["station_id"].nunique()),
            "stations_test": int(test_df["station_id"].nunique()),
            "note": "No test split possible with given data/horizon; trained model only.",
        }
        return {"metrics": metrics, "pred_path": None, "feature_cols": feature_cols, "model_used": model_used}

    # Predict on requested splits
    def build_pred_frame(sub: pd.DataFrame, split_name: str) -> pd.DataFrame:
        X = sub[feature_cols]
        y_true = sub["y_label"].astype(float).values
        y_pred_t = model.predict(X)
        y_pred = np.expm1(y_pred_t) if log1p else y_pred_t
        y_pred = np.maximum(y_pred, 0)

        base1 = sub["baseline_lag1"].fillna(0).astype(float).values
        base12 = sub["baseline_lag12"].fillna(0).astype(float).values

        pred = sub[
            ["mode", "station_id", "start_station_name", "station_lng", "station_lat", "year", "month", "date_index", "label_date_index"]
        ].copy()
        pred = pred.rename(columns={"year": "feature_year", "month": "feature_month"})

        label_ym = pred["label_date_index"].apply(lambda idx: pd.Series(index_to_ym(int(idx)), index=["label_year", "label_month"]))
        pred = pd.concat([pred.reset_index(drop=True), label_ym.reset_index(drop=True)], axis=1)

        pred["y_true_next_month"] = y_true
        pred["y_pred_next_month"] = y_pred
        pred["baseline_lag1_pred"] = base1
        pred["baseline_lag12_pred"] = base12
        pred["split"] = split_name
        return pred

    frames = []
    if write_splits in ("all", "valtest"):
        if len(val_df) > 0:
            frames.append(build_pred_frame(val_df, "val"))
    if write_splits in ("all", "valtest", "test"):
        frames.append(build_pred_frame(test_df, "test"))
    if write_splits == "all":
        # optionally also write train (can be huge)
        frames.append(build_pred_frame(train_df, "train"))

    pred_out = pd.concat(frames, ignore_index=True)

    # Metrics (computed on test only)
    # (If you want, you can also compute on val block separately.)
    y_pred_test = pred_out.loc[pred_out["split"] == "test", "y_pred_next_month"].astype(float).values
    y_true_test = pred_out.loc[pred_out["split"] == "test", "y_true_next_month"].astype(float).values
    base1_test = pred_out.loc[pred_out["split"] == "test", "baseline_lag1_pred"].astype(float).values
    base12_test = pred_out.loc[pred_out["split"] == "test", "baseline_lag12_pred"].astype(float).values

    mae_b1 = mean_absolute_error(y_true_test, base1_test)
    mae_b12 = mean_absolute_error(y_true_test, base12_test)
    mae_m = mean_absolute_error(y_true_test, y_pred_test)

    metrics = {
        "mode": mode,
        "model": model_used,
        "horizon_months": int(horizon_months),
        "split_mode": plan.split_mode,
        "degraded_plan": plan.note,
        "label_months_available": int(plan.label_months_available),
        "val_months_requested": int(plan.val_months_req),
        "test_months_requested": int(plan.test_months_req),
        "effective_val_months": int(plan.val_months_eff),
        "effective_test_months": int(plan.test_months_eff),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "stations_train": int(train_df["station_id"].nunique()),
        "stations_test": int(test_df["station_id"].nunique()),
        "mae_baseline_lag1": float(mae_b1),
        "smape_baseline_lag1": float(safe_smape(y_true_test, base1_test)),
        "mae_baseline_lag12": float(mae_b12),
        "smape_baseline_lag12": float(safe_smape(y_true_test, base12_test)),
        "mae_model": float(mae_m),
        "smape_model": float(safe_smape(y_true_test, y_pred_test)),
        "mae_improvement_vs_lag1_pct": float(100.0 * (mae_b1 - mae_m) / max(mae_b1, 1e-9)),
        "mae_improvement_vs_lag12_pct": float(100.0 * (mae_b12 - mae_m) / max(mae_b12, 1e-9)),
    }

    # Backtest stability metrics
    bt = rolling_backtest(
        data,
        model_factory=model_factory,
        feature_cols=feature_cols,
        target_col="y_label",
        label_months_sorted=label_months_sorted,
        backtest_months=backtest_months,
        log1p=log1p,
    )
    metrics.update(bt)

    # Write per-mode predictions
    out_mode = out_dir / f"predictions_demand_station_month_{mode}.csv"
    pred_out.to_csv(out_mode, index=False)

    # Feature importance
    fi_path = out_dir / f"feature_importance_{mode}.csv"
    try:
        if model_used == "lightgbm":
            imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
            imp = imp.sort_values("importance", ascending=False)
            imp.to_csv(fi_path, index=False)
    except Exception:
        pass

    return {"metrics": metrics, "pred_path": str(out_mode), "feature_cols": feature_cols, "model_used": model_used}


# ----------------------------
# FUTURE forecasting (iterative 1-step rolling)
# ----------------------------
def train_and_forecast_future(
    df_mode: pd.DataFrame,
    mode: str,
    out_dir: Path,
    model_name: str,
    log1p: bool,
    future_years_ahead: int,
    future_months: List[int],
) -> Dict[str, object]:
    """
    Train on ALL available labeled rows (horizon=1 internally) and roll forward month-by-month
    to reach requested future months in year = last_year + future_years_ahead.
    """
    df = df_mode.copy()
    df = df.sort_values(["station_id", "date_index"]).copy()

    last_idx = int(df["date_index"].max())
    last_year, last_month = index_to_ym(last_idx)

    target_year = int(last_year + future_years_ahead)
    target_months = sorted(set(int(m) for m in future_months if 1 <= int(m) <= 12))
    if not target_months:
        raise ValueError("--future-months must contain valid months 1..12")

    # We need to forecast until the maximum requested target month in target_year
    target_end_idx = ym_to_index(target_year, max(target_months))

    # Build a per-station time series with possible gaps filled using station_month_mean fallback
    df = add_station_level_fallbacks(df)
    # We'll maintain a working frame with y_work (actual where present, else fallback)
    df["y_work"] = df["y_trips"].copy()
    df["y_work"] = df["y_work"].where(df["y_work"].notna(), df["station_month_mean"])

    # Model: horizon=1 training labels from y_work (best-effort)
    df["label_date_index"] = df["date_index"] + 1
    df = df.sort_values(["mode", "station_id", "date_index"]).copy()
    g = df.groupby(["mode", "station_id"], group_keys=False)
    df["y_label"] = g["y_work"].apply(lambda s: s.shift(-1))

    # Feature engineering based on y_work
    df_feat = df.copy()
    df_feat["y_trips"] = df_feat["y_work"]  # reuse feature functions
    df_feat = add_seasonality_features(df_feat)
    df_feat = add_lag_and_rolling_features(df_feat, group_cols=["mode", "station_id"], y_col="y_trips")
    df_feat = add_optional_component_lags(df_feat, group_cols=["mode", "station_id"])

    feature_cols = [
        "month_sin",
        "month_cos",
        "station_mean",
        "station_month_mean",
        "has_y_trips",
        "y_trips_lag_1",
        "y_trips_lag_2",
        "y_trips_lag_3",
        "y_trips_lag_12",
        "y_trips_rollmean_3",
        "y_trips_rollmean_6",
        "y_trips_rollstd_3",
        "y_trips_rollstd_6",
    ]
    for c in ["station_lat", "station_lng"]:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
            feature_cols.append(c)
    for c in ["start_trips_lag_1", "end_trips_lag_1", "touchpoints_lag_1"]:
        if c in df_feat.columns:
            feature_cols.append(c)

    train_rows = df_feat.dropna(subset=["y_label"]).copy()
    X_train = train_rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_rows["y_label"].astype(float).values
    y_train_t = np.log1p(np.maximum(y_train, 0)) if log1p else y_train

    if model_name == "lightgbm" and lgb is not None:
        model = make_lightgbm()
        model_used = "lightgbm"
    elif model_name == "lightgbm" and lgb is None:
        model = make_histgb()
        model_used = "histgb_fallback"
    else:
        model = make_histgb()
        model_used = "histgb"

    model.fit(X_train, y_train_t)

    # Rolling forward predictions month-by-month
    # We'll build a dictionary of last known y_work per station per month index.
    stations = df_feat["station_id"].unique().tolist()
    meta = (
        df_feat.sort_values("date_index")
        .groupby("station_id", as_index=False)
        .tail(1)[["station_id", "start_station_name", "station_lat", "station_lng"]]
    )
    meta = meta.set_index("station_id")

    # Create a working panel for forecast generation:
    # For each station, we need y_work history up to current month.
    # We'll store a dataframe of (station_id, date_index, y_work) for existing months,
    # and append predicted months as we go.
    hist = df_feat[["mode", "station_id", "date_index", "year", "month", "y_work", "station_mean", "station_month_mean", "has_y_trips",
                    "start_station_name", "station_lat", "station_lng"]].copy()

    # Ensure numeric
    hist["y_work"] = pd.to_numeric(hist["y_work"], errors="coerce").fillna(0.0)

    preds_future = []

    cur_idx = last_idx
    while cur_idx < target_end_idx:
        # we are predicting next month (cur_idx + 1)
        next_idx = cur_idx + 1
        next_year, next_month = index_to_ym(next_idx)

        # Build feature rows for feature month = cur_idx per station
        # Take the latest available row for each station at date_index == cur_idx if exists,
        # otherwise synthesize from last known history by carrying station fallbacks and month info.
        cur_rows = hist[hist["date_index"] == cur_idx].copy()

        if cur_rows.empty:
            # If the run is extremely sparse and no station has cur_idx row, synthesize from last available per station
            last_per_station = hist.sort_values("date_index").groupby("station_id", as_index=False).tail(1).copy()
            last_per_station["date_index"] = cur_idx
            last_per_station["year"] = index_to_ym(cur_idx)[0]
            last_per_station["month"] = index_to_ym(cur_idx)[1]
            cur_rows = last_per_station

        cur_rows["mode"] = mode
        cur_rows["year"] = index_to_ym(cur_idx)[0]
        cur_rows["month"] = index_to_ym(cur_idx)[1]

        # Compute features based on hist (needs lags/rolls)
        # Construct a temporary combined frame for feature computation:
        tmp = pd.concat(
            [
                hist.rename(columns={"y_work": "y_trips"}),
                cur_rows.rename(columns={"y_work": "y_trips"}),
            ],
            ignore_index=True,
        )
        tmp = tmp.drop_duplicates(subset=["station_id", "date_index"], keep="last")
        tmp = add_seasonality_features(tmp)
        tmp = add_lag_and_rolling_features(tmp, group_cols=["mode", "station_id"], y_col="y_trips")
        tmp = tmp.replace([np.inf, -np.inf], np.nan)

        feat_cur = tmp[tmp["date_index"] == cur_idx].copy()
        X_feat = feat_cur[feature_cols].fillna(0.0)

        y_pred_t = model.predict(X_feat)
        y_pred = np.expm1(y_pred_t) if log1p else y_pred_t
        y_pred = np.maximum(y_pred, 0)

        # Append predictions as next month's y_work into hist
        next_rows = feat_cur[["mode", "station_id", "start_station_name", "station_lat", "station_lng", "station_mean", "station_month_mean", "has_y_trips"]].copy()
        next_rows["date_index"] = next_idx
        next_rows["year"] = next_year
        next_rows["month"] = next_month
        next_rows["y_work"] = y_pred
        # Mark as model-generated (not observed)
        next_rows["has_y_trips"] = 0

        hist = pd.concat([hist, next_rows], ignore_index=True)

        # If this next month is one of the requested targets, record it
        if next_year == target_year and next_month in target_months:
            out = next_rows[["mode", "station_id", "start_station_name", "station_lng", "station_lat", "year", "month", "date_index"]].copy()
            out = out.rename(columns={"year": "label_year", "month": "label_month", "date_index": "label_date_index"})
            out["feature_year"] = index_to_ym(cur_idx)[0]
            out["feature_month"] = index_to_ym(cur_idx)[1]
            out["y_pred_next_month"] = y_pred
            out["split"] = "future"
            preds_future.append(out)

        cur_idx = next_idx

    pred_future = pd.concat(preds_future, ignore_index=True) if preds_future else pd.DataFrame()

    out_path = out_dir / f"predictions_future_station_month_{mode}.csv"
    pred_future.to_csv(out_path, index=False)

    return {
        "future_mode": mode,
        "model_used": model_used,
        "last_observed_year": int(last_year),
        "last_observed_month": int(last_month),
        "target_year": int(target_year),
        "target_months": target_months,
        "rows_future_pred": int(len(pred_future)),
        "pred_future_path": str(out_path),
    }


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Run summary dir, e.g. summaries/<RUN_TAG>")
    ap.add_argument("--out-dir", default=None, help="Defaults to <in-dir>/ml")
    ap.add_argument("--model", choices=["lightgbm", "histgb"], default="lightgbm")
    ap.add_argument("--horizon-months", type=int, default=1)
    ap.add_argument("--val-months", type=int, default=1)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--backtest-months", type=int, default=6)
    ap.add_argument("--log1p", action="store_true")

    ap.add_argument("--write-splits", choices=["test", "valtest", "all"], default="test",
                    help="Which splits to write to predictions CSVs (default: test only).")

    ap.add_argument("--predict-future", action="store_true",
                    help="Train on all history and forecast beyond dataset for requested future months.")
    ap.add_argument("--future-years-ahead", type=int, default=1,
                    help="Forecast year = last_year_in_data + this value (default: 1).")
    ap.add_argument("--future-months", nargs="*", type=int, default=[],
                    help="Months-of-year to output in the target future year, e.g. 3 4 5")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "ml")
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = in_dir / "citibike_station_exposure.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input: {in_csv}")

    df = pd.read_csv(in_csv)

    # Normalize schema
    if "station_id" not in df.columns:
        if "start_station_id" in df.columns:
            df = df.rename(columns={"start_station_id": "station_id"})
    if "y_trips" not in df.columns:
        if "trips" in df.columns:
            df = df.rename(columns={"trips": "y_trips"})

    required = ["mode", "year", "month", "station_id", "y_trips"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {in_csv.name}: {missing}")

    if "start_station_name" not in df.columns:
        df["start_station_name"] = ""
    if "station_lat" not in df.columns:
        df["station_lat"] = np.nan
    if "station_lng" not in df.columns:
        df["station_lng"] = np.nan

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["y_trips"] = pd.to_numeric(df["y_trips"], errors="coerce")

    df = df.dropna(subset=["mode", "station_id", "year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["station_id"] = df["station_id"].astype(str)

    df["date_index"] = df.apply(lambda r: ym_to_index(int(r["year"]), int(r["month"])), axis=1)

    results: Dict[str, object] = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "model_requested": args.model,
        "horizon_months": int(args.horizon_months),
        "val_months": int(args.val_months),
        "test_months": int(args.test_months),
        "backtest_months": int(args.backtest_months),
        "write_splits": args.write_splits,
        "log1p": bool(args.log1p),
        "modes": {},
        "future": {},
    }

    all_pred_paths: List[str] = []
    feature_cols_any: Optional[List[str]] = None

    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode].copy()
        if df_mode.empty:
            continue

        # Evaluation predictions
        eval_res = train_predict_one_mode_eval(
            df_mode=df_mode,
            mode=str(mode),
            out_dir=out_dir,
            model_name=args.model,
            horizon_months=int(args.horizon_months),
            val_months=int(args.val_months),
            test_months=int(args.test_months),
            backtest_months=int(args.backtest_months),
            log1p=bool(args.log1p),
            write_splits=args.write_splits,
        )

        results["modes"][str(mode)] = eval_res["metrics"]
        if eval_res["pred_path"]:
            all_pred_paths.append(eval_res["pred_path"])
        if feature_cols_any is None:
            feature_cols_any = eval_res["feature_cols"]

        # Future forecasting (optional)
        if args.predict_future:
            if not args.future_months:
                raise ValueError("--predict-future requires --future-months, e.g. --future-months 3 4 5")
            fut = train_and_forecast_future(
                df_mode=df_mode,
                mode=str(mode),
                out_dir=out_dir,
                model_name=args.model,
                log1p=bool(args.log1p),
                future_years_ahead=int(args.future_years_ahead),
                future_months=args.future_months,
            )
            results["future"][str(mode)] = fut

    # Combine eval predictions into one file
    preds = []
    for p in all_pred_paths:
        preds.append(pd.read_csv(p))
    if preds:
        pred_all = pd.concat(preds, ignore_index=True)
        combined_path = out_dir / "predictions_demand_station_month.csv"
        pred_all.to_csv(combined_path, index=False)
        results["combined_predictions"] = str(combined_path)

    # Write metrics.json
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Write model_card.md
    card_lines = []
    card_lines.append("# Model card: Station-month demand forecasting")
    card_lines.append("")
    card_lines.append("## Task")
    card_lines.append(f"- Predict trips per station at horizon = {int(args.horizon_months)} month(s).")
    card_lines.append("- Trained separately per mode (NYC vs JC).")
    card_lines.append("")
    card_lines.append("## Data")
    card_lines.append(f"- Source: {in_csv.name}")
    card_lines.append("- Unit: mode × station_id × (year, month)")
    card_lines.append("- Label is created by shifting trips forward within station (no leakage).")
    card_lines.append("")
    card_lines.append("## Model")
    if args.model == "lightgbm" and lgb is not None:
        card_lines.append("- LightGBM Regressor (tree boosting).")
    elif args.model == "lightgbm" and lgb is None:
        card_lines.append("- Requested LightGBM, but unavailable; used sklearn HistGradientBoostingRegressor fallback.")
    else:
        card_lines.append("- sklearn HistGradientBoostingRegressor.")
    if args.log1p:
        card_lines.append("- Target transform: log1p(y) during training; inverted with expm1 at prediction time.")
    card_lines.append("")
    card_lines.append("## Outputs")
    card_lines.append(f"- write_splits = {args.write_splits} (controls how many months appear in prediction CSVs)")
    if args.predict_future:
        card_lines.append("- Future forecasting enabled: iterative rolling 1-month-ahead to reach requested future months.")
    card_lines.append("")
    card_lines.append("## Features")
    if feature_cols_any:
        card_lines.append("- " + ", ".join(feature_cols_any))
    else:
        card_lines.append("- (No features recorded.)")
    card_lines.append("")
    card_lines.append("## Evaluation")
    card_lines.append("- Time-based split on LABEL month; degrades gracefully if not enough months.")
    card_lines.append("- Metrics: MAE and SMAPE; baselines derived from lag features.")
    card_lines.append("")
    card_lines.append("## Caveats")
    card_lines.append("- Sparse month selections reduce effectiveness of lag/rolling features.")
    card_lines.append("- Without external weather, seasonality is captured by month-of-year (sin/cos).")
    card_lines.append("")

    model_card_path = out_dir / "model_card.md"
    model_card_path.write_text("\n".join(card_lines), encoding="utf-8")

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {model_card_path}")
    if preds:
        print(f"Wrote: {out_dir / 'predictions_demand_station_month.csv'}")
    for p in all_pred_paths:
        print(f"Wrote: {p}")
    if args.predict_future:
        for mode, v in results["future"].items():
            print(f"Wrote future: {v.get('pred_future_path')}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_axa_scorecard.py

Build an AXA-ready station scorecard from:
  <in-dir>/station_risk_exposure_plus_crashproximity.csv

Writes:
  <out-dir>/axa_partner_scorecard_<radius>m.csv
    e.g. axa_partner_scorecard_500m.csv
         axa_partner_scorecard_750m.csv

Key features:
- Works with ANY radius as long as the input CSV contains:
    crashes_within_<R>m
    crashes_within_<R>m_per_100k_trips
- Supports:
    --radius 750m
    --radius 750
    --radius 1km
    --radius auto  (use the maximum available radius in the input CSV)

Method:
- Exposure = trips
- Risk proxy = crash proximity rate per 100k trips
- Empirical Bayes smoothing for Poisson rates + credibility threshold to avoid tiny-denominator noise
- Per-mode processing (nyc/jc): never rank across modes
- If risk proxy has no usable signal for a mode, fall back to exposure-only scoring
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2  # type: ignore
except Exception:
    chi2 = None


# -----------------------------
# Small helpers
# -----------------------------
def poisson_rate_ci(
    count: int, exposure: float, scale: float = 100_000.0, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Exact Poisson CI for rate=(count/exposure)*scale.
    If scipy isn't available, return (rate, rate).
    """
    if exposure <= 0:
        return float("nan"), float("nan")

    rate = (count / exposure) * scale

    if chi2 is None:
        return rate, rate

    k = int(count)
    if k == 0:
        lam_lo = 0.0
    else:
        lam_lo = 0.5 * chi2.ppf(alpha / 2, 2 * k)
    lam_hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))

    return (lam_lo / exposure * scale, lam_hi / exposure * scale)


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average") * 100.0


# -----------------------------
# Radius parsing / discovery
# -----------------------------
_RADIUS_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>m|km)?\s*$", re.IGNORECASE)


def parse_radius_to_m(raw: str) -> int:
    """
    Parse "500m", "500", "1km", "0.75km" -> integer meters.
    """
    s = str(raw).strip().lower()
    m = _RADIUS_RE.match(s)
    if not m:
        raise ValueError(
            f"Invalid --radius={raw!r}. Use like 250m, 750m, 750, 1km, or auto."
        )
    val = float(m.group("num"))
    unit = (m.group("unit") or "m").lower()
    meters = val * (1000.0 if unit == "km" else 1.0)
    if meters <= 0:
        raise ValueError(f"--radius must be > 0 (got {raw!r})")
    return int(round(meters))


def available_radii_in_df(df: pd.DataFrame) -> list[int]:
    radii: set[int] = set()
    for c in df.columns:
        mm = re.match(r"^crashes_within_(\d+)m(?:_per_100k_trips)?$", str(c))
        if mm:
            radii.add(int(mm.group(1)))
    return sorted(radii)


# -----------------------------
# EB smoothing
# -----------------------------
def eb_rate_per_trip(counts: pd.Series, exposures: pd.Series, m_prior: float) -> pd.Series:
    """
    EB posterior mean for Poisson rates:
      r0 = total_count/total_exposure
      r_EB = (count + r0*m) / (exposure + m)
    """
    exposures = exposures.astype(float)
    counts = counts.astype(float)

    total_exposure = float(exposures.sum())
    if total_exposure <= 0:
        return pd.Series(np.nan, index=counts.index)

    baseline_rate_per_trip = float(counts.sum() / total_exposure)
    prior_count = baseline_rate_per_trip * float(m_prior)

    return (counts + prior_count) / (exposures + float(m_prior))


def compute_optimal_eb_prior(counts: pd.Series, exposures: pd.Series) -> float:
    """
    Method-of-moments estimate for EB prior strength m (bounded).
    """
    valid = (exposures > 0) & (counts >= 0)
    if valid.sum() < 30:
        print("  WARNING: <30 valid observations for EB prior estimation -> using m=20,000")
        return 20000.0

    counts_v = counts[valid]
    exposures_v = exposures[valid]
    rates = counts_v / exposures_v

    mean_rate = float(rates.mean())
    var_observed = float(rates.var())

    mean_exposure = float(exposures_v.mean())
    var_sampling = (mean_rate / mean_exposure) if mean_exposure > 0 else 0.0

    var_true = max(1e-10, var_observed - var_sampling)

    if var_true < 1e-10:
        print("  INFO: No detectable true variance -> using strong prior m=100,000")
        return 100000.0

    m_est = (mean_rate**2) / var_true
    m_bounded = float(max(1000.0, min(100000.0, m_est)))

    print(f"  EB prior auto-calibration: m_est={m_est:.1f} -> m={m_bounded:.1f}")
    return m_bounded


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build AXA partner scorecard from station risk exposure data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-dir", required=True, help="Run summary directory, e.g. summaries/<RUN_TAG>")
    ap.add_argument("--out-dir", required=True, help="Output directory (usually same as --in-dir)")
    ap.add_argument(
        "--radius",
        default="500m",
        help="Crash proximity radius to use (e.g. 250m, 450m, 750m, 1km, 750) or 'auto' (max available).",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for confidence intervals (default 0.05 -> 95%% CI)")
    ap.add_argument("--min-trips", type=int, default=5_000, help="Minimum trips for credible risk ranking")
    ap.add_argument("--m-prior", type=float, default=None, help="EB prior strength (pseudo-trips). If omitted, auto-calibrated.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_path = in_dir / "station_risk_exposure_plus_crashproximity.csv"
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing required input: {risk_path}")

    df = pd.read_csv(risk_path)

    # Resolve radius
    radius_raw = str(args.radius).strip()
    if radius_raw.lower() in {"auto", "max"}:
        radii = available_radii_in_df(df)
        if not radii:
            raise ValueError(
                f"No crash proximity columns found in {risk_path.name}. "
                "Did you run summarize with NYPD + --radii-m?"
            )
        radius_m = max(radii)
        print(f"Auto-selected radius: {radius_m}m")
    else:
        radius_m = parse_radius_to_m(radius_raw)

    crash_col = f"crashes_within_{radius_m}m"
    rate_col = f"{crash_col}_per_100k_trips"

    required = [
        "mode",
        "start_station_id",
        "start_station_name",
        "station_lat",
        "station_lng",
        "trips",
        crash_col,
        rate_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        avail = available_radii_in_df(df)
        raise ValueError(
            f"Missing required columns in {risk_path.name}: {missing}\n"
            f"Requested radius: {radius_m}m\n"
            f"Available radii in file: {avail}\n"
            "Fix: re-run summarize with --radii-m including your radius."
        )

    # Clean types
    df["mode"] = df["mode"].astype(str)
    df["trips"] = pd.to_numeric(df["trips"], errors="coerce").fillna(0).astype(int)
    df[crash_col] = pd.to_numeric(df[crash_col], errors="coerce").fillna(0).astype(int)
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)

    # Base fields
    df["exposure_trips"] = df["trips"]
    df["crash_count"] = df[crash_col]
    df["risk_rate_per_100k_trips"] = df[rate_col]

    # Poisson CI
    ci = df.apply(
        lambda r: poisson_rate_ci(int(r["crash_count"]), float(r["exposure_trips"]), alpha=args.alpha),
        axis=1,
    )
    df["risk_rate_ci_low"] = [x[0] for x in ci]
    df["risk_rate_ci_high"] = [x[1] for x in ci]

    out_parts: list[pd.DataFrame] = []

    for mode, g in df.groupby("mode", dropna=False):
        g = g.copy()
        print(f"\n{'='*70}\nProcessing mode: {mode}\n{'='*70}")

        g["exposure_index_pct"] = pct_rank_0_100(g["exposure_trips"])
        min_trips = int(args.min_trips)
        sufficient = g["exposure_trips"] >= min_trips

        # Decide if risk proxy is usable among credible stations
        if sufficient.sum() < 10:
            risk_has_signal = False
        else:
            risk_has_signal = (g.loc[sufficient, "risk_rate_per_100k_trips"].nunique(dropna=True) > 1)

        g["risk_proxy_available"] = bool(risk_has_signal)

        if not risk_has_signal:
            # Exposure-only fallback
            g["eb_risk_rate_per_100k_trips"] = np.nan
            g["risk_index_pct"] = np.nan
            g["expected_incidents_proxy"] = np.nan
            g["credibility_flag"] = "no_risk_data"

            g["exposure_pct"] = g["exposure_index_pct"] / 100.0
            g["risk_pct"] = 0.0
            g["axa_priority_score"] = g["exposure_pct"]
            g["scoring_strategy"] = "exposure_only_no_risk_signal"

            g["prevention_hotspot"] = g["exposure_index_pct"] >= 90.0
            g["product_hotspot"] = g["exposure_index_pct"] >= 80.0
            g["acquisition_hotspot"] = g["exposure_index_pct"] >= 70.0

            out_parts.append(g)
            continue

        # EB smoothing
        if args.m_prior is None:
            m_prior = compute_optimal_eb_prior(g["crash_count"], g["exposure_trips"])
        else:
            m_prior = float(args.m_prior)
            print(f"Using user EB prior m={m_prior:.1f}")

        eb_per_trip = eb_rate_per_trip(g["crash_count"], g["exposure_trips"], m_prior)
        g["eb_risk_rate_per_100k_trips"] = eb_per_trip * 100_000.0

        # Risk ranking ONLY for credible stations
        g["risk_index_pct"] = np.nan
        g["credibility_flag"] = "insufficient_data"
        g.loc[sufficient, "risk_index_pct"] = pct_rank_0_100(g.loc[sufficient, "eb_risk_rate_per_100k_trips"])
        g.loc[sufficient, "credibility_flag"] = "credible"

        # Expected incidents proxy (freq × exposure)
        g["expected_incidents_proxy"] = eb_per_trip * g["exposure_trips"]

        g["exposure_pct"] = g["exposure_index_pct"] / 100.0
        g["risk_pct"] = g["risk_index_pct"].fillna(0.0) / 100.0

        # Priority score: percentile of expected incidents (0..1)
        g["axa_priority_score"] = pct_rank_0_100(g["expected_incidents_proxy"]) / 100.0
        g["scoring_strategy"] = f"eb_expected_incidents_mintrips{min_trips}_mprior{int(m_prior)}"

        g["prevention_hotspot"] = (
            (g["exposure_index_pct"] >= 80.0)
            & (g["risk_pct"] >= 0.8)
            & (g["credibility_flag"] == "credible")
        )
        g["product_hotspot"] = g["exposure_index_pct"] >= 80.0
        g["acquisition_hotspot"] = (g["exposure_index_pct"] >= 70.0) & (g["risk_pct"] <= 0.3)

        out_parts.append(g)

    out_all = pd.concat(out_parts, axis=0, ignore_index=True)

    out_cols = [
        "mode",
        "start_station_id",
        "start_station_name",
        "station_lat",
        "station_lng",
        "exposure_trips",
        "crash_count",
        "risk_rate_per_100k_trips",
        "risk_rate_ci_low",
        "risk_rate_ci_high",
        "risk_proxy_available",
        "credibility_flag",
        "exposure_pct",
        "risk_pct",
        "axa_priority_score",
        "prevention_hotspot",
        "product_hotspot",
        "acquisition_hotspot",
        "exposure_index_pct",
        "eb_risk_rate_per_100k_trips",
        "risk_index_pct",
        "expected_incidents_proxy",
        "scoring_strategy",
    ]
    out_cols = [c for c in out_cols if c in out_all.columns]
    out = out_all[out_cols].copy()

    out = out.sort_values(["axa_priority_score", "exposure_trips"], ascending=False).reset_index(drop=True)

    out_path = out_dir / f"axa_partner_scorecard_{radius_m}m.csv"
    out.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"Total stations: {len(out):,}")


if __name__ == "__main__":
    main()

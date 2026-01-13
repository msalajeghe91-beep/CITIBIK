#!/usr/bin/env python3
"""
Build an AXA-ready 'when to act' table from per-run time summaries.

Inputs (in summaries/<RUN_TAG>/):
  - citibike_trips_by_hour.csv
  - citibike_trips_by_dow.csv
  - citibike_trips_by_month.csv

Output:
  - axa_target_windows.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def top_n(df: pd.DataFrame, n: int, sort_col: str) -> pd.DataFrame:
    return df.sort_values(sort_col, ascending=False).head(n).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Run summary directory, e.g. summaries/<RUN_TAG>")
    ap.add_argument("--out-dir", required=True, help="Output directory (usually same as --in-dir)")
    ap.add_argument("--top-hours", type=int, default=5, help="Top hours per week_part")
    ap.add_argument("--top-dows", type=int, default=3, help="Top days-of-week per week_part")
    ap.add_argument("--top-months", type=int, default=6, help="Top months overall (by trips_per_day)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_hour = in_dir / "citibike_trips_by_hour.csv"
    p_dow = in_dir / "citibike_trips_by_dow.csv"
    p_month = in_dir / "citibike_trips_by_month.csv"

    for p in (p_hour, p_dow, p_month):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    hour = pd.read_csv(p_hour)
    dow = pd.read_csv(p_dow)
    month = pd.read_csv(p_month)

    # ---- Hour targeting (weekday vs weekend) ----
    hour["hour"] = pd.to_numeric(hour["hour"], errors="coerce")
    hour["trips"] = pd.to_numeric(hour["trips"], errors="coerce").fillna(0).astype(int)
    hour["pct_within_week_part"] = pd.to_numeric(hour["pct_within_week_part"], errors="coerce")

    hour_out = []
    for wp in ["weekday", "weekend"]:
        sub = hour[hour["week_part"] == wp].copy()
        sub = top_n(sub, args.top_hours, "trips")
        sub["window_type"] = "hour_peak"
        sub["segment"] = wp
        sub["window_label"] = sub["hour"].astype(int).astype(str).str.zfill(2) + ":00"
        sub["priority_metric"] = sub["pct_within_week_part"]
        hour_out.append(sub)

    hour_out = pd.concat(hour_out, ignore_index=True)[
        ["window_type", "segment", "window_label", "trips", "pct_of_mode_year_trips", "pct_within_week_part", "priority_metric"]
    ]

    # Optional: commuter windows (fixed definition)
    commuter = hour[hour["week_part"] == "weekday"].copy()
    commuter["commuter_window"] = commuter["hour"].isin([7, 8, 9, 16, 17, 18, 19])
    commuter = commuter[commuter["commuter_window"]].copy()
    commuter = commuter.sort_values("trips", ascending=False)
    commuter["window_type"] = "commuter_hours"
    commuter["segment"] = "weekday"
    commuter["window_label"] = commuter["hour"].astype(int).astype(str).str.zfill(2) + ":00"
    commuter["priority_metric"] = commuter["pct_within_week_part"]
    commuter_out = commuter[
        ["window_type", "segment", "window_label", "trips", "pct_of_mode_year_trips", "pct_within_week_part", "priority_metric"]
    ].head(7)

    # ---- Day-of-week targeting ----
    dow["trips"] = pd.to_numeric(dow["trips"], errors="coerce").fillna(0).astype(int)
    dow["pct_within_week_part"] = pd.to_numeric(dow["pct_within_week_part"], errors="coerce")

    dow_out = []
    for wp in ["weekday", "weekend"]:
        sub = dow[dow["week_part"] == wp].copy()
        sub = top_n(sub, args.top_dows, "trips")
        sub["window_type"] = "day_of_week_peak"
        sub["segment"] = wp
        sub["window_label"] = sub["dow_name"]
        sub["priority_metric"] = sub["pct_within_week_part"]
        dow_out.append(sub)

    dow_out = pd.concat(dow_out, ignore_index=True)[
        ["window_type", "segment", "window_label", "trips", "pct_of_mode_year_trips", "pct_within_week_part", "priority_metric"]
    ]

    # ---- Month targeting (campaign seasonality) ----
    # We rank by trips_per_day to avoid month-length bias.
    month["trips_per_day"] = pd.to_numeric(month["trips_per_day"], errors="coerce")
    month["trips"] = pd.to_numeric(month["trips"], errors="coerce").fillna(0).astype(int)
    month["year"] = pd.to_numeric(month["year"], errors="coerce").astype("Int64")
    month["month"] = pd.to_numeric(month["month"], errors="coerce").astype("Int64")

    month_top = month.dropna(subset=["trips_per_day"]).sort_values("trips_per_day", ascending=False).head(args.top_months).copy()
    month_top["window_type"] = "month_peak"
    month_top["segment"] = "overall"
    month_top["window_label"] = month_top["year"].astype(int).astype(str) + "-" + month_top["month"].astype(int).astype(str).str.zfill(2)
    month_top["priority_metric"] = month_top["trips_per_day"]
    month_out = month_top.rename(columns={"trips_per_day": "pct_within_week_part"})[
        ["window_type", "segment", "window_label", "trips", "pct_within_week_part", "priority_metric"]
    ]
    month_out["pct_of_mode_year_trips"] = None  # not defined for month table
    month_out = month_out[
        ["window_type", "segment", "window_label", "trips", "pct_of_mode_year_trips", "pct_within_week_part", "priority_metric"]
    ]

    # ---- Combine ----
    out = pd.concat([hour_out, commuter_out, dow_out, month_out], ignore_index=True)

    # Add recommended actions (simple mapping)
    def recommend_action(row):
        if row["window_type"] == "hour_peak" and row["segment"] == "weekday":
            return "Acquisition + product upsell (commute peaks); consider safety nudge"
        if row["window_type"] == "hour_peak" and row["segment"] == "weekend":
            return "Leisure targeting + safety messaging"
        if row["window_type"] == "commuter_hours":
            return "Strong commuter acquisition + product bundle"
        if row["window_type"] == "day_of_week_peak":
            return "Campaign scheduling + staffing/ops alignment"
        if row["window_type"] == "month_peak":
            return "Seasonal campaign planning"
        return "General"

    out["recommended_action"] = out.apply(recommend_action, axis=1)

    out_path = out_dir / "axa_target_windows.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
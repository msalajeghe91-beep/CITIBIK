#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def _mode_match(p: Path, mode: str) -> bool:
    s = str(p).lower()
    if mode == "any":
        return True
    if mode == "nyc":
        return ("modenyc" in s) or ("mode=nyc" in s) or ("/nyc" in s)
    if mode == "jc":
        return ("modejc" in s) or ("mode=jc" in s) or ("/jc" in s)
    return True


def _find_default_input(mode: str) -> Path | None:
    """
    Prefer summaries/latest/station_risk_exposure_plus_crashproximity.csv if present.
    Else: pick the most recently modified matching csv, preferring non-_archive paths.
    """
    latest_candidate = Path("summaries/latest/station_risk_exposure_plus_crashproximity.csv")
    if latest_candidate.exists() and _mode_match(latest_candidate, mode):
        return latest_candidate

    roots = [Path("summaries"), Path("reports")]
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*station_risk_exposure_plus_crashproximity*.csv"):
            if not _mode_match(p, mode):
                continue
            hits.append(p)

    if not hits:
        return None

    non_archive = [p for p in hits if "_archive" not in str(p).lower()]
    archive = [p for p in hits if "_archive" in str(p).lower()]

    def newest(ps: list[Path]) -> Path:
        # newest by modification time
        return max(ps, key=lambda x: x.stat().st_mtime)

    return newest(non_archive) if non_archive else newest(archive)


def _read_csv_typed(path: Path) -> pd.DataFrame:
    # Force IDs/names to stay as strings (prevents station_id becoming floats)
    dtype = {
        "mode": "string",
        "start_station_id": "string",
        "end_station_id": "string",
        "start_station_name": "string",
        "end_station_name": "string",
        "station_id": "string",
        "station_name": "string",
    }
    return pd.read_csv(path, dtype=dtype)


def _maybe_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "start_station_id" in df.columns and "station_id" not in df.columns:
        rename_map["start_station_id"] = "station_id"
    if "start_station_name" in df.columns and "station_name" not in df.columns:
        rename_map["start_station_name"] = "station_name"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check station risk output and rank sensibly.")
    ap.add_argument("--in", dest="in_path", type=Path, default=None,
                    help="Path to station risk CSV. If omitted, auto-detect (prefers summaries/latest).")
    ap.add_argument("--mode", choices=["nyc", "jc", "any"], default="nyc",
                    help="When auto-detecting, prefer this mode (default: nyc).")
    ap.add_argument("--head", type=int, default=15, help="Rows to print for head().")
    ap.add_argument("--top", type=int, default=20, help="Top N to show.")
    ap.add_argument("--min-trips", type=int, default=1000,
                    help="Minimum trips required to rank by crash rate (default: 1000).")
    ap.add_argument("--out", type=Path, default=None,
                    help="If set, write output CSVs to this directory (e.g. reports/check_station_risk).")
    args = ap.parse_args()

    in_path = args.in_path or _find_default_input(args.mode)
    if in_path is None:
        print("Couldn't auto-detect an input file. Pass one with --in.", file=sys.stderr)
        return 2
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 2

    df = _read_csv_typed(in_path)
    print(f"\nLoaded: {in_path}  (rows={len(df):,}, cols={len(df.columns)})")
    print("Columns:\n - " + "\n - ".join(map(str, df.columns)))

    df = _maybe_rename_columns(df)

    # numeric coercions (only if present)
    for c in [
        "trips",
        "station_lat",
        "station_lng",
        "crashes_within_250m",
        "crashes_within_500m",
        "crashes_within_250m_per_100k_trips",
        "crashes_within_500m_per_100k_trips",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "trips" not in df.columns:
        print("\nERROR: No 'trips' column found.", file=sys.stderr)
        return 2

    # Recompute rates defensively
    if "crashes_within_250m" in df.columns:
        df["crashes_250m_per_100k_trips_recalc"] = (df["crashes_within_250m"] / df["trips"]) * 100000
    if "crashes_within_500m" in df.columns:
        df["crashes_500m_per_100k_trips_recalc"] = (df["crashes_within_500m"] / df["trips"]) * 100000

    print("\nHead:")
    show_cols = [c for c in [
        "mode", "station_id", "station_name", "trips",
        "crashes_within_250m", "crashes_within_500m",
        "crashes_250m_per_100k_trips_recalc", "crashes_500m_per_100k_trips_recalc",
    ] if c in df.columns]
    print(df[show_cols].head(args.head).to_string(index=False))

    # Top by trips
    top_trips_cols = [c for c in ["mode", "station_id", "station_name", "trips"] if c in df.columns]
    top_by_trips = df.sort_values("trips", ascending=False).head(args.top)[top_trips_cols]
    print(f"\nTop {args.top} by trips:")
    print(top_by_trips.to_string(index=False))

    # Top by crash rate, filtered
    filt = df[df["trips"] >= args.min_trips].copy()
    top_by_rate = None
    metric = None
    if len(filt) == 0:
        print(f"\nNo rows with trips >= {args.min_trips}. Try a smaller --min-trips.")
    else:
        for k in ["crashes_250m_per_100k_trips_recalc", "crashes_500m_per_100k_trips_recalc"]:
            if k in filt.columns:
                metric = k
                break
        if metric:
            top_rate_cols = [c for c in [
                "mode", "station_id", "station_name", "trips",
                "crashes_within_250m", "crashes_within_500m",
                metric,
            ] if c in filt.columns]
            top_by_rate = filt.sort_values(metric, ascending=False).head(args.top)[top_rate_cols]
            print(f"\nTop {args.top} by {metric} (filtered: trips >= {args.min_trips}):")
            print(top_by_rate.to_string(index=False))
        else:
            print("\nNo crash-rate metric columns found to rank.")

    # sanity
    print("\nSanity stats:")
    print(f"  trips sum: {df['trips'].sum(skipna=True):,.0f}")
    print(f"  stations:  {df['station_id'].nunique(dropna=True) if 'station_id' in df.columns else 'n/a'}")
    print(f"  trips min/median/max: {df['trips'].min(skipna=True):,.0f} / {df['trips'].median(skipna=True):,.0f} / {df['trips'].max(skipna=True):,.0f}")

    tiny = (df["trips"] <= 10).sum()
    print(f"  stations with trips <= 10: {tiny:,} ({tiny/len(df)*100:.1f}%)")

    # Write outputs if requested
    if args.out is not None:
        outdir = args.out
        out_trips = outdir / "top_by_trips.csv"
        _ensure_parent(out_trips)
        top_by_trips.to_csv(out_trips, index=False)

        if top_by_rate is not None and metric is not None:
            out_rate = outdir / f"top_by_{metric}.csv"
            top_by_rate.to_csv(out_rate, index=False)

        out_meta = outdir / "input_used.txt"
        out_meta.write_text(str(in_path) + "\n", encoding="utf-8")

        print(f"\nWrote:\n - {out_trips}")
        if top_by_rate is not None and metric is not None:
            print(f" - {outdir / f'top_by_{metric}.csv'}")
        print(f" - {out_meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

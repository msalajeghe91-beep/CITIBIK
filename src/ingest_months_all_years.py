#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


def pick_csv(names: list[str]) -> str:
    csvs = [n for n in names if n.lower().endswith(".csv") and "__macosx" not in n.lower()]
    if not csvs:
        raise ValueError("No CSV found inside zip")
    return csvs[0]


def parse_year_month(zip_name: str) -> tuple[int, int]:
    """
    Supports:
      JC-YYYYMM-citibike-tripdata...
      YYYYMM-citibike-tripdata...
    """
    m = re.search(r"^(?:JC-)?(\d{4})(\d{2})-", zip_name)
    if not m:
        raise ValueError(f"Could not parse year/month from {zip_name}")
    return int(m.group(1)), int(m.group(2))


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    # your existing logic (kept)
    df.columns = [c.strip().lower() for c in df.columns]

    if "started_at" in df.columns and "ended_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
        df["trip_minutes"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0

        keep = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id",
            "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng",
            "member_casual", "trip_minutes",
        ]
        df = df[[c for c in keep if c in df.columns]]
    else:
        start_col = "starttime" if "starttime" in df.columns else ("start time" if "start time" in df.columns else None)
        stop_col = "stoptime" if "stoptime" in df.columns else ("stop time" if "stop time" in df.columns else None)
        if start_col is None or stop_col is None:
            raise ValueError(f"Unknown schema; columns begin: {list(df.columns)[:25]}")

        df["started_at"] = pd.to_datetime(df[start_col], errors="coerce")
        df["ended_at"] = pd.to_datetime(df[stop_col], errors="coerce")
        df["trip_minutes"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0

        rename_map = {
            "start station name": "start_station_name",
            "start station id": "start_station_id",
            "end station name": "end_station_name",
            "end station id": "end station_id",
            "start station latitude": "start_lat",
            "start station longitude": "start_lng",
            "end station latitude": "end_lat",
            "end station longitude": "end_lng",
            "usertype": "member_casual",
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        if "member_casual" in df.columns:
            df["member_casual"] = df["member_casual"].replace({"Subscriber": "member", "Customer": "casual"})

        keep = [
            "started_at", "ended_at",
            "start_station_name", "start_station_id",
            "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng",
            "member_casual", "trip_minutes",
        ]
        df = df[[c for c in keep if c in df.columns]]

    df = df.dropna(subset=["started_at", "ended_at"])
    df = df[df["trip_minutes"].between(1, 180)]

    df["hour"] = df["started_at"].dt.hour
    df["weekday"] = df["started_at"].dt.day_name()
    df["date"] = df["started_at"].dt.date
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/citibike")
    ap.add_argument("--out-dir", default="data/processed/citibike_parquet")
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--months", nargs="+", type=int, required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = set(args.years)
    months = set(args.months)

    zips = sorted(raw_dir.glob("*.zip"))
    selected: list[Path] = []
    for zp in zips:
        try:
            y, m = parse_year_month(zp.name)
        except ValueError:
            continue
        if y in years and m in months:
            selected.append(zp)

    if not selected:
        raise SystemExit(f"No matching zips found in {raw_dir} for years={sorted(years)} months={sorted(months)}")

    for zp in selected:
        y, m = parse_year_month(zp.name)
        out_path = out_dir / f"year={y}" / f"month={m:02d}" / "part.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zp, "r") as z:
            csv_name = pick_csv(z.namelist())
            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        df = standardize(df)
        df.to_parquet(out_path, index=False)
        print(f"✅ {zp.name} -> {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()

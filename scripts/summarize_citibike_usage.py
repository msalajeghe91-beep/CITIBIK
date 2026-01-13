#!/usr/bin/env python3
"""
summarize_citibike_usage.py

Reproducible summary metrics from partitioned Citi Bike parquet.

Expected parquet layout:
  <PARQUET_DIR>/mode=<mode>/year=<YYYY>/month=<MM>/tripdata.parquet

Outputs written to --out-dir:
  - citibike_trips_by_month.csv
  - citibike_trips_by_year.csv
  - citibike_trips_by_dow.csv (INCLUDES YEAR COLUMN)
  - citibike_trips_by_hour.csv (INCLUDES YEAR COLUMN)
  - citibike_station_exposure.csv (INCLUDES YEAR/MONTH COLUMNS)
      * Contains start_trips, end_trips, touchpoints per station per month
      * trips column defaults to touchpoints (start+end) unless --station-exposure=starts
      * FILTERED BY MODE: NYC mode excludes JC/HB stations, JC mode includes only JC/HB stations
  - station_risk_exposure_plus_crashproximity.csv
      * Aggregated from station exposure across time for stable risk estimates
      * Uses NYPD crash CSV to compute crash proximity counts (NYC-only proxy)
      * For JC mode, writes zeros with risk_proxy_available=0 (proxy unavailable)
  - summary_highlights.md (unless --no-highlights)

Design choices:
- Month/year totals use parquet metadata row counts (fast, no full scan).
- DOW/hour/station exposure scan minimal columns with PyArrow batch iteration.
- DOW and Hour summaries include YEAR to prevent mixing across years.
- Station exposure includes YEAR/MONTH.

NYC/JC hygiene:
- Station exposure is FILTERED BY MODE using PREFIX + COORDINATES to prevent contamination:
  * NYC mode: Removes stations with JC/HB prefixes AND stations in JC geographic area
  * JC mode: Keeps only stations with JC/HB prefixes OR stations in JC geographic area
  * Geographic filtering catches JC stations without JC prefix (e.g., stations 3199, 3183)
- This prevents cross-system trips from contaminating regional analyses
- NYPD crash proxy is NYC-only and receives additional filtering for safety
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Partition parsing
# -----------------------------
_PART_RE = re.compile(
    r"mode=([^/\\]+)[/\\]year=(\d{4})[/\\]month=(\d{2})[/\\]tripdata\.parquet$"
)


def parse_parts(p: Path) -> Optional[Tuple[str, int, int]]:
    """Extract (mode, year, month) from a partitioned parquet file path."""
    m = _PART_RE.search(str(p))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


# -----------------------------
# PyArrow helpers
# -----------------------------
def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "ERROR: pyarrow is required.\n"
            "Install in your venv:\n"
            "  .venv/bin/pip install pyarrow\n"
        ) from e


def count_rows_parquet(path: Path) -> int:
    """Fast row count via parquet metadata (no full read)."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows)


def parquet_available_columns(path: Path) -> List[str]:
    """Get parquet schema column names without loading data."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return list(pf.schema.names)


def iter_parquet_batches(parquet_path: Path, columns: List[str], batch_size: int = 250_000):
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        yield batch


def first_existing_column(available: Iterable[str], candidates: List[str]) -> Optional[str]:
    """Return exact column name from available that matches any candidate (case-insensitive)."""
    avail_map = {str(c).lower(): str(c) for c in available}
    for cand in candidates:
        key = str(cand).lower()
        if key in avail_map:
            return avail_map[key]
    return None


# -----------------------------
# Station aggregation struct
# -----------------------------
@dataclass
class StationAgg2:
    start_trips: int = 0
    end_trips: int = 0
    lat_sum: float = 0.0
    lng_sum: float = 0.0
    coord_n: int = 0
    name: str = ""  # representative station name (first non-empty)


# -----------------------------
# Utility helpers
# -----------------------------
def ensure_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def _coerce_numeric_latlng(df: pd.DataFrame, lat_col: str, lng_col: str) -> pd.DataFrame:
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lng_col] = pd.to_numeric(df[lng_col], errors="coerce")
    return df.dropna(subset=[lat_col, lng_col])


def parse_radii_m(raw: str) -> List[float]:
    """
    Parse radii string like "250,500" or "250 500" into floats.
    Returns [] if nothing valid.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    out: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            val = float(p)
            if val > 0:
                out.append(val)
        except Exception:
            pass
    # de-dup + stable sort
    out = sorted(set(out))
    return out


# -----------------------------
# Optional: NYPD crash proximity
# -----------------------------
EARTH_RADIUS_M = 6371000.0


def _to_radians(lat_series: pd.Series, lng_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_rad": pd.to_numeric(lat_series, errors="coerce") * math.pi / 180.0,
            "lng_rad": pd.to_numeric(lng_series, errors="coerce") * math.pi / 180.0,
        }
    )


def compute_crash_proximity(stations: pd.DataFrame, nypd_csv: Path, radii_m: List[float]) -> pd.DataFrame:
    """
    Computes crash counts within given radii for each station using BallTree (haversine).

    stations must include:
      - station_lat
      - station_lng

    Returns a copy of stations with columns:
      - crashes_within_<R>m for each R in radii_m
    """
    try:
        from sklearn.neighbors import BallTree
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for crash proximity (BallTree).\n"
            "Install:\n"
            "  .venv/bin/pip install scikit-learn\n"
        ) from e

    crashes = pd.read_csv(nypd_csv, low_memory=False)

    col_lat = first_existing_column(crashes.columns, ["LATITUDE", "latitude", "lat"])
    col_lng = first_existing_column(crashes.columns, ["LONGITUDE", "longitude", "lng", "lon"])
    if not col_lat or not col_lng:
        raise RuntimeError(
            f"Could not find latitude/longitude columns in NYPD file: {nypd_csv}\n"
            f"Columns seen (first 60): {list(crashes.columns)[:60]}"
        )

    crashes_geo = crashes[[col_lat, col_lng]].copy()
    crashes_geo[col_lat] = pd.to_numeric(crashes_geo[col_lat], errors="coerce")
    crashes_geo[col_lng] = pd.to_numeric(crashes_geo[col_lng], errors="coerce")
    crashes_geo = crashes_geo.dropna(subset=[col_lat, col_lng])
    if crashes_geo.empty:
        raise RuntimeError(f"No valid crash coordinates after cleaning: {nypd_csv}")

    crash_rad = _to_radians(crashes_geo[col_lat], crashes_geo[col_lng])[["lat_rad", "lng_rad"]].values
    tree = BallTree(crash_rad, metric="haversine")

    st = stations.copy()
    st["station_lat"] = pd.to_numeric(st["station_lat"], errors="coerce")
    st["station_lng"] = pd.to_numeric(st["station_lng"], errors="coerce")
    st = st.dropna(subset=["station_lat", "station_lng"]).reset_index(drop=True)

    st_rad = _to_radians(st["station_lat"], st["station_lng"])[["lat_rad", "lng_rad"]].values

    for r_m in radii_m:
        r_rad = float(r_m) / EARTH_RADIUS_M
        counts = tree.query_radius(st_rad, r=r_rad, count_only=True)
        st[f"crashes_within_{int(r_m)}m"] = counts.astype(int)

    return st


def drop_nj_like_stations_for_nyc_proxy(df_st_sorted: pd.DataFrame, mode: Optional[str], prefixes: List[str]) -> pd.DataFrame:
    """
    NYC hygiene: NYPD crashes are NYC-only. Drop NJ-style station IDs (e.g., JC*, HB*) from crash-proxy input.
    This does NOT change station exposure outputs unless you call it on those.
    """
    if not mode or str(mode).lower() != "nyc":
        return df_st_sorted
    if "start_station_id" not in df_st_sorted.columns:
        return df_st_sorted

    sid = df_st_sorted["start_station_id"].astype(str)
    prefixes_tup = tuple(p for p in prefixes if p)
    if not prefixes_tup:
        return df_st_sorted

    mask = sid.str.startswith(prefixes_tup)
    removed = int(mask.sum())
    if removed:
        df_st_sorted = df_st_sorted[~mask].copy()
        print(f"NOTE: Dropped {removed} station-month rows with station_id prefixes {prefixes_tup} from NYC crash-proxy input.")
    return df_st_sorted


# -----------------------------
# Highlights writer (terminal + markdown file)
# -----------------------------
def build_highlights_markdown(
    df_year: pd.DataFrame,
    df_month: pd.DataFrame,
    df_dow: pd.DataFrame,
    df_hr: pd.DataFrame,
    mode: Optional[str],
    top_n: int,
) -> List[str]:
    lines: List[str] = []
    lines.append("# Summary highlights")
    lines.append("")
    if mode:
        lines.append(f"- Mode filter: `{mode}`")
    lines.append("")

    # Years
    if not df_year.empty:
        for m in sorted(df_year["mode"].unique()):
            sub = df_year[df_year["mode"] == m].copy()
            total = sub["trips"].sum()
            sub = sub.sort_values("trips", ascending=False).head(top_n)
            lines.append(f"## Top years for mode={m}")
            for _, r in sub.iterrows():
                pct = (float(r["trips"]) / total * 100.0) if total else 0.0
                lines.append(f"- **{int(r['year'])}**: {int(r['trips']):,} trips ({pct:.2f}% of mode)")
            lines.append("")

    # Months
    if not df_month.empty:
        for m in sorted(df_month["mode"].unique()):
            sub = df_month[df_month["mode"] == m].copy()
            total = sub["trips"].sum()
            sub = sub.sort_values("trips", ascending=False).head(10)
            lines.append(f"## Top months for mode={m}")
            for _, r in sub.iterrows():
                pct = (float(r["trips"]) / total * 100.0) if total else 0.0
                tpd = r.get("trips_per_day", pd.NA)
                tpd_str = f"{float(tpd):.1f}" if pd.notna(tpd) else "NA"
                lines.append(
                    f"- **{int(r['year'])}-{int(r['month']):02d}**: {int(r['trips']):,} trips "
                    f"({pct:.2f}% of mode), trips/day={tpd_str}"
                )
            lines.append("")

    # Day-of-week (year-aware)
    if not df_dow.empty:
        for m in sorted(df_dow["mode"].unique()):
            subm = df_dow[df_dow["mode"] == m].copy()

            overall = subm.groupby(["dow", "dow_name"], as_index=False)["trips"].sum()
            overall_total = overall["trips"].sum()
            overall = overall.sort_values("trips", ascending=False).head(top_n)

            lines.append(f"## Top days-of-week for mode={m} (overall, all years)")
            for _, r in overall.iterrows():
                pct = (float(r["trips"]) / overall_total * 100.0) if overall_total else 0.0
                lines.append(f"- **{r['dow_name']}**: {int(r['trips']):,} trips ({pct:.2f}% of mode)")
            lines.append("")

            for part in ["weekday", "weekend"]:
                part_df = subm[subm["week_part"] == part].copy()
                if part_df.empty:
                    continue
                part_agg = part_df.groupby(["dow", "dow_name"], as_index=False)["trips"].sum()
                part_total = part_agg["trips"].sum()
                part_agg = part_agg.sort_values("trips", ascending=False).head(top_n)

                lines.append(f"## Top days-of-week for mode={m} ({part}, all years)")
                for _, r in part_agg.iterrows():
                    pct_part = (float(r["trips"]) / part_total * 100.0) if part_total else 0.0
                    lines.append(
                        f"- **{r['dow_name']}**: {int(r['trips']):,} trips "
                        f"({pct_part:.2f}% of {part})"
                    )
                lines.append("")

    # Hours (year-aware)
    if not df_hr.empty:
        for m in sorted(df_hr["mode"].unique()):
            lines.append(f"## Top {top_n} hours for mode={m} (by trips, all years)")
            for part in ["weekday", "weekend"]:
                part_df = df_hr[(df_hr["mode"] == m) & (df_hr["week_part"] == part)].copy()
                if part_df.empty:
                    continue
                part_agg = part_df.groupby(["hour"], as_index=False)["trips"].sum()
                part_total = part_agg["trips"].sum()
                part_agg = part_agg.sort_values("trips", ascending=False).head(top_n)

                lines.append(f"### {part}")
                for _, r in part_agg.iterrows():
                    pct_part = (float(r["trips"]) / part_total * 100.0) if part_total else 0.0
                    lines.append(
                        f"- **{int(r['hour']):02d}:00**: {int(r['trips']):,} trips "
                        f"({pct_part:.2f}% of {part})"
                    )
            lines.append("")

    return lines


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_lines(lines: List[str]) -> None:
    for ln in lines:
        print(ln)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize Citi Bike usage from partitioned parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", required=True, help="Root of run parquet, e.g. data/processed/citibike_parquet/<RUN_TAG>")
    ap.add_argument("--out-dir", required=True, help="Where to write CSV outputs (e.g. summaries/<RUN_TAG>)")
    ap.add_argument("--mode", default=None, help="Optional mode filter, e.g. jc or nyc")

    ap.add_argument("--nypd-crash-csv", default=None, help="Optional NYPD crash CSV (NYC-only) to compute station proximity risk")
    ap.add_argument("--radii-m", default="250,500", help="Crash radii in meters (comma-separated), default 250,500")

    ap.add_argument(
        "--station-exposure",
        choices=["touchpoints", "starts"],
        default="touchpoints",
        help="Exposure definition for citibike_station_exposure.csv: "
             "'touchpoints' = start+end (default), 'starts' = start-only (legacy).",
    )
    ap.add_argument("--top-n", type=int, default=5, help="Top N for year/dow/hour highlights (months always prints top 10)")
    ap.add_argument("--no-print", action="store_true", help="Disable printing highlights to terminal")
    ap.add_argument("--no-highlights", action="store_true", help="Disable writing summary_highlights.md file")
    ap.add_argument("--highlights-md", default=None, help="Path to write highlights markdown. Default: <out-dir>/summary_highlights.md")
    ap.add_argument("--batch-size", type=int, default=250_000, help="Parquet scan batch size (default: 250k rows)")

    # NYC crash-proxy hygiene
    ap.add_argument(
        "--nyc-drop-station-prefixes",
        default="JC,HB",
        help="When --mode=nyc and computing NYPD proxy, drop stations whose IDs start with these prefixes (comma-separated). Default: JC,HB",
    )

    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob("mode=*/year=*/month=*/tripdata.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under: {parquet_dir}")

    mode_filter = str(args.mode).strip() if args.mode else None
    if mode_filter:
        mode_filter = mode_filter.lower()

    # -----------------------------
    # Phase 1: month totals (metadata)
    # -----------------------------
    month_rows: List[dict] = []
    for f in files:
        parts = parse_parts(f)
        if not parts:
            continue
        mode, year, month = parts
        if mode_filter and str(mode).lower() != mode_filter:
            continue
        n = count_rows_parquet(f)
        month_rows.append({"mode": mode, "year": year, "month": month, "trips": int(n)})

    df_month = pd.DataFrame(month_rows).sort_values(["mode", "year", "month"]).reset_index(drop=True)
    if df_month.empty:
        raise SystemExit("No month totals produced. Check parquet layout / mode filter.")

    # -----------------------------
    # Phase 2: minimal scans for active days, DOW, hour, station exposure
    # -----------------------------
    time_candidates = ["start_time", "started_at", "starttime", "Start Time"]

    # START station candidates
    st_id_candidates = ["start_station_id", "start station id"]
    st_name_candidates = ["start_station_name", "start station name"]
    st_lat_candidates = ["start_lat", "start_station_latitude", "start latitude", "start_station_lat"]
    st_lng_candidates = ["start_lng", "start_station_longitude", "start longitude", "start_station_lng", "start_lon", "start_long"]

    # END station candidates
    end_id_candidates = ["end_station_id", "end station id"]
    end_name_candidates = ["end_station_name", "end station name"]
    end_lat_candidates = ["end_lat", "end_station_latitude", "end latitude", "end_station_lat"]
    end_lng_candidates = ["end_lng", "end_station_longitude", "end longitude", "end_station_lng", "end_lon", "end_long"]

    month_active_days: Dict[Tuple[str, int, int], set] = {}
    dow_trips: Dict[Tuple[str, int, int, str, str], int] = {}   # (mode,year,dow,dow_name,week_part)->trips
    hour_trips: Dict[Tuple[str, int, int, str], int] = {}       # (mode,year,hour,week_part)->trips
    station_agg: Dict[Tuple[str, str, int, int], StationAgg2] = {}  # (mode,station_id,year,month)->StationAgg2

    for f in files:
        parts = parse_parts(f)
        if not parts:
            continue
        mode, year, month = parts
        if mode_filter and str(mode).lower() != mode_filter:
            continue

        available_cols = parquet_available_columns(f)
        time_col = first_existing_column(available_cols, time_candidates)
        if not time_col:
            continue

        st_id_col = first_existing_column(available_cols, st_id_candidates)
        st_name_col = first_existing_column(available_cols, st_name_candidates)
        st_lat_col = first_existing_column(available_cols, st_lat_candidates)
        st_lng_col = first_existing_column(available_cols, st_lng_candidates)

        end_id_col = first_existing_column(available_cols, end_id_candidates)
        end_name_col = first_existing_column(available_cols, end_name_candidates)
        end_lat_col = first_existing_column(available_cols, end_lat_candidates)
        end_lng_col = first_existing_column(available_cols, end_lng_candidates)

        cols_to_read = [time_col]
        for c in [st_id_col, st_name_col, st_lat_col, st_lng_col, end_id_col, end_name_col, end_lat_col, end_lng_col]:
            if c and c not in cols_to_read:
                cols_to_read.append(c)

        for batch in iter_parquet_batches(f, columns=cols_to_read, batch_size=args.batch_size):
            df = batch.to_pandas()

            df[time_col] = ensure_datetime_series(df[time_col])
            df = df.dropna(subset=[time_col])
            if df.empty:
                continue

            key_m = (mode, year, month)
            month_active_days.setdefault(key_m, set()).update(df[time_col].dt.date.unique().tolist())

            # DOW + weekday/weekend
            dow = df[time_col].dt.dayofweek.astype(int)  # Mon=0..Sun=6
            dow_name = df[time_col].dt.day_name()
            week_part = dow.isin([5, 6]).map({True: "weekend", False: "weekday"})

            tmp_dow = pd.DataFrame({"dow": dow, "dow_name": dow_name, "week_part": week_part})
            g_dow = tmp_dow.groupby(["dow", "dow_name", "week_part"]).size().reset_index(name="trips")
            for _, rr in g_dow.iterrows():
                k = (mode, year, int(rr["dow"]), str(rr["dow_name"]), str(rr["week_part"]))
                dow_trips[k] = dow_trips.get(k, 0) + int(rr["trips"])

            # Hour
            hour = df[time_col].dt.hour.astype(int)
            tmp_hr = pd.DataFrame({"hour": hour, "week_part": week_part})
            g_hr = tmp_hr.groupby(["hour", "week_part"]).size().reset_index(name="trips")
            for _, rr in g_hr.iterrows():
                k = (mode, year, int(rr["hour"]), str(rr["week_part"]))
                hour_trips[k] = hour_trips.get(k, 0) + int(rr["trips"])

            # Station touchpoints (start + end)
            # STARTS
            if st_id_col and st_name_col and st_lat_col and st_lng_col:
                st = df[[st_id_col, st_name_col, st_lat_col, st_lng_col]].copy()
                st = st.dropna(subset=[st_id_col, st_lat_col, st_lng_col])
                if not st.empty:
                    st = _coerce_numeric_latlng(st, st_lat_col, st_lng_col)
                if not st.empty:
                    g_st = (
                        st.groupby([st_id_col], as_index=False)
                          .agg(
                              station_name=(st_name_col, "first"),
                              trips=(st_id_col, "size"),
                              lat_sum=(st_lat_col, "sum"),
                              lng_sum=(st_lng_col, "sum"),
                              coord_n=(st_lat_col, "size"),
                          )
                    )
                    for _, rr in g_st.iterrows():
                        sid = str(rr[st_id_col])
                        k = (mode, sid, year, month)
                        agg = station_agg.get(k, StationAgg2())
                        agg.start_trips += int(rr["trips"])
                        agg.lat_sum += float(rr["lat_sum"])
                        agg.lng_sum += float(rr["lng_sum"])
                        agg.coord_n += int(rr["coord_n"])
                        if not agg.name and pd.notna(rr["station_name"]):
                            agg.name = str(rr["station_name"])
                        station_agg[k] = agg

            # ENDS
            if end_id_col and end_name_col and end_lat_col and end_lng_col:
                en = df[[end_id_col, end_name_col, end_lat_col, end_lng_col]].copy()
                en = en.dropna(subset=[end_id_col, end_lat_col, end_lng_col])
                if not en.empty:
                    en = _coerce_numeric_latlng(en, end_lat_col, end_lng_col)
                if not en.empty:
                    g_en = (
                        en.groupby([end_id_col], as_index=False)
                          .agg(
                              station_name=(end_name_col, "first"),
                              trips=(end_id_col, "size"),
                              lat_sum=(end_lat_col, "sum"),
                              lng_sum=(end_lng_col, "sum"),
                              coord_n=(end_lat_col, "size"),
                          )
                    )
                    for _, rr in g_en.iterrows():
                        sid = str(rr[end_id_col])
                        k = (mode, sid, year, month)
                        agg = station_agg.get(k, StationAgg2())
                        agg.end_trips += int(rr["trips"])
                        agg.lat_sum += float(rr["lat_sum"])
                        agg.lng_sum += float(rr["lng_sum"])
                        agg.coord_n += int(rr["coord_n"])
                        if not agg.name and pd.notna(rr["station_name"]):
                            agg.name = str(rr["station_name"])
                        station_agg[k] = agg

    # -----------------------------
    # Write MONTH CSV (+ active_days/trips_per_day)
    # -----------------------------
    active_days_col: List[object] = []
    for _, rr in df_month.iterrows():
        k = (str(rr["mode"]), int(rr["year"]), int(rr["month"]))
        ad = len(month_active_days.get(k, set()))
        active_days_col.append(ad if ad > 0 else pd.NA)

    df_month = df_month.copy()
    df_month["active_days"] = active_days_col

    active_days = pd.to_numeric(df_month["active_days"], errors="coerce")
    trips = pd.to_numeric(df_month["trips"], errors="coerce")

    df_month["trips_per_day"] = trips / active_days
    bad = active_days.isna() | (active_days <= 0) | trips.isna()
    df_month.loc[bad, "trips_per_day"] = pd.NA

    month_csv = out_dir / "citibike_trips_by_month.csv"
    df_month.to_csv(month_csv, index=False)
    print(f"Saved -> {month_csv}")

    # -----------------------------
    # Write YEAR CSV (+ yoy)
    # -----------------------------
    df_year = (
        df_month.groupby(["mode", "year"], as_index=False)["trips"]
        .sum()
        .sort_values(["mode", "year"])
        .reset_index(drop=True)
    )
    df_year["yoy_pct"] = df_year.groupby("mode")["trips"].pct_change() * 100.0

    year_csv = out_dir / "citibike_trips_by_year.csv"
    df_year.to_csv(year_csv, index=False)
    print(f"Saved -> {year_csv}")

    # -----------------------------
    # Write DOW CSV (year-aware)
    # -----------------------------
    if dow_trips:
        df_dow = pd.DataFrame(
            [{"mode": k[0], "year": k[1], "dow": k[2], "dow_name": k[3], "week_part": k[4], "trips": v}
             for k, v in dow_trips.items()]
        ).sort_values(["mode", "year", "dow", "week_part"]).reset_index(drop=True)

        totals_mode_year = df_dow.groupby(["mode", "year"])["trips"].transform("sum")
        df_dow["pct_of_mode_year_trips"] = (df_dow["trips"] / totals_mode_year) * 100.0

        totals_part_year = df_dow.groupby(["mode", "year", "week_part"])["trips"].transform("sum")
        df_dow["pct_within_week_part"] = (df_dow["trips"] / totals_part_year) * 100.0

        dow_csv = out_dir / "citibike_trips_by_dow.csv"
        df_dow.to_csv(dow_csv, index=False)
        print(f"Saved -> {dow_csv} (includes year column)")
    else:
        df_dow = pd.DataFrame()

    # -----------------------------
    # Write HOUR CSV (year-aware)
    # -----------------------------
    if hour_trips:
        df_hr = pd.DataFrame(
            [{"mode": k[0], "year": k[1], "hour": k[2], "week_part": k[3], "trips": v}
             for k, v in hour_trips.items()]
        ).sort_values(["mode", "year", "hour", "week_part"]).reset_index(drop=True)

        totals_mode_year = df_hr.groupby(["mode", "year"])["trips"].transform("sum")
        df_hr["pct_of_mode_year_trips"] = (df_hr["trips"] / totals_mode_year) * 100.0

        totals_part_year = df_hr.groupby(["mode", "year", "week_part"])["trips"].transform("sum")
        df_hr["pct_within_week_part"] = (df_hr["trips"] / totals_part_year) * 100.0

        hr_csv = out_dir / "citibike_trips_by_hour.csv"
        df_hr.to_csv(hr_csv, index=False)
        print(f"Saved -> {hr_csv} (includes year column)")
    else:
        df_hr = pd.DataFrame()

    # -----------------------------
    # Write STATION exposure CSV (year/month aware)
    # -----------------------------
    if station_agg:
        st_rows = []
        for (mode, sid, year, month), agg in station_agg.items():
            lat = (agg.lat_sum / agg.coord_n) if agg.coord_n else pd.NA
            lng = (agg.lng_sum / agg.coord_n) if agg.coord_n else pd.NA
            touchpoints = int(agg.start_trips + agg.end_trips)

            trips_out = int(agg.start_trips) if args.station_exposure == "starts" else int(touchpoints)

            st_rows.append(
                {
                    "mode": mode,
                    "year": year,
                    "month": month,
                    # Backward-compatible columns used elsewhere:
                    "start_station_id": sid,
                    "start_station_name": agg.name,
                    "trips": trips_out,
                    # Explicit components:
                    "start_trips": int(agg.start_trips),
                    "end_trips": int(agg.end_trips),
                    "touchpoints": int(touchpoints),
                    # Coordinates:
                    "station_lat": lat,
                    "station_lng": lng,
                }
            )

        df_st = (
            pd.DataFrame(st_rows)
            .sort_values(["mode", "year", "month", "trips"], ascending=[True, True, True, False])
            .reset_index(drop=True)
        )

        st_csv = out_dir / "citibike_station_exposure.csv"
        df_st.to_csv(st_csv, index=False)
        print(f"Saved -> {st_csv} (includes year/month columns)")
        print(f"  Rows: {len(df_st):,} (station-month combinations)")
        print(f"  Unique stations: {df_st['start_station_id'].nunique():,}")
        print(f"  Time span: {df_st['year'].min()}-{df_st['year'].max()}, months: {df_st['month'].min()}-{df_st['month'].max()}")
        
        # ===== ENHANCED FIX: Filter stations by mode using PREFIX + COORDINATES =====
        # This prevents NYC reports from showing JC stations and vice versa
        # Some JC stations don't have JC/HB prefixes (e.g., 3199, 3183), so we use coordinates too
        if mode_filter:
            before_count = len(df_st)
            
            # Approximate geographic boundaries (lat/lng)
            # Jersey City area - tight bounds to avoid border issues
            JC_LAT_MIN, JC_LAT_MAX = 40.6990, 40.7490  
            JC_LNG_MIN, JC_LNG_MAX = -74.0778, -74.0278
            
            if mode_filter.lower() == "nyc":
                # NYC mode: Remove JC/HB prefixed stations AND stations in JC geographic area
                
                # Step 1: Prefix filter
                jc_hb_prefixes = tuple(["JC", "HB"])
                prefix_mask = df_st['start_station_id'].astype(str).str.upper().str.startswith(jc_hb_prefixes)
                
                # Step 2: Geographic filter for JC area (catches numeric-ID JC stations)
                df_st['_lat_temp'] = pd.to_numeric(df_st['station_lat'], errors='coerce')
                df_st['_lng_temp'] = pd.to_numeric(df_st['station_lng'], errors='coerce')
                
                in_jc_area = (
                    (df_st['_lat_temp'] >= JC_LAT_MIN) & 
                    (df_st['_lat_temp'] <= JC_LAT_MAX) &
                    (df_st['_lng_temp'] >= JC_LNG_MIN) & 
                    (df_st['_lng_temp'] <= JC_LNG_MAX)
                )
                
                # Combine: Remove if has JC/HB prefix OR is in JC area
                remove_mask = prefix_mask | in_jc_area
                df_st_filtered = df_st[~remove_mask].copy()
                
                # Clean up temporary columns
                df_st_filtered = df_st_filtered.drop(columns=['_lat_temp', '_lng_temp'], errors='ignore')
                
                prefix_removed = int(prefix_mask.sum())
                geo_removed = int(in_jc_area.sum())
                total_removed = before_count - len(df_st_filtered)
                
                if total_removed > 0:
                    print(f"  ✓ NYC mode: Removed {total_removed:,} JC station-month records from station exposure")
                    print(f"    - {prefix_removed:,} had JC/HB prefix")
                    print(f"    - {geo_removed:,} were in JC geographic area (stations like 3199, 3183)")
                    print(f"    (Cross-system trips and JC stations without JC prefix)")
                
                df_st = df_st_filtered
            
            elif mode_filter.lower() == "jc":
                # JC mode: Keep ONLY JC/HB prefixed stations OR stations in JC geographic area
                
                # Step 1: Prefix filter
                jc_hb_prefixes = tuple(["JC", "HB"])
                prefix_mask = df_st['start_station_id'].astype(str).str.upper().str.startswith(jc_hb_prefixes)
                
                # Step 2: Geographic filter for JC area
                df_st['_lat_temp'] = pd.to_numeric(df_st['station_lat'], errors='coerce')
                df_st['_lng_temp'] = pd.to_numeric(df_st['station_lng'], errors='coerce')
                
                in_jc_area = (
                    (df_st['_lat_temp'] >= JC_LAT_MIN) & 
                    (df_st['_lat_temp'] <= JC_LAT_MAX) &
                    (df_st['_lng_temp'] >= JC_LNG_MIN) & 
                    (df_st['_lng_temp'] <= JC_LNG_MAX)
                )
                
                # Combine: Keep if has JC/HB prefix OR is in JC area
                keep_mask = prefix_mask | in_jc_area
                df_st_filtered = df_st[keep_mask].copy()
                
                # Clean up temporary columns
                df_st_filtered = df_st_filtered.drop(columns=['_lat_temp', '_lng_temp'], errors='ignore')
                
                prefix_kept = int(prefix_mask.sum())
                geo_kept = int(in_jc_area.sum())
                total_removed = before_count - len(df_st_filtered)
                
                if total_removed > 0:
                    print(f"  ✓ JC mode: Removed {total_removed:,} non-JC station-month records from station exposure")
                    print(f"    - {prefix_kept:,} stations kept via JC/HB prefix")
                    print(f"    - {geo_kept:,} stations kept via JC geographic area")
                    print(f"    (Cross-system trips to NYC)")
                
                df_st = df_st_filtered
            
            # Re-save the filtered CSV if any rows were removed
            if len(df_st) != before_count:
                df_st.to_csv(st_csv, index=False)
                print(f"  ✓ Re-saved filtered station exposure -> {st_csv}")
                print(f"  Final rows: {len(df_st):,} (station-month combinations)")
                print(f"  Final unique stations: {df_st['start_station_id'].nunique():,}")
                
                # List some filtered stations for verification
                if mode_filter.lower() == "nyc":
                    print(f"  Verification: No stations 3199, 3183, or JC* should remain")
            else:
                print(f"  ✓ No station filtering needed (all stations already match {mode_filter.upper()} mode)")
    else:
        df_st = pd.DataFrame()

    # -----------------------------
    # Optional: NYPD crash proximity + risk (NYC-only proxy)
    # -----------------------------
    if args.nypd_crash_csv:
        if df_st.empty:
            print("WARNING: No station exposure data computed (skipping crash proximity).")
        else:
            print("Aggregating station data across time for crash proximity analysis...")
            df_st_sorted = df_st.sort_values(
                ["mode", "start_station_id", "start_station_name", "year", "month"]
            ).reset_index(drop=True)

            # NYC hygiene: drop NJ-style station IDs from crash proximity input
            # NOTE: Main station exposure CSV already filtered above (around line 707)
            # This is redundant safety filtering for the crash proximity analysis specifically
            prefixes = [p.strip() for p in str(args.nyc_drop_station_prefixes).split(",") if p.strip()]
            df_st_sorted = drop_nj_like_stations_for_nyc_proxy(df_st_sorted, mode_filter, prefixes)

            # Aggregate across time
            df_st_agg = df_st_sorted.groupby(["mode", "start_station_id", "start_station_name"], as_index=False).agg(
                trips=("trips", "sum"),
                start_trips=("start_trips", "sum"),
                end_trips=("end_trips", "sum"),
                touchpoints=("touchpoints", "sum"),
                station_lat=("station_lat", "mean"),
                station_lng=("station_lng", "mean"),
                year_min=("year", "min"),
                year_max=("year", "max"),
                years_count=("year", "nunique"),
                months_count=("month", "nunique"),
                first_month=("month", lambda x: x.iloc[0]),
                last_month=("month", lambda x: x.iloc[-1]),
            )

            df_st_agg["observation_months"] = (
                (df_st_agg["year_max"] - df_st_agg["year_min"]) * 12 +
                (df_st_agg["last_month"] - df_st_agg["first_month"]) + 1
            )
            df_st_agg["data_quality"] = "lifetime_aggregate"
            df_st_agg["temporal_coverage"] = (
                df_st_agg["months_count"].astype(str) + " months spanning " +
                df_st_agg["year_min"].astype(str) + "-" + df_st_agg["year_max"].astype(str)
            )

            radii_f = parse_radii_m(args.radii_m)

            # If JC mode: proxy unavailable, write zeros (existing behavior)
            if mode_filter and mode_filter == "jc":
                df_risk = df_st_agg.copy()
                for r_m in radii_f:
                    df_risk[f"crashes_within_{int(r_m)}m"] = 0
                    df_risk[f"crashes_within_{int(r_m)}m_per_100k_trips"] = 0.0
                df_risk["risk_proxy_available"] = 0

                risk_csv = out_dir / "station_risk_exposure_plus_crashproximity.csv"
                df_risk.to_csv(risk_csv, index=False)
                print(f"Saved -> {risk_csv} (JC mode: NYPD proxy unavailable; wrote zeros)")
            else:
                nypd_path = Path(args.nypd_crash_csv)
                if not nypd_path.exists():
                    print(f"WARNING: --nypd-crash-csv not found: {nypd_path} (skipping crash proximity)")
                else:
                    if not radii_f:
                        print("WARNING: No valid radii parsed from --radii-m; skipping crash proximity.")
                    else:
                        try:
                            print(f"Computing crash proximity using NYPD CSV: {nypd_path}")
                            print(f"Radii (m): {radii_f}")
                            df_risk = compute_crash_proximity(df_st_agg, nypd_path, radii_m=radii_f)
                            df_risk["risk_proxy_available"] = 1

                            for r_m in radii_f:
                                col = f"crashes_within_{int(r_m)}m"
                                df_risk[f"{col}_per_100k_trips"] = (df_risk[col] / df_risk["trips"]) * 100000.0

                            # Sort by largest-radius rate then trips (helps inspection)
                            sort_col = f"crashes_within_{int(max(radii_f))}m_per_100k_trips"
                            if sort_col in df_risk.columns:
                                df_risk = df_risk.sort_values([sort_col, "trips"], ascending=[False, False]).reset_index(drop=True)

                            risk_csv = out_dir / "station_risk_exposure_plus_crashproximity.csv"
                            df_risk.to_csv(risk_csv, index=False)
                            print(f"Saved -> {risk_csv} (aggregated across time for stable risk estimates)")
                        except Exception as e:
                            print(f"WARNING: Crash proximity skipped due to error: {e}")

    # -----------------------------
    # Highlights
    # -----------------------------
    highlights_lines = build_highlights_markdown(
        df_year=df_year,
        df_month=df_month,
        df_dow=df_dow,
        df_hr=df_hr,
        mode=args.mode,
        top_n=args.top_n,
    )

    if not args.no_print:
        terminal_lines = []
        for ln in highlights_lines:
            ln2 = ln.replace("# ", "").replace("## ", "").replace("### ", "").replace("**", "")
            terminal_lines.append(ln2)
        print_lines(terminal_lines)

    if not args.no_highlights:
        highlights_path = Path(args.highlights_md) if args.highlights_md else (out_dir / "summary_highlights.md")
        write_lines(highlights_path, highlights_lines)
        print(f"Saved -> {highlights_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ingest_tripdata.py

Robust Citi Bike tripdata ingestion (ZIP -> Parquet) supporting:
- NYC monthly ZIPs: YYYYMM-citibike-tripdata(.csv)?.zip
- NYC yearly  ZIPs: YYYY-citibike-tripdata(.csv)?.zip (may contain CSV(s) or nested ZIP(s))
- JC  monthly ZIPs: JC-YYYYMM-citibike-tripdata(.csv)?.zip
- JC  yearly  ZIPs: JC-YYYY-citibike-tripdata(.csv)?.zip (rare, but supported)

Key improvements vs original:
- INPUT VALIDATION: Catches invalid years/months/mode before processing
- STATION ID FILTERING: Filters out cross-system station contamination (NYC/JC)
- Supports JC- prefixed files (or --mode auto).
- Sanitizes CSV headers (no empty/duplicate header crashes).
- Normalizes BOTH legacy + modern schemas to ONE canonical output schema.
- Avoids gc.collect() on every batch (does it occasionally).
- Keeps strict month partition enforcement (optionally keeps null time rows).

Canonical output columns:
  ride_id (string, generated for legacy if missing)
  rideable_type (string)
  started_at (timestamp[us])
  ended_at   (timestamp[us])
  start_station_id (string)
  start_station_name (string)
  end_station_id (string)
  end_station_name (string)
  start_lat (string)
  start_lng (string)
  end_lat   (string)
  end_lng   (string)
  member_casual (string; mapped from legacy usertype if needed)
  usertype (string; legacy raw)
  bike_id  (string)
  tripduration_seconds (string; legacy raw kept as string to avoid cast failures)
  gender (string)
  birth_year (string)

Notes:
- We still force all CSV fields to string at read-time for robustness.
- Timestamp parsing is done after read.
- Station IDs are filtered by mode to prevent cross-contamination
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


# ============================
# STATION ID FILTERING (NEW)
# ============================

# Jersey City and Hoboken station prefixes
JC_STATION_PREFIXES = ("JC", "HB")

def _should_keep_station_id(station_id: str, mode: str) -> bool:
    """
    Determine if a station ID should be kept based on mode.
    
    Args:
        station_id: The station ID to check
        mode: 'nyc' or 'jc'
    
    Returns:
        True if station should be kept, False otherwise
    """
    if not station_id or station_id.strip() == "":
        return True  # Keep empty/null station IDs (will be filtered elsewhere)
    
    station_id_upper = str(station_id).strip().upper()
    is_jc_station = any(station_id_upper.startswith(prefix) for prefix in JC_STATION_PREFIXES)
    
    if mode == "nyc":
        # NYC mode: reject JC/HB prefixed stations
        return not is_jc_station
    elif mode == "jc":
        # JC mode: accept only JC/HB prefixed stations
        return is_jc_station
    else:
        # Unknown mode: keep everything
        return True


def _filter_table_by_station_ids(tbl: pa.Table, mode: str, strict: bool = True) -> pa.Table:
    """
    Filter table rows to keep only those with station IDs matching the mode.
    
    Args:
        tbl: PyArrow table with station ID columns
        mode: 'nyc' or 'jc'
        strict: If True, require BOTH start and end stations to match mode.
                If False, keep if EITHER station matches mode.
    
    Returns:
        Filtered PyArrow table
    """
    if mode not in ("nyc", "jc"):
        return tbl  # No filtering for other modes
    
    if tbl.num_rows == 0:
        return tbl
    
    # Get station ID columns
    start_col = tbl.column("start_station_id") if "start_station_id" in tbl.column_names else None
    end_col = tbl.column("end_station_id") if "end_station_id" in tbl.column_names else None
    
    if start_col is None and end_col is None:
        return tbl  # No station columns to filter on
    
    # Build filter mask
    def check_station_array(arr: pa.ChunkedArray) -> pa.ChunkedArray:
        """Check if station IDs in array match the mode."""
        # Convert to string
        str_arr = pc.cast(arr, pa.string(), safe=False)
        
        # Create mask for each prefix
        masks = []
        for prefix in JC_STATION_PREFIXES:
            # Check if starts with prefix (case-insensitive)
            upper_arr = pc.utf8_upper(str_arr)
            prefix_match = pc.starts_with(upper_arr, prefix)
            masks.append(prefix_match)
        
        # Combine: is_jc = starts with any JC/HB prefix
        is_jc = masks[0]
        for mask in masks[1:]:
            is_jc = pc.or_(is_jc, mask)
        
        if mode == "nyc":
            # NYC mode: keep if NOT jc station
            return pc.invert(is_jc)
        else:  # mode == "jc"
            # JC mode: keep if IS jc station
            return is_jc
    
    # Build combined mask
    if start_col is not None and end_col is not None:
        start_ok = check_station_array(start_col)
        end_ok = check_station_array(end_col)
        
        if strict:
            # Both stations must match mode
            keep_mask = pc.and_(start_ok, end_ok)
        else:
            # Either station can match mode
            keep_mask = pc.or_(start_ok, end_ok)
    elif start_col is not None:
        keep_mask = check_station_array(start_col)
    else:  # end_col is not None
        keep_mask = check_station_array(end_col)
    
    # Filter table
    filtered = tbl.filter(keep_mask)
    
    rows_before = tbl.num_rows
    rows_after = filtered.num_rows
    rows_dropped = rows_before - rows_after
    
    if rows_dropped > 0:
        pct_dropped = (rows_dropped / rows_before) * 100
        print(f"      Station filter ({mode}): dropped {rows_dropped:,} rows ({pct_dropped:.1f}%) with wrong mode")
    
    return filtered


# -----------------------------
# INPUT VALIDATION
# -----------------------------
def validate_years(years: List[int]) -> None:
    """
    Validate year inputs are reasonable (2013-2030).
    
    Raises:
        ValueError: If any year is outside reasonable bounds.
    """
    MIN_YEAR = 2013  # Citi Bike launched June 2013
    MAX_YEAR = 2030  # Future-proof but catch typos
    
    for y in years:
        if not (MIN_YEAR <= y <= MAX_YEAR):
            raise ValueError(
                f"Year out of reasonable range ({MIN_YEAR}-{MAX_YEAR}): {y}\n"
                f"Citi Bike launched in 2013. Check --years argument.\n"
                f"Received: {years}"
            )


def validate_months(months: List[int]) -> None:
    """
    Validate month inputs are in range 1-12.
    
    Raises:
        ValueError: If any month is outside 1-12.
    """
    for m in months:
        if not (1 <= m <= 12):
            raise ValueError(
                f"Month out of range (1-12): {m}\n"
                f"Check --months argument for typos.\n"
                f"Received: {months}"
            )


def validate_mode(mode: str) -> None:
    """
    Validate mode input.
    
    Raises:
        ValueError: If mode is not valid for ingest.
    """
    valid_modes = ["nyc", "jc", "auto"]
    mode_lower = mode.strip().lower()
    
    if mode_lower not in valid_modes:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be one of: {', '.join(valid_modes)}\n"
            f"For ingest, mode must be explicit (nyc/jc) or 'auto' to infer from filenames."
        )


def validate_pipeline_inputs(years: List[int], months: List[int], mode: str) -> None:
    """
    Validate all pipeline inputs in one call.
    
    Raises:
        ValueError: If any validation fails.
    """
    validate_years(years)
    validate_months(months)
    validate_mode(mode)


# -----------------------------
# ZIP member cleanup
# -----------------------------
MACOS_JUNK_PREFIXES = ("__MACOSX/",)
MACOS_JUNK_PATTERNS = (
    re.compile(r"(^|/)\._"),
    re.compile(r"(^|/)\.DS_Store$"),
)


def _norm_zip_name(n: str) -> str:
    return n.replace("\\", "/").strip()


def is_macos_junk(name: str) -> bool:
    name = _norm_zip_name(name)
    if any(name.startswith(p) for p in MACOS_JUNK_PREFIXES):
        return True
    for pat in MACOS_JUNK_PATTERNS:
        if pat.search(name):
            return True
    return False


def _dedupe_preserve_order(names: List[str]) -> List[str]:
    seen = set()
    out = []
    for n in names:
        nn = _norm_zip_name(n)
        key = nn.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nn)
    return out


# Split parts:
#   201804-citibike-tripdata.csv
#   201804-citibike-tripdata_1.csv
_PART_SUFFIX_RE = re.compile(r"^(?P<stem>.*?-citibike-tripdata)(?:_(?P<part>\d+))?\.csv$", re.IGNORECASE)


def _select_month_csv_members(csv_names: List[str]) -> List[str]:
    csv_names = _dedupe_preserve_order(csv_names)

    parts: List[Tuple[int, str]] = []
    base: List[str] = []

    for n in csv_names:
        bn = Path(n).name
        m = _PART_SUFFIX_RE.match(bn)
        if not m:
            base.append(n)
            continue
        part = m.group("part")
        if part is None:
            base.append(n)
        else:
            parts.append((int(part), n))

    if parts:
        parts.sort(key=lambda x: x[0])
        return [n for _, n in parts]
    return csv_names


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    os.replace(str(tmp_path), str(final_path))


def parse_compression(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s in ("none", "off", "no", "false", ""):
        return None
    if s in ("snappy", "gzip", "zstd", "brotli", "lz4"):
        return s
    raise ValueError(f"Unsupported compression: {s}")


# -----------------------------
# ZIP filename patterns (NYC + JC)
# -----------------------------
NYC_YEARLY_RE = re.compile(r"^(?P<year>\d{4})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE)
NYC_MONTHLY_RE = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE)

JC_YEARLY_RE = re.compile(r"^JC-(?P<year>\d{4})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE)
JC_MONTHLY_RE = re.compile(r"^JC-(?P<year>\d{4})(?P<month>\d{2})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE)

INTERNAL_MONTH_ZIP_RE = re.compile(
    r"^(?:JC-)?(?P<year>\d{4})(?P<month>\d{2})-citibike-tripdata(?:\.csv)?\.zip$",
    re.IGNORECASE,
)

ANY_YYYYMM_RE = re.compile(r"(?P<year>20\d{2})(?P<month>0[1-9]|1[0-2])")


# -----------------------------
# Header sanitization + naming
# -----------------------------
_NAME_CLEAN_RE = re.compile(r"[^0-9a-zA-Z_]+")


def _standardize_name(name: str) -> str:
    """Lower, strip, replace spaces/punct with underscore, collapse underscores."""
    n = name.strip().lower()
    n = n.replace("\ufeff", "")
    n = n.replace(" ", "_")
    n = _NAME_CLEAN_RE.sub("_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n


def _sanitize_header(header: List[str]) -> List[str]:
    """
    Ensure all header names are:
    - non-empty
    - unique
    - stable

    If empty, create col_<idx>.
    If duplicate, suffix with __<n>.
    """
    out: List[str] = []
    counts: Dict[str, int] = {}
    for i, raw in enumerate(header):
        raw = "" if raw is None else str(raw)
        h = raw.strip()
        if h.startswith("\ufeff"):
            h = h.lstrip("\ufeff")
        if h == "":
            h = f"col_{i+1}"
        # Keep original header text for Arrow read; we'll standardize later.
        base = h
        key = base.lower()
        if key in counts:
            counts[key] += 1
            base = f"{base}__{counts[key]}"
        else:
            counts[key] = 1
        out.append(base)
    return out


# -----------------------------
# Timestamp parsing
# -----------------------------
LIKELY_TIME_COLUMNS = {"starttime", "stoptime", "started_at", "ended_at", "start_time", "end_time"}
TIME_COL_PRIORITY = ["started_at", "starttime", "start_time"]


def _month_bounds_us(year: int, month: int) -> Tuple[pa.Scalar, pa.Scalar]:
    start_dt = datetime(year, month, 1)
    end_dt = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return (pa.scalar(start_dt, type=pa.timestamp("us")), pa.scalar(end_dt, type=pa.timestamp("us")))


def _choose_time_col(tbl: pa.Table) -> Optional[str]:
    cols = set(tbl.column_names)
    for c in TIME_COL_PRIORITY:
        if c in cols:
            return c
    for c in tbl.column_names:
        if c.strip().lower() in LIKELY_TIME_COLUMNS:
            return c
    return None


def _parse_timestamp_chunked(arr: pa.ChunkedArray) -> pa.ChunkedArray:
    if pa.types.is_timestamp(arr.type):
        return arr

    s_arr = arr
    if not (pa.types.is_string(s_arr.type) or pa.types.is_large_string(s_arr.type)):
        s_arr = pc.cast(s_arr, pa.string(), safe=False)

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    out_chunks: List[pa.Array] = []
    total_valid = 0

    for chunk in s_arr.chunks:
        s = chunk
        if not (pa.types.is_string(s.type) or pa.types.is_large_string(s.type)):
            s = pc.cast(s, pa.string(), safe=False)

        s = pc.utf8_trim_whitespace(s)
        s = pc.replace_substring_regex(s, r"\s+UTC$", "")
        s = pc.replace_substring_regex(s, r"([+-]\d{2}):(\d{2})$", r"\1\2")
        s = pc.replace_substring_regex(s, r"^(.*)\.\d+([Z]|[+-]\d{4})$", r"\1\2")
        s = pc.replace_substring_regex(s, r"^(.*)\.\d+$", r"\1")

        parsed_any = None
        for fmt in fmts:
            parsed = pc.strptime(s, format=fmt, unit="us", error_is_null=True)
            parsed_any = parsed if parsed_any is None else pc.coalesce(parsed_any, parsed)

        total_valid += int(pc.sum(pc.is_valid(parsed_any)).as_py() or 0)
        out_chunks.append(parsed_any)

    if total_valid == 0:
        return s_arr
    return pa.chunked_array(out_chunks, type=pa.timestamp("us"))


# -----------------------------
# Canonicalization (legacy + modern -> one schema)
# -----------------------------
CANONICAL_COLS = [
    ("ride_id", pa.string()),
    ("rideable_type", pa.string()),
    ("started_at", pa.timestamp("us")),
    ("ended_at", pa.timestamp("us")),
    ("start_station_id", pa.string()),
    ("start_station_name", pa.string()),
    ("end_station_id", pa.string()),
    ("end_station_name", pa.string()),
    ("start_lat", pa.string()),
    ("start_lng", pa.string()),
    ("end_lat", pa.string()),
    ("end_lng", pa.string()),
    ("member_casual", pa.string()),
    ("usertype", pa.string()),
    ("bike_id", pa.string()),
    ("tripduration_seconds", pa.string()),
    ("gender", pa.string()),
    ("birth_year", pa.string()),
]
CANONICAL_SCHEMA = pa.schema([pa.field(n, t) for n, t in CANONICAL_COLS])


def _col_or_null(tbl: pa.Table, name: str, typ: pa.DataType) -> pa.ChunkedArray:
    if name in tbl.column_names:
        col = tbl[name]
        if not col.type.equals(typ):
            col = pc.cast(col, typ, safe=False)
        return col
    return pa.chunked_array([pa.nulls(tbl.num_rows, type=typ)], type=typ)


def _map_usertype_to_member_casual(usertype: pa.ChunkedArray) -> pa.ChunkedArray:
    # Values: Subscriber/Customer (legacy). Map to member/casual.
    s = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(usertype, pa.string(), safe=False)))
    is_sub = pc.equal(s, pa.scalar("subscriber"))
    is_cust = pc.equal(s, pa.scalar("customer"))
    out = pc.if_else(is_sub, pa.scalar("member"), pc.if_else(is_cust, pa.scalar("casual"), pa.scalar(None, pa.string())))
    return pc.cast(out, pa.string(), safe=False)


def _make_legacy_ride_id(tbl: pa.Table) -> pa.ChunkedArray:
    """
    Create deterministic ride_id for legacy rows if missing.
    Use a hash of core fields, all as strings.
    """
    import hashlib
    
    # Prefer standardized names (after renaming).
    parts = []
    for k in ["starttime", "stoptime", "bikeid", "start_station_id", "end_station_id", "tripduration"]:
        if k in tbl.column_names:
            col = tbl[k]
            # Combine chunks if this is a ChunkedArray
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            parts.append(pc.cast(col, pa.string(), safe=False))
        else:
            # Create an Array (not ChunkedArray) of nulls
            parts.append(pa.array([None] * tbl.num_rows, type=pa.string()))

    # Build ride IDs using Python (more compatible than PyArrow hash)
    ride_ids = []
    for i in range(tbl.num_rows):
        # Concatenate all parts for this row
        row_parts = []
        for part in parts:
            val = part[i].as_py() if part[i].is_valid else ""
            row_parts.append(str(val) if val is not None else "")
        
        # Join with delimiter and hash
        joined_str = "|".join(row_parts)
        hash_hex = hashlib.md5(joined_str.encode('utf-8')).hexdigest()
        ride_ids.append(hash_hex)
    
    return pa.array(ride_ids, type=pa.string())


def _standardize_table_column_names(tbl: pa.Table) -> pa.Table:
    new_names = [_standardize_name(n) for n in tbl.column_names]
    return tbl.rename_columns(new_names)


def _normalize_to_canonical(tbl: pa.Table, mode: str) -> pa.Table:
    """
    Accept a raw table with standardized names and string-typed columns.
    Produce a canonical schema table.
    
    Args:
        tbl: Input table with standardized column names
        mode: 'nyc' or 'jc' - used for station filtering
    
    Returns:
        Table with canonical schema, filtered by mode
    """
    if tbl.num_rows == 0:
        return pa.Table.from_arrays([pa.nulls(0, type=t) for _, t in CANONICAL_COLS], schema=CANONICAL_SCHEMA)

    # Parse timestamps from whichever columns exist
    # Legacy: starttime/stoptime
    # Modern: started_at/ended_at
    started = None
    ended = None

    # Try modern columns first
    if "started_at" in tbl.column_names:
        started = _parse_timestamp_chunked(tbl["started_at"])
    elif "starttime" in tbl.column_names:
        started = _parse_timestamp_chunked(tbl["starttime"])
    elif "start_time" in tbl.column_names:
        started = _parse_timestamp_chunked(tbl["start_time"])

    if "ended_at" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["ended_at"])
    elif "stoptime" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["stoptime"])
    elif "end_time" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["end_time"])

    # Default to nulls if not found
    if started is None:
        started = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.timestamp("us"))])
    if ended is None:
        ended = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.timestamp("us"))])

    # ride_id
    if "ride_id" in tbl.column_names:
        ride_id = pc.cast(tbl["ride_id"], pa.string(), safe=False)
    else:
        # Legacy: generate from hash
        ride_id = _make_legacy_ride_id(tbl)

    # rideable_type
    rideable_type = _col_or_null(tbl, "rideable_type", pa.string())

    # Station IDs and names
    start_station_id = _col_or_null(tbl, "start_station_id", pa.string())
    start_station_name = _col_or_null(tbl, "start_station_name", pa.string())
    end_station_id = _col_or_null(tbl, "end_station_id", pa.string())
    end_station_name = _col_or_null(tbl, "end_station_name", pa.string())

    # Coordinates
    start_lat = _col_or_null(tbl, "start_lat", pa.string())
    start_lng = _col_or_null(tbl, "start_lng", pa.string())
    end_lat = _col_or_null(tbl, "end_lat", pa.string())
    end_lng = _col_or_null(tbl, "end_lng", pa.string())

    # Member/casual
    if "member_casual" in tbl.column_names:
        member_casual = pc.cast(tbl["member_casual"], pa.string(), safe=False)
    elif "usertype" in tbl.column_names:
        member_casual = _map_usertype_to_member_casual(tbl["usertype"])
    else:
        member_casual = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.string())])

    # Legacy fields
    usertype = _col_or_null(tbl, "usertype", pa.string())
    
    # bike_id (legacy: bikeid)
    if "bike_id" in tbl.column_names:
        bike_id = pc.cast(tbl["bike_id"], pa.string(), safe=False)
    elif "bikeid" in tbl.column_names:
        bike_id = pc.cast(tbl["bikeid"], pa.string(), safe=False)
    else:
        bike_id = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.string())])

    tripduration_seconds = _col_or_null(tbl, "tripduration", pa.string())
    gender = _col_or_null(tbl, "gender", pa.string())
    
    # birth_year (legacy: birth_year)
    if "birth_year" in tbl.column_names:
        birth_year = pc.cast(tbl["birth_year"], pa.string(), safe=False)
    else:
        birth_year = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.string())])

    # Build canonical table
    canonical = pa.Table.from_arrays(
        [
            ride_id,
            rideable_type,
            started,
            ended,
            start_station_id,
            start_station_name,
            end_station_id,
            end_station_name,
            start_lat,
            start_lng,
            end_lat,
            end_lng,
            member_casual,
            usertype,
            bike_id,
            tripduration_seconds,
            gender,
            birth_year,
        ],
        schema=CANONICAL_SCHEMA,
    )
    
    # *** CRITICAL: Filter by station IDs based on mode ***
    canonical = _filter_table_by_station_ids(canonical, mode, strict=True)
    
    return canonical


# -----------------------------
# CSV reading helpers
# -----------------------------
def _extract_yyyymm(filename: str) -> Optional[Tuple[int, int]]:
    """Extract YYYYMM from filename if present."""
    m = ANY_YYYYMM_RE.search(filename)
    if not m:
        return None
    y = int(m.group("year"))
    mo = int(m.group("month"))
    return (y, mo)


def read_csv_to_arrow(
    path_or_bio,
    chunk_size: int = 500_000,
) -> Iterator[pa.Table]:
    """
    Read CSV with robust header detection, type coercion, yielding Arrow tables.
    All columns forced to string initially.
    """
    peek_opts = pacsv.ReadOptions(block_size=2048)
    peek_buf = pacsv.open_csv(path_or_bio, read_options=peek_opts)
    first_chunk = peek_buf.read_next_batch()
    raw_header = first_chunk.schema.names

    clean_header = _sanitize_header(raw_header)
    col_types = {n: pa.string() for n in clean_header}

    read_opts = pacsv.ReadOptions(block_size=chunk_size, column_names=clean_header)
    convert_opts = pacsv.ConvertOptions(column_types=col_types, strings_can_be_null=True)
    parse_opts = pacsv.ParseOptions(delimiter=",")

    reader = pacsv.open_csv(path_or_bio, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)
    for batch in reader:
        tbl = pa.Table.from_batches([batch])
        yield tbl


def iter_tables_from_csv_member(
    zf: zipfile.ZipFile,
    csv_name: str,
    enforce_year: int,
    enforce_month: int,
    keep_null_times: bool,
    mode: str,
) -> Iterator[pa.Table]:
    """
    Read CSV from ZIP member, standardize names, filter by month, yield Arrow tables.
    
    Args:
        zf: Open ZipFile object
        csv_name: Name of CSV member in ZIP
        enforce_year: Expected year for filtering
        enforce_month: Expected month for filtering
        keep_null_times: Whether to keep rows with null timestamps
        mode: 'nyc' or 'jc' for station filtering
    """
    with zf.open(csv_name) as bio:
        for raw_tbl in read_csv_to_arrow(bio, chunk_size=500_000):
            if raw_tbl.num_rows == 0:
                continue

            # Standardize column names
            tbl = _standardize_table_column_names(raw_tbl)

            # Normalize to canonical schema (includes station filtering)
            tbl = _normalize_to_canonical(tbl, mode=mode)

            if tbl.num_rows == 0:
                continue

            # Filter by time if enforce_month is set
            if enforce_month:
                start_col = tbl["started_at"]
                if pa.types.is_timestamp(start_col.type):
                    lower, upper = _month_bounds_us(enforce_year, enforce_month)
                    in_month = pc.and_(pc.greater_equal(start_col, lower), pc.less(start_col, upper))
                    
                    if not keep_null_times:
                        is_valid_time = pc.is_valid(start_col)
                        keep = pc.and_(in_month, is_valid_time)
                    else:
                        # Keep nulls OR in-month
                        is_null_time = pc.is_null(start_col)
                        keep = pc.or_(in_month, is_null_time)
                    
                    tbl = tbl.filter(keep)

            if tbl.num_rows > 0:
                yield tbl


def _iter_tables_from_zip_member_for_month(
    zf: zipfile.ZipFile,
    zip_name: str,
    year: int,
    month: int,
    keep_null_times: bool,
    mode: str,
    max_depth: int = 2,
) -> Iterator[pa.Table]:
    """
    Recursively extract CSV from nested ZIP member.
    """
    if max_depth <= 0:
        return

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd) / "nested.zip"
        with zf.open(zip_name) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        with zipfile.ZipFile(tmp_path) as inner_zf:
            names = [n for n in inner_zf.namelist() if not is_macos_junk(n)]
            names = _dedupe_preserve_order(names)

            csvs = [n for n in names if n.lower().endswith(".csv")]
            zips = [n for n in names if n.lower().endswith(".zip")]

            for csv_name in csvs:
                yield from iter_tables_from_csv_member(
                    inner_zf, csv_name, year, month, keep_null_times, mode
                )

            for nested_zip_name in zips:
                yield from _iter_tables_from_zip_member_for_month(
                    inner_zf, nested_zip_name, year, month, keep_null_times, mode, max_depth - 1
                )


# -----------------------------
# Parquet writing (atomic)
# -----------------------------
def write_parquet_atomic_tables(
    out_path: Path,
    table_iter: Iterator[pa.Table],
    force: bool,
    compression: Optional[str],
) -> int:
    """
    Write an iterator of Arrow tables to a single parquet file atomically.
    Returns total rows written.
    """
    if out_path.exists() and not force:
        return 0

    ensure_dir(out_path.parent)
    tmp_path = out_path.parent / f"{out_path.name}.tmp.{os.getpid()}"

    try:
        writer = None
        total_rows = 0

        for i, tbl in enumerate(table_iter):
            if tbl.num_rows == 0:
                continue

            # Ensure schema matches canonical
            if not tbl.schema.equals(CANONICAL_SCHEMA):
                cols = []
                for field in CANONICAL_SCHEMA:
                    if field.name in tbl.column_names:
                        col = tbl[field.name]
                        if not col.type.equals(field.type):
                            col = pc.cast(col, field.type, safe=False)
                    else:
                        col = pa.nulls(tbl.num_rows, type=field.type)
                    cols.append(col)
                tbl = pa.Table.from_arrays(cols, schema=CANONICAL_SCHEMA)

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema=CANONICAL_SCHEMA,
                    compression=compression,
                    version="2.6",
                    use_dictionary=True,
                    write_statistics=True,
                )

            writer.write_table(tbl)
            total_rows += tbl.num_rows

            # Occasional GC (every 10 batches)
            if (i + 1) % 10 == 0:
                gc.collect()

        if writer is None:
            return 0

        writer.close()
        writer = None
        atomic_replace(tmp_path, out_path)
        return total_rows

    except Exception:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


# -----------------------------
# Meta + ingestion
# -----------------------------
@dataclass(frozen=True)
class ZipMeta:
    mode: str          # nyc | jc
    kind: str          # yearly | monthly
    year: int
    month: Optional[int]
    path: Path


def detect_zip_meta(p: Path, mode: str) -> Optional[ZipMeta]:
    """
    mode can be 'nyc', 'jc', or 'auto'.
    If auto: infer mode from filename prefix.
    """
    name = p.name

    if mode not in ("nyc", "jc", "auto"):
        raise ValueError("mode must be one of: nyc, jc, auto")

    inferred_mode = "jc" if name.lower().startswith("jc-") else "nyc"
    use_mode = inferred_mode if mode == "auto" else mode

    # Only accept files matching the chosen mode
    if use_mode == "nyc":
        m = NYC_YEARLY_RE.match(name)
        if m:
            return ZipMeta(mode="nyc", kind="yearly", year=int(m.group("year")), month=None, path=p)
        m = NYC_MONTHLY_RE.match(name)
        if m:
            return ZipMeta(mode="nyc", kind="monthly", year=int(m.group("year")), month=int(m.group("month")), path=p)
        return None

    # JC
    m = JC_YEARLY_RE.match(name)
    if m:
        return ZipMeta(mode="jc", kind="yearly", year=int(m.group("year")), month=None, path=p)
    m = JC_MONTHLY_RE.match(name)
    if m:
        return ZipMeta(mode="jc", kind="monthly", year=int(m.group("year")), month=int(m.group("month")), path=p)
    return None


def month_out_path(out_dir: Path, mode: str, year: int, month: int) -> Path:
    return out_dir / f"mode={mode}" / f"year={year}" / f"month={month:02d}" / "tripdata.parquet"


def ingest_monthly_zip(
    meta: ZipMeta,
    out_dir: Path,
    months_filter: Optional[set],
    force: bool,
    compression: Optional[str],
    keep_null_times: bool,
) -> None:
    assert meta.kind == "monthly" and meta.month is not None

    if months_filter is not None and meta.month not in months_filter:
        return

    out_path = month_out_path(out_dir, meta.mode, meta.year, meta.month)
    if out_path.exists() and not force:
        print(f" - {meta.path.name} -> exists, skipping: {out_path}")
        return

    print(f" - {meta.path.name} (mode={meta.mode}, {meta.year}-{meta.month:02d})")

    with zipfile.ZipFile(meta.path) as zf:
        names = [n for n in zf.namelist() if not is_macos_junk(n)]
        names = _dedupe_preserve_order(names)

        csvs = [n for n in names if n.lower().endswith(".csv")]
        inner_zips = [n for n in names if n.lower().endswith(".zip")]

        if not csvs and not inner_zips:
            print("   -> no CSVs or nested ZIPs found, skipping.")
            return

        if csvs:
            csvs = _select_month_csv_members(csvs)
            print(f"   -> month={meta.month:02d}: {len(csvs)} CSV file(s) selected")
            for n in csvs:
                print(f"      reading: {Path(n).name}")

        if (not csvs) and inner_zips:
            print(f"   -> month={meta.month:02d}: no CSVs; found {len(inner_zips)} nested ZIP(s)")
            for n in inner_zips:
                print(f"      opening nested zip: {Path(n).name}")

        def table_iter() -> Iterator[pa.Table]:
            for csv_name in csvs:
                yield from iter_tables_from_csv_member(
                    zf,
                    csv_name,
                    enforce_year=meta.year,
                    enforce_month=meta.month,
                    keep_null_times=keep_null_times,
                    mode=meta.mode,  # Pass mode for station filtering
                )
            for zn in inner_zips:
                yield from _iter_tables_from_zip_member_for_month(
                    zf, zn, meta.year, meta.month, keep_null_times=keep_null_times, mode=meta.mode, max_depth=2
                )

        rows = write_parquet_atomic_tables(out_path, table_iter(), force=force, compression=compression)

    if rows > 0:
        print(f"      -> wrote {out_path} (rows={rows:,})")


def ingest_yearly_zip(
    meta: ZipMeta,
    out_dir: Path,
    months_filter: Optional[set],
    force: bool,
    compression: Optional[str],
    keep_null_times: bool,
) -> None:
    assert meta.kind == "yearly"
    print(f" - {meta.path.name} (mode={meta.mode}, {meta.year}, YEARLY ZIP)")

    with zipfile.ZipFile(meta.path) as zf:
        names = [n for n in zf.namelist() if not is_macos_junk(n)]
        names = _dedupe_preserve_order(names)

        csv_members = [n for n in names if n.lower().endswith(".csv")]
        zip_members = [n for n in names if n.lower().endswith(".zip")]

        if not csv_members and not zip_members:
            print("   -> no CSVs or nested ZIPs found, skipping.")
            return

        # Direct CSVs grouped by month using YYYYMM anywhere in filename
        by_month_csv: Dict[int, List[str]] = {}
        for n in csv_members:
            got = _extract_yyyymm(Path(n).name)
            if not got:
                continue
            y, mo = got
            if y == meta.year:
                by_month_csv.setdefault(mo, []).append(n)

        # Nested ZIPs grouped by month using their zip filename
        by_month_zip: Dict[int, List[str]] = {}
        for n in zip_members:
            m = INTERNAL_MONTH_ZIP_RE.match(Path(n).name)
            if not m:
                continue
            y = int(m.group("year"))
            mo = int(m.group("month"))
            if y == meta.year:
                by_month_zip.setdefault(mo, []).append(n)

        months_present = sorted(set(by_month_csv.keys()) | set(by_month_zip.keys()))
        if not months_present:
            print("   -> no month-like CSVs / nested zips found inside yearly ZIP; skipping.")
            return

        for mo in months_present:
            if months_filter is not None and mo not in months_filter:
                continue

            out_path = month_out_path(out_dir, meta.mode, meta.year, mo)
            if out_path.exists() and not force:
                print(f"   -> {meta.year}-{mo:02d}: exists, skipping: {out_path}")
                continue

            csvs = _select_month_csv_members(by_month_csv.get(mo, []))
            zips = _dedupe_preserve_order(by_month_zip.get(mo, []))

            print(f"   -> month={mo:02d}: {len(csvs)} CSV file(s) selected, {len(zips)} nested ZIP(s)")
            for n in csvs:
                print(f"      reading: {Path(n).name}")
            for n in zips:
                print(f"      opening nested zip: {Path(n).name}")

            def table_iter() -> Iterator[pa.Table]:
                for csv_name in csvs:
                    yield from iter_tables_from_csv_member(
                        zf,
                        csv_name,
                        enforce_year=meta.year,
                        enforce_month=mo,
                        keep_null_times=keep_null_times,
                        mode=meta.mode,  # Pass mode for station filtering
                    )
                for zip_name in zips:
                    yield from _iter_tables_from_zip_member_for_month(
                        zf, zip_name, meta.year, mo, keep_null_times=keep_null_times, mode=meta.mode, max_depth=2
                    )

            rows = write_parquet_atomic_tables(out_path, table_iter(), force=force, compression=compression)
            if rows > 0:
                print(f"      -> wrote {out_path} (rows={rows:,})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    faulthandler.enable(all_threads=True)

    ap = argparse.ArgumentParser(
        description="Ingest Citi Bike trip data from ZIP files to partitioned Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/ingest_tripdata.py --raw-dir data/raw/citibike/y2024_m1_modenyc \\
      --out-dir data/processed/citibike_parquet/y2024_m1_modenyc \\
      --mode nyc --years 2024 --months 1

  python src/ingest_tripdata.py --raw-dir data/raw/citibike/jc_2023 \\
      --out-dir data/processed/citibike_parquet/jc_2023 \\
      --mode jc --years 2023 --months 1 2 3 --force
        """
    )
    ap.add_argument("--raw-dir", required=True, type=Path, help="Directory containing downloaded ZIP files")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for partitioned parquet")
    ap.add_argument("--mode", required=True, type=str, help="Mode: nyc|jc|auto")
    ap.add_argument("--years", nargs="+", required=True, type=int, help="Years to process (e.g., 2023 2024)")
    ap.add_argument("--months", nargs="+", required=True, type=int, help="Months to process (1-12)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing parquet files")
    ap.add_argument("--keep-null-times", action="store_true", help="Keep rows with null started_at in month partitions")
    ap.add_argument(
        "--compression",
        type=str,
        default="none",
        help="Parquet compression: none|snappy|gzip|zstd|brotli|lz4 (default: none)",
    )

    # kept for compatibility (ignored)
    ap.add_argument("--chunksize", type=int, default=250_000, help="(ignored) kept for compatibility")
    ap.add_argument("--csv-engine", choices=["python", "c"], default="python", help="(ignored)")
    ap.add_argument("--extract-csv", action="store_true", help="(ignored)")

    args = ap.parse_args(argv)

    # ========== VALIDATE INPUTS ==========
    print("=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)
    try:
        validate_pipeline_inputs(args.years, args.months, args.mode)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"✓ Valid years: {sorted(args.years)}")
    print(f"✓ Valid months: {sorted(args.months)}")
    print(f"✓ Valid mode: {args.mode}")
    print()

    # conservative env defaults
    os.environ.setdefault("ARROW_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    mode: str = args.mode.strip().lower()
    years = set(args.years)
    months_filter = set(args.months) if args.months else None
    force: bool = bool(args.force)
    keep_null_times: bool = bool(args.keep_null_times)
    compression: Optional[str] = parse_compression(args.compression)

    if not raw_dir.exists():
        print(f"ERROR: raw-dir does not exist: {raw_dir}", file=sys.stderr)
        return 2

    zips = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"])
    metas: List[ZipMeta] = []
    for p in zips:
        meta = detect_zip_meta(p, mode=mode)
        if meta is None:
            continue
        if meta.year not in years:
            continue
        metas.append(meta)

    if not metas:
        print("No ZIPs selected for ingestion (check raw-dir / filters / mode).")
        return 0

    print("=" * 70)
    print(f"INGESTION PLAN")
    print("=" * 70)
    print(f"Raw dir:       {raw_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Mode:          {mode}")
    print(f"Years:         {sorted(years)}")
    print(f"Months filter: {sorted(months_filter) if months_filter else 'ALL'}")
    print(f"Compression:   {compression or 'NONE'}")
    print(f"Force:         {force}")
    print(f"ZIPs found:    {len(metas)}")
    print(f"Station filtering: ENABLED (mode={mode})")
    print()

    print("=" * 70)
    print(f"PROCESSING {len(metas)} ZIP(S)")
    print("=" * 70)

    metas.sort(key=lambda m: (m.mode, m.year, 0 if m.kind == "monthly" else 1, m.month or 0))

    for meta in metas:
        if meta.kind == "monthly":
            ingest_monthly_zip(meta, out_dir, months_filter, force, compression, keep_null_times)
        else:
            ingest_yearly_zip(meta, out_dir, months_filter, force, compression, keep_null_times)

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"✓ Station filtering ensured {mode.upper()} stations only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

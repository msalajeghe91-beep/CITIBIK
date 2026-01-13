#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

NUMERIC_COLS = ["start_lat", "start_lng", "end_lat", "end_lng"]
TIME_COLS = ["started_at", "ended_at"]
STRING_COLS = ["ride_id", "start_station_id", "end_station_id"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True, type=Path)
    ap.add_argument("--out-parquet", required=True, type=Path)
    ap.add_argument("--compression", default="snappy")
    args = ap.parse_args()

    table = pq.read_table(args.in_parquet)

    cols = []
    for name in table.column_names:
        col = table[name]
        lname = name.lower()

        if lname in STRING_COLS:
            col = pc.cast(col, pa.string(), safe=False)

        if lname in NUMERIC_COLS:
            # parse numeric strings; invalid -> null
            col = pc.cast(col, pa.float64(), safe=False)

        # TIME_COLS already parsed to timestamp in ingest; keep as-is
        cols.append(col)

    out = pa.table(cols, names=table.column_names)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out_parquet, compression=None if args.compression=="none" else args.compression)
    print(f"Wrote {args.out_parquet} rows={out.num_rows:,}")

if __name__ == "__main__":
    main()

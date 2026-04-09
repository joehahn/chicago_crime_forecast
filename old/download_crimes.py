#!/usr/bin/env python3
"""Download Chicago crime data from the Socrata API for the past 4 years."""

import csv
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
PAGE_SIZE = 50000
import os
os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/crimes.csv"

cutoff = (datetime.now() - timedelta(days=4 * 365)).strftime("%Y-%m-%dT%H:%M:%S")
print(f"Fetching records since {cutoff}")

start_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
total = start_offset
offset = start_offset
header_written = start_offset > 0
file_mode = "a" if start_offset > 0 else "w"

with open(OUTPUT_FILE, file_mode, newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)

    while True:
        params = urllib.parse.urlencode({
            "$where": f"date >= '{cutoff}'",
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": "date DESC",
        })
        url = f"{BASE_URL}?{params}"
        print(f"  Fetching offset={offset} ...", end=" ", flush=True)

        req = urllib.request.Request(url, headers={"Accept": "text/csv"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"timeout/error (attempt {attempt+1}), retrying... ({e})", end=" ", flush=True)

        lines = raw.splitlines()
        if len(lines) <= 1:
            print("done (no more records)")
            break

        reader = csv.reader(lines)
        rows = list(reader)
        header = rows[0]
        data_rows = rows[1:]

        if not header_written:
            writer.writerow(header)
            header_written = True

        writer.writerows(data_rows)
        count = len(data_rows)
        total += count
        print(f"got {count} rows (total: {total})")

        if count < PAGE_SIZE:
            break
        offset += PAGE_SIZE

print(f"\nDone. {total} records saved to {OUTPUT_FILE}")

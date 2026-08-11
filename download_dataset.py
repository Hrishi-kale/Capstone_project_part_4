#!/usr/bin/env python3
"""
download_dataset.py — Fetches the UCI Phishing Websites dataset (11,055
samples, 30 features, binary label) and saves it to data/phishing_raw.csv.

The dataset is small enough (~800 KB) that a copy is committed directly to
this repository under data/phishing_raw.csv, so running this script is
optional — it exists for reproducibility / in case you want a fresh copy.

Source: UCI Machine Learning Repository, "Phishing Websites" dataset
(Mohammad, Thabtah & McCluskey, 2015), mirrored here in CSV form.
"""

import os
import sys

import requests

DATASET_URL = (
    "https://raw.githubusercontent.com/sachinshubhams/Website-Phishing/"
    "main/csv_result-Training%20Dataset.csv"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "phishing_raw.csv")


def main():
    print(f"Downloading dataset from {DATASET_URL} ...")
    try:
        resp = requests.get(DATASET_URL, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Error: download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(resp.content)

    print(f"Saved {len(resp.content):,} bytes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
virustotal_check.py — Enriches public IPv4 addresses extracted from a log
file with reputation data from the VirusTotal public API v3.

The VirusTotal API key is NEVER hardcoded: it is read from the VT_API_KEY
environment variable (see .env.example). If the variable is not set, the
script exits with a clear error instead of silently failing later.

Usage:
    export VT_API_KEY="your-key-here"
    python3 virustotal_check.py <path_to_log_file>

This script reuses the IP-extraction logic from log_enricher.py so both
tools agree on what counts as a "unique public IP".
"""

import argparse
import json
import os
import sys

import requests

from log_enricher import extract_public_ips

VT_URL = "https://www.virustotal.com/api/v3/ip_addresses/{ip}"


def load_api_key() -> str:
    """Read the VirusTotal API key from the environment. Exit with a clear
    error if it isn't set, rather than sending an unauthenticated request."""
    api_key = os.environ.get("VT_API_KEY")
    if not api_key:
        print(
            "Error: VT_API_KEY environment variable is not set.\n"
            "Set it with: export VT_API_KEY=\"your-key-here\"\n"
            "(see .env.example for the expected variable name)",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def query_virustotal(ip: str, api_key: str, timeout: float = 10.0) -> dict:
    """
    Query VirusTotal for a single IP and return the fields the task cares
    about: malicious vendor count, harmless vendor count, and last analysis
    date. Any failure (rate limit, invalid key, 404, network error, bad
    JSON) is caught and returned as a friendly error message instead of
    raising, so one bad lookup can't crash a batch run.
    """
    headers = {"x-apikey": api_key}
    try:
        resp = requests.get(VT_URL.format(ip=ip), headers=headers, timeout=timeout)

        if resp.status_code == 401:
            return {"error": "invalid or unauthorized VirusTotal API key"}
        if resp.status_code == 404:
            return {"error": "IP not found in VirusTotal's database"}
        if resp.status_code == 429:
            return {"error": "VirusTotal rate limit exceeded — try again later"}
        resp.raise_for_status()

        data = resp.json()
        attributes = data["data"]["attributes"]
        stats = attributes.get("last_analysis_stats", {})
        last_analysis_ts = attributes.get("last_analysis_date")

        return {
            "malicious_detections": stats.get("malicious"),
            "harmless_detections": stats.get("harmless"),
            "last_analysis_date": last_analysis_ts,  # Unix timestamp; see README for formatting note
        }
    except requests.exceptions.RequestException as exc:
        return {"error": f"request failed: {exc}"}
    except (KeyError, json.JSONDecodeError):
        return {"error": "unexpected response format from VirusTotal"}


def enrich_with_virustotal(ips: set, api_key: str) -> dict:
    return {ip: query_virustotal(ip, api_key) for ip in sorted(ips)}


def main():
    parser = argparse.ArgumentParser(description="Enrich public IPs from a log file using VirusTotal.")
    parser.add_argument("log_path", help="Path to a plain-text log file (syslog/firewall format)")
    args = parser.parse_args()

    api_key = load_api_key()

    try:
        with open(args.log_path, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
    except OSError as exc:
        print(f"Error: could not read log file '{args.log_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    public_ips = extract_public_ips(log_text)
    if not public_ips:
        print("No public IPv4 addresses found in the log file.")
        return

    print(f"Querying VirusTotal for {len(public_ips)} unique public IP(s)...\n")
    results = enrich_with_virustotal(public_ips, api_key)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

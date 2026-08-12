#!/usr/bin/env python3
"""
log_enricher.py — Extracts public IPv4 addresses from a log file and
enriches each unique one with geolocation / ISP / hosting-proxy-VPN data
from the free ip-api.com REST API.

Usage:
    python3 log_enricher.py <path_to_log_file>

Example:
    python3 log_enricher.py sample_logs/sample.log
"""

import argparse
import ipaddress
import json
import re
import sys
import time

import requests

# Matches four dot-separated groups of 1-3 digits. This is deliberately a
# *syntactic* match (it will match e.g. 999.999.999.999) — semantic
# validation of whether it's a real, routable, public address is handled
# afterwards by the ipaddress module, which is far less error-prone than
# hand-rolling that logic in the regex itself.
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

IP_API_URL = "http://ip-api.com/json/{ip}?fields=status,message,country,isp,hosting,proxy,mobile"

def extract_public_ips(log_text: str) -> set:
    """
    Extract all IPv4 addresses from log_text, discard anything that is not
    a valid, publicly routable unicast address (private ranges like
    10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, loopback, link-local, etc.
    are skipped), and return the deduplicated set of public IPs.
    """
    candidates = IPV4_PATTERN.findall(log_text)
    public_ips = set()  # a set gives us O(1) deduplication for free

    for candidate in candidates:
        try:
            addr = ipaddress.IPv4Address(candidate)
        except ipaddress.AddressValueError:
            continue  # not a real IPv4 address (e.g. octet > 255) — skip

        # is_private covers RFC1918 (10/8, 172.16/12, 192.168/16) as well
        # as loopback/link-local; is_global is the inverse of "internal"
        # ranges such as CGNAT (100.64/10) and multicast. Skipping both
        # keeps us aligned with the task's private-range exclusion list
        # while also not shipping obviously-non-public junk to a public API.
        if addr.is_private or not addr.is_global:
            continue

        public_ips.add(str(addr))

    return public_ips


def query_ip_api(ip: str, session: requests.Session, timeout: float = 5.0) -> dict:
    """
    Query ip-api.com for a single IP and return the fields we care about.
    Returns a dict with an 'error' key on any failure instead of raising,
    so one bad lookup can't crash the whole enrichment run.
    """
    try:
        resp = session.get(IP_API_URL.format(ip=ip), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": f"request failed: {exc}"}
    except json.JSONDecodeError:
        return {"error": "response was not valid JSON"}

    if data.get("status") != "success":
        return {"error": data.get("message", "lookup failed")}

    return {
        "country": data.get("country"),
        "isp": data.get("isp"),
        "hosting": data.get("hosting"),
        "proxy": data.get("proxy"),
        "mobile": data.get("mobile"),
    }


def enrich_ips(ips: set) -> dict:
    """Enrich each IP in `ips` and return {ip: enrichment_dict}."""
    results = {}
    with requests.Session() as session:
        for ip in sorted(ips):
            results[ip] = query_ip_api(ip, session)
            # ip-api.com's free tier is rate-limited (45 req/min); a small
            # delay keeps a large log file from tripping that limit.
            time.sleep(1.5)
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract public IPs from a log file and enrich them via ip-api.com.")
    parser.add_argument("log_path", help="Path to a plain-text log file (syslog/firewall format)")
    args = parser.parse_args()

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

    print(f"Found {len(public_ips)} unique public IP(s): {', '.join(sorted(public_ips))}")
    print("Querying ip-api.com for enrichment (this may take a few seconds)...\n")

    enrichment = enrich_ips(public_ips)
    print(json.dumps(enrichment, indent=2))


if __name__ == "__main__":
    main()

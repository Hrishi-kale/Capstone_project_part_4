#!/usr/bin/env python3
"""
port_scanner.py — Multithreaded TCP connect-scanner with banner grabbing.

Uses only the Python standard library (socket + threading) — no nmap,
no subprocess calls to external scanner binaries. Intended for scanning
hosts/networks you own or are explicitly authorized to test.

Usage:
    python3 port_scanner.py <target_ip> <start_port>-<end_port> [--timeout SECONDS] [--workers N]

Example:
    python3 port_scanner.py 192.168.1.10 1-1024
    python3 port_scanner.py 192.168.1.10 1-1024 --timeout 0.5 --workers 200
"""

import argparse
import socket
import threading
import sys


def parse_port_range(port_range_str: str):
    """Parse a 'start-end' string into a (start, end) tuple of ints."""
    try:
        start_str, end_str = port_range_str.split("-")
        start, end = int(start_str), int(end_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Port range must look like '1-1024', got '{port_range_str}'"
        ) from exc

    if not (0 < start <= end <= 65535):
        raise argparse.ArgumentTypeError(
            "Port range must satisfy 0 < start <= end <= 65535"
        )
    return start, end


def grab_banner(sock: socket.socket) -> str:
    """
    Attempt to read a service banner from an already-connected socket.

    Many services (SMTP, FTP, SSH) announce themselves as soon as a client
    connects, so we just try a recv() first. If nothing arrives, we send a
    generic CRLF probe to nudge line-based protocols (like HTTP-adjacent
    text services) into responding, then try to read again.
    """
    sock.settimeout(0.8)
    banner = b""

    # Step 1: some services (SSH, FTP, SMTP) announce themselves the
    # instant a client connects, with no input required. Try that first.
    try:
        banner = sock.recv(1024)
    except (socket.timeout, ConnectionResetError, OSError):
        banner = b""

    # Step 2: if nothing came back, the service is probably waiting for
    # input before it replies (common for line-based text protocols), so
    # send a harmless generic probe and give it one more chance. This is
    # a *separate* try/except from step 1 — a timeout on the first recv()
    # must not prevent us from attempting the probe-and-retry.
    if not banner:
        try:
            sock.sendall(b"\r\n")
            banner = sock.recv(1024)
        except (socket.timeout, ConnectionResetError, OSError):
            banner = b""

    # Banners are not guaranteed to be valid UTF-8 (binary protocols),
    # so decode defensively and strip control characters/whitespace.
    return banner.decode("utf-8", errors="replace").strip()


def scan_port(target: str, port: int, timeout: float, results: list, lock: threading.Lock):
    """
    Attempt a TCP connect to (target, port). If it succeeds, try to grab a
    banner and append (port, "open", banner) to the shared results list.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)  # short timeout so a filtered/closed port
                              # doesn't stall the whole scan for other threads
    try:
        result = sock.connect_ex((target, port))  # connect_ex returns an
        # errno instead of raising, so we can branch on it without a
        # try/except purely for control flow on the common "closed" case
        if result == 0:
            banner = grab_banner(sock)
            # The lock prevents two threads from interleaving list.append()
            # calls, which could otherwise corrupt the shared `results` list
            # or silently drop an entry under CPython's GIL scheduling.
            with lock:
                results.append((port, "open", banner))
    except socket.timeout:
        pass  # port did not respond within the timeout window; treat as filtered/closed
    except (ConnectionRefusedError, OSError):
        pass  # port actively refused the connection or host unreachable; treat as closed
    finally:
        sock.close()


def run_scan(target: str, start_port: int, end_port: int, timeout: float, max_workers: int):
    results = []
    lock = threading.Lock()
    threads = []

    # Cap the number of concurrently live threads with a semaphore so a
    # huge port range (e.g. 1-65535) doesn't spawn tens of thousands of
    # OS threads at once.
    semaphore = threading.Semaphore(max_workers)

    def worker(port):
        with semaphore:
            scan_port(target, port, timeout, results, lock)

    for port in range(start_port, end_port + 1):
        t = threading.Thread(target=worker, args=(port,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return sorted(results, key=lambda r: r[0])


def print_results(target: str, results: list):
    print(f"\nScan results for {target}")
    print(f"{'Port':<8}{'State':<8}{'Banner'}")
    print("-" * 60)
    if not results:
        print("No open ports found.")
        return
    for port, state, banner in results:
        display_banner = banner if banner else "(no banner)"
        print(f"{port:<8}{state:<8}{display_banner}")


def main():
    parser = argparse.ArgumentParser(description="Multithreaded TCP port scanner with banner grabbing.")
    parser.add_argument("target", help="Target IP address, e.g. 192.168.1.10")
    parser.add_argument("port_range", type=parse_port_range, help="Port range, e.g. 1-1024")
    parser.add_argument("--timeout", type=float, default=1.0, help="Per-connection timeout in seconds (default: 1.0)")
    parser.add_argument("--workers", type=int, default=200, help="Max concurrent scanning threads (default: 200)")
    args = parser.parse_args()

    # Basic validation that the target resolves, so we fail fast with a
    # clear message instead of every single thread raising individually.
    try:
        socket.gethostbyname(args.target)
    except socket.gaierror:
        print(f"Error: could not resolve target '{args.target}'", file=sys.stderr)
        sys.exit(1)

    start_port, end_port = args.port_range
    print(f"Scanning {args.target} ports {start_port}-{end_port} "
          f"(timeout={args.timeout}s, workers={args.workers})...")

    results = run_scan(args.target, start_port, end_port, args.timeout, args.workers)
    print_results(args.target, results)


if __name__ == "__main__":
    main()

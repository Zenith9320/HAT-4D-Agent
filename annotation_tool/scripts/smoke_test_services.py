#!/usr/bin/env python3
"""Check configured annotation backend health endpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Mapping, Sequence


SERVICE_ARGUMENTS = {
    "roi_backend": "roi_url",
    "sam3d_backend": "sam3d_url",
    "vlm_backend": "vlm_url",
    "mvtap_backend": "mvtap_url",
}


def _health_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/health") else f"{url}/health"


def check_services(
    service_urls: Mapping[str, str | None],
    timeout: float = 3.0,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for expected_service, base_url in service_urls.items():
        if not base_url:
            report[expected_service] = {
                "status": "skipped",
                "reason": "service URL was not configured",
            }
            continue

        url = _health_url(str(base_url))
        started = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("health response must be a JSON object")
            if payload.get("status") != "ok":
                raise ValueError(f"health status is {payload.get('status')!r}, expected 'ok'")
            if payload.get("service") != expected_service:
                raise ValueError(
                    f"health service is {payload.get('service')!r}, expected {expected_service!r}"
                )
            report[expected_service] = {
                "status": "ready",
                "url": url,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report[expected_service] = {
                "status": "failed",
                "url": url,
                "error": str(exc),
            }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-url")
    parser.add_argument("--sam3d-url")
    parser.add_argument("--vlm-url")
    parser.add_argument("--mvtap-url")
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    service_urls = {
        service: getattr(args, argument)
        for service, argument in SERVICE_ARGUMENTS.items()
    }
    report = check_services(service_urls, timeout=args.timeout)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if any(item["status"] == "failed" for item in report.values()) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"service smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

"""Post-deployment smoke test.

Usage:
    python smoke_test.py https://iris-api.onrender.com
    python smoke_test.py https://iris-api.onrender.com --attempts 12 --delay 15
"""
import argparse
import json
import sys
import time
import urllib.request


def call(url, data=None, timeout=60):
    if data is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def wait_for_health(base, attempts, delay):
    """Retry /health: a sleeping free service can need a minute to wake up."""
    for i in range(1, attempts + 1):
        try:
            status, body = call(base + "/health")
            if status == 200 and body == {"status": "ok"}:
                print(f"health OK (attempt {i})")
                return
        except Exception as exc:  # noqa: BLE001 - any failure just means "not ready yet"
            print(f"attempt {i}/{attempts}: not ready ({exc.__class__.__name__})")
        time.sleep(delay)
    sys.exit("FAILED: service never became healthy")


def main(base, attempts=10, delay=15):
    base = base.rstrip("/")
    wait_for_health(base, attempts, delay)

    status, body = call(base + "/model-info")
    print("model-info:", body)

    status, body = call(base + "/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
    assert status == 200 and body["label"] == "setosa", f"predict failed: {status} {body}"
    print("predict OK:", body)

    status, body = call(base + "/predict", {"features": [6.7, 3.0, 5.2, 2.3]})
    assert body["label"] == "virginica", f"predict failed: {body}"
    print("predict OK:", body)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--delay", type=float, default=15)
    args = parser.parse_args()
    main(args.url, args.attempts, args.delay)

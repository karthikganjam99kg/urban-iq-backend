#!/usr/bin/env python3
"""
Stamp the five demo buses with the current time so Live Fleet reads "live".

Fleet telemetry is only considered live for 300 seconds, so run this within
five minutes of a demo. These rows are presentation fixtures, not real GPS:
the buses table has no source column, so they cannot be told apart from device
reports afterwards. Prefer public/fleet-device.html when a real phone is
available.

Usage:
    set -a; . ./.env.supabase; set +a
    python3 scripts/seed_presentation_fleet.py
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Vehicle, route, and a plausible position along that corridor.
FIXTURES = [
    ("HYD-BUS-001", "R001", 17.3850, 78.4867, 31, 270),
    ("HYD-BUS-002", "R002", 17.3958, 78.4674, 26, 92),
    ("HYD-BUS-003", "R003", 17.3692, 78.5315, 34, 315),
    ("HYD-BUS-004", "R004", 17.4399, 78.4983, 22, 145),
    ("HYD-BUS-005", "R005", 17.4948, 78.3996, 29, 110),
]


def main():
    base = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()
    if not base or not key:
        sys.exit("SUPABASE_URL and SUPABASE_SECRET_KEY must be set.")

    now = datetime.now(timezone.utc)
    rows = [
        {
            "bus_id": bus_id,
            "route_id": route_id,
            "latitude": latitude,
            "longitude": longitude,
            "speed": speed,
            "heading": heading,
            "status": "ACTIVE",
            "recorded_at": now.isoformat(),
        }
        for bus_id, route_id, latitude, longitude, speed, heading in FIXTURES
    ]

    request = urllib.request.Request(
        # bus_id is unique, so re-running this updates the existing rows.
        f"{base}/rest/v1/buses?on_conflict=bus_id",
        method="POST",
        data=json.dumps(rows).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )
    try:
        urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        sys.exit(f"Supabase rejected the insert: {error.code} {error.read().decode()}")

    print(f"{len(rows)} buses marked ACTIVE at {now.isoformat()}")
    print("Live for 300 seconds. Refresh the dashboard now.")


if __name__ == "__main__":
    main()

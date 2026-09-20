#!/usr/bin/env python3
"""Seed every presentation-only UrbanIQ signal from one command.

This utility intentionally separates presentation fixtures from real inputs:

* Verified reference catalog: fitness routes (idempotent upsert).
* Presentation fixtures: fresh fleet positions, route conditions, and
  time-spread YOLO demand observations.
* Never seeded: TomTom traffic, detections, incidents, alerts, or model health.
  Those continue to come from their real APIs and inference paths.

The generated operational rows use a distinct ``UrbanIQ_presentation_seed``
source where the table supports it. Re-running replaces those generated rows
instead of accumulating duplicates.

Usage:
    set -a; . ./.env.supabase; set +a
    python3 scripts/seed_presentation_data.py
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


PRESENTATION_SOURCE = "UrbanIQ_presentation_seed"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same three distances shown by the original hardcoded UI, followed by
# publicly documented Hyderabad park, track, and cycling options.
FITNESS_ROUTES = [
    ("HYD-FIT-W01", "Walking Route · Tank Bund promenade", "walking", 2.4, 30, 17.4239, 78.4738),
    ("HYD-FIT-J01", "Jogging Route · Durgam Cheruvu lakeside", "jogging", 3.2, 20, 17.42886, 78.387794),
    ("HYD-FIT-C01", "Cycling Route · Necklace Road", "cycling", 4.1, 16, 17.4235, 78.4738),
    ("HYD-FIT-W02", "KBR National Park visitor trails", "walking", 5.0, 60, 17.448584, 78.379348),
    ("HYD-FIT-J02", "KBR Park peripheral jogging track", "jogging", 5.0, 30, 17.448584, 78.379348),
    ("HYD-FIT-C02", "Necklace Road exclusive cycling track", "cycling", 6.0, 24, 17.4235, 78.4738),
    ("HYD-FIT-J03", "G. M. C. Balayogi athletic track", "jogging", 0.4, 2, 17.4467833, 78.3446972),
    ("HYD-FIT-W03", "Durgam Cheruvu lake loop", "walking", 3.7, 56, 17.42886, 78.387794),
]

# Vehicle, route, plausible corridor position, speed, and heading.
FLEET_FIXTURES = [
    ("HYD-BUS-001", "R001", 17.3850, 78.4867, 31, 270),
    ("HYD-BUS-002", "R002", 17.3958, 78.4674, 26, 92),
    ("HYD-BUS-003", "R003", 17.3692, 78.5315, 34, 315),
    ("HYD-BUS-004", "R004", 17.4399, 78.4983, 22, 145),
    ("HYD-BUS-005", "R005", 17.4948, 78.3996, 29, 110),
]

# Two fresh signals per route make route comparison eligible without inventing
# a waterlogging observation.
ROUTE_CONDITIONS = [
    ("R001", 18, 1),
    ("R002", 34, 2),
    ("R003", 52, 4),
    ("R004", 67, 3),
    ("R005", 24, 1),
]

# Six minute buckets spanning 35 minutes satisfy the documented demand gate.
DEMAND_OFFSETS_MINUTES = (35, 28, 21, 14, 7, 0)
DEMAND_PEOPLE = {
    "HYD-BUS-001": (12, 15, 18, 22, 26, 30),
    "HYD-BUS-002": (14, 16, 17, 19, 21, 23),
    "HYD-BUS-003": (8, 11, 15, 19, 24, 29),
    "HYD-BUS-004": (18, 20, 21, 22, 23, 24),
    "HYD-BUS-005": (10, 13, 16, 18, 21, 24),
}


class Supabase:
    def __init__(self):
        load_local_environment()
        self.base = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
        self.key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()
        if not self.base or not self.key:
            sys.exit(
                "SUPABASE_URL and SUPABASE_SECRET_KEY must be set "
                "in .env.supabase, .env, or the shell."
            )

    def request(self, method, table, *, params=None, rows=None, prefer="return=minimal"):
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base}/rest/v1/{table}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            method=method,
            data=json.dumps(rows).encode() if rows is not None else None,
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": prefer,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode()[:800]
            raise RuntimeError(
                f"Supabase {table} rejected {method}: {error.code} {detail}"
            ) from error

    def upsert(self, table, rows, conflict):
        self.request(
            "POST",
            table,
            params={"on_conflict": conflict},
            rows=rows,
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def delete_source(self, table):
        self.request(
            "DELETE",
            table,
            params={"source": f"eq.{PRESENTATION_SOURCE}"},
        )


def load_local_environment():
    """Load simple KEY=VALUE files without requiring python-dotenv."""
    for path in (PROJECT_ROOT / ".env.supabase", PROJECT_ROOT / ".env"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            os.environ.setdefault(key.strip(), value)


def fitness_rows():
    return [
        {
            "route_code": code,
            "name": name,
            "activity_type": activity,
            "distance_km": distance,
            "duration_minutes": duration,
            "latitude": latitude,
            "longitude": longitude,
            "verified": True,
            "active": True,
        }
        for code, name, activity, distance, duration, latitude, longitude in FITNESS_ROUTES
    ]


def fleet_rows(now):
    return [
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
        for bus_id, route_id, latitude, longitude, speed, heading in FLEET_FIXTURES
    ]


def condition_rows(now):
    return [
        {
            "route_id": route_id,
            "congestion_score": congestion,
            "pothole_count": potholes,
            "waterlogging_count": None,
            "source": PRESENTATION_SOURCE,
            "observed_at": now.isoformat(),
        }
        for route_id, congestion, potholes in ROUTE_CONDITIONS
    ]


def demand_rows(now):
    routes = {bus_id: route_id for bus_id, route_id, *_rest in FLEET_FIXTURES}
    # (bus_id, frame) is unique. Timestamp-based values avoid colliding with
    # real camera frame numbers, while delete_source() removes prior fixtures.
    frame_base = int(now.timestamp())
    rows = []
    for bus_id, counts in DEMAND_PEOPLE.items():
        for sequence, (offset, people) in enumerate(
            zip(DEMAND_OFFSETS_MINUTES, counts)
        ):
            vehicles = max(3, round(people * 0.6))
            rows.append(
                {
                    "bus_id": bus_id,
                    "route_id": routes[bus_id],
                    "frame": frame_base + sequence,
                    "person_count": people,
                    "vehicle_count": vehicles,
                    "bus_count": 1,
                    "congestion_level": (
                        "HIGH"
                        if vehicles >= 15
                        else "MODERATE"
                        if vehicles >= 8
                        else "LOW"
                    ),
                    "source": PRESENTATION_SOURCE,
                    "recorded_at": (now - timedelta(minutes=offset)).isoformat(),
                }
            )
    return rows


def main():
    db = Supabase()
    now = datetime.now(timezone.utc)

    db.upsert("fitness_routes", fitness_rows(), "route_code")
    print(f"fitness_routes: {len(FITNESS_ROUTES)} verified catalog rows")

    db.upsert("buses", fleet_rows(now), "bus_id")
    print(f"buses: {len(FLEET_FIXTURES)} fresh presentation positions")

    db.delete_source("route_conditions")
    conditions = condition_rows(now)
    db.request("POST", "route_conditions", rows=conditions)
    print(f"route_conditions: {len(conditions)} fresh two-signal observations")

    db.delete_source("vehicle_density")
    demand = demand_rows(now)
    db.request("POST", "vehicle_density", rows=demand)
    print(
        "vehicle_density: "
        f"{len(demand)} observations across {len(DEMAND_PEOPLE)} buses"
    )

    print(f"Seeded at {now.isoformat()}")
    print("Fleet remains live for 300 seconds; run once immediately before a demo.")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import os
import requests
import uuid
import time
import tempfile
import threading
from collections import defaultdict
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Local development reads secrets from .env; hosted runs use platform secrets.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.supabase")
except ImportError:
    pass

# Serverless / Space filesystems: keep ML caches on writable tmp.
if os.getenv("VERCEL") or os.getenv("SPACE_ID"):
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("TORCH_HOME", "/tmp/torch")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from flask import Flask, g, jsonify, request
from flask_cors import CORS

# Vehicle detection model
# AI models are loaded only when required
vehicle_model = None
plate_model = None
plate_reader = None
model_readiness_lock = threading.Lock()
model_readiness_cache = {"checked_at": 0.0, "ready": {}, "errors": {}}
def get_vehicle_model():
    global vehicle_model

    if vehicle_model is None:
        from ultralytics import YOLO
        vehicle_model = YOLO(str(PROJECT_ROOT / "yolo11n.pt"))

    return vehicle_model


def get_plate_model():
    global plate_model

    if plate_model is None:
        from ultralytics import YOLO
        plate_model = YOLO(str(PROJECT_ROOT / "weights" / "license_plate.pt"))

    return plate_model

def get_plate_reader():
    global plate_reader

    if plate_reader is None:
        import easyocr
        plate_reader = easyocr.Reader(["en"], gpu=False)

    return plate_reader


def model_readiness():
    global model_readiness_cache
    from src.garbage_ai.garbage_detector import get_garbage_model
    from src.pothole_ai.pothole_detector import get_pothole_model

    with model_readiness_lock:
        cache_age = time.monotonic() - model_readiness_cache["checked_at"]
        if model_readiness_cache["ready"] and cache_age <= 30:
            return (
                dict(model_readiness_cache["ready"]),
                dict(model_readiness_cache["errors"]),
            )

        factories = {
            "vehicle": get_vehicle_model,
            "garbage": get_garbage_model,
            "pothole": get_pothole_model,
            "plate": get_plate_model,
        }
        ready = {}
        errors = {}
        for name, factory in factories.items():
            try:
                ready[name] = factory() is not None
            except Exception as exc:
                ready[name] = False
                errors[name] = f"{type(exc).__name__}: {exc}"
        model_readiness_cache = {
            "checked_at": time.monotonic(),
            "ready": ready,
            "errors": errors,
        }
        return dict(ready), dict(errors)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
# Rash driving tracking
previous_vehicle_positions = {}
rash_driving_alerts = []


class UploadValidationError(ValueError):
    pass


def save_upload(upload, prefix):
    suffix = (Path(upload.filename or "upload.jpg").suffix or ".jpg").lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise UploadValidationError("Only JPG, PNG, and WEBP images are supported.")
    if upload.mimetype and not upload.mimetype.startswith("image/"):
        raise UploadValidationError("The uploaded file must be an image.")
    handle, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
    os.close(handle)
    upload.save(path)
    upload_path = Path(path)
    g.upload_paths = [*getattr(g, "upload_paths", []), upload_path]
    return upload_path


@app.errorhandler(UploadValidationError)
def invalid_upload(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "Image exceeds the 12 MB upload limit."}), 413


@app.teardown_request
def cleanup_uploads(_error):
    for upload_path in getattr(g, "upload_paths", []):
        upload_path.unlink(missing_ok=True)

def supabase_base():
    base = os.getenv("SUPABASE_URL")
    return base.strip().rstrip("/") if base else None


def supabase_request(method, path, params=None, json_body=None, prefer="return=representation"):
    headers = supabase_headers()
    base = supabase_base()
    if not headers or not base:
        raise RuntimeError("Supabase is not configured")

    headers = {**headers, "Prefer": prefer}
    response = requests.request(
        method,
        f"{base}/rest/v1/{path.lstrip('/')}",
        headers=headers,
        params=params,
        json=json_body,
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(f"Supabase {path} {response.status_code}: {response.text[:300]}")
    if not response.content:
        return []
    return response.json()


def optional_supabase_rows(path, params=None):
    """Read a migration-managed table; return None until its SQL is installed."""
    try:
        return supabase_request("GET", path, params=params)
    except RuntimeError as exc:
        message = str(exc)
        if (
            "PGRST205" in message
            or "PGRST204" in message
            or "Could not find the table" in message
        ):
            return None
        raise


def insert_with_schema_fallback(table, body, optional_keys, prefer="return=representation"):
    """Insert new-schema fields when available, then retry legacy columns."""
    try:
        return supabase_request(
            "POST",
            table,
            json_body=body,
            prefer=prefer,
        )
    except RuntimeError as exc:
        if "PGRST204" not in str(exc):
            raise
        legacy_body = {
            key: value for key, value in body.items() if key not in optional_keys
        }
        return supabase_request(
            "POST",
            table,
            json_body=legacy_body,
            prefer=prefer,
        )


def known_vehicle_routes():
    """Map configured bus_id to route_id, or None when the catalog is absent."""
    rows = optional_supabase_rows(
        "fleet_vehicles",
        params={"select": "bus_id,route_id", "active": "eq.true"},
    )
    if rows is None:
        return None
    return {row["bus_id"]: row.get("route_id") for row in rows if row.get("bus_id")}


def coerce_number(value, field, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")
    if number != number or not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return number


def build_telemetry_row(reading, catalog, observed_at):
    """Validate one device reading into a buses row, or raise ValueError."""
    if not isinstance(reading, dict):
        raise ValueError("each reading must be an object")

    bus_id = str(reading.get("bus_id") or "").strip()
    if not bus_id:
        raise ValueError("bus_id is required")
    if catalog is not None and bus_id not in catalog:
        raise ValueError(f"{bus_id} is not a configured vehicle")

    row = {
        "bus_id": bus_id,
        "latitude": coerce_number(reading.get("latitude"), "latitude", 16.5, 18.5),
        "longitude": coerce_number(
            reading.get("longitude"), "longitude", 77.5, 79.5
        ),
        "recorded_at": observed_at,
    }

    route_id = reading.get("route_id") or (catalog or {}).get(bus_id)
    if route_id:
        row["route_id"] = route_id
    if reading.get("speed") is not None:
        row["speed"] = coerce_number(reading.get("speed"), "speed", 0, 200)
    if reading.get("heading") is not None:
        row["heading"] = coerce_number(reading.get("heading"), "heading", 0, 360)

    # Operational status is only stored when the device reports it, so the UI
    # keeps showing UNKNOWN instead of assuming a bus is in service.
    status = reading.get("status")
    if status is not None:
        status = str(status).strip().upper()
        if not status or len(status) > 32:
            raise ValueError("status must be 1-32 characters")
        row["status"] = status

    return row


def map_incident(row):
    return {
        "id": row.get("incident_id") or str(row.get("id")),
        "type": row.get("incident_type"),
        "severity": row.get("severity"),
        "location": row.get("location"),
        "description": row.get("description"),
        "timestamp": row.get("timestamp") or row.get("created_at"),
        "status": row.get("status") or "New",
        "source": row.get("source"),
    }


def map_alert(row):
    return {
        "id": row.get("alert_id") or str(row.get("id")),
        "incident_id": row.get("incident_id"),
        "type": row.get("alert_type"),
        "severity": row.get("severity"),
        "location": row.get("location"),
        "message": row.get("message"),
        "title": row.get("title") or f"{row.get('alert_type') or 'Civic'} alert",
        "recommendation": row.get("recommendation"),
        "bus_id": row.get("bus_id"),
        "route_id": row.get("route_id"),
        "timestamp": row.get("detected_at") or row.get("created_at"),
        "status": row.get("status") or "New",
    }


def fetch_incidents():
    rows = supabase_request(
        "GET",
        "incidents",
        params={"select": "*", "order": "timestamp.desc.nullslast,created_at.desc"},
    )
    return [map_incident(row) for row in rows]


def fetch_alerts():
    rows = supabase_request(
        "GET",
        "alerts",
        params={"select": "*", "order": "detected_at.desc.nullslast,created_at.desc"},
    )
    return [map_alert(row) for row in rows]


def fetch_fleet():
    telemetry_rows = supabase_request(
        "GET",
        "buses",
        params={"select": "*", "order": "recorded_at.desc", "limit": "500"},
    )
    vehicle_rows = optional_supabase_rows(
        "fleet_vehicles",
        params={"select": "*", "active": "eq.true", "order": "bus_id.asc"},
    )
    route_rows = optional_supabase_rows(
        "fleet_routes",
        params={"select": "*", "active": "eq.true", "order": "route_id.asc"},
    )

    latest_by_bus = {}
    for row in telemetry_rows:
        bus_id = row.get("bus_id") or str(row.get("id"))
        if bus_id not in latest_by_bus:
            latest_by_bus[bus_id] = row

    routes = {
        row.get("route_id"): row
        for row in (route_rows or [])
        if row.get("route_id")
    }
    vehicles = {
        row.get("bus_id"): row
        for row in (vehicle_rows or [])
        if row.get("bus_id")
    }
    all_bus_ids = sorted(set(latest_by_bus) | set(vehicles))

    now = datetime.now(timezone.utc)
    buses = []
    latest = None
    for bus_id in all_bus_ids:
        row = latest_by_bus.get(bus_id, {})
        vehicle = vehicles.get(bus_id, {})
        recorded_at = parse_timestamp(row.get("recorded_at"))
        if recorded_at:
            recorded_at = recorded_at.astimezone(timezone.utc)
            latest = max(latest, recorded_at) if latest else recorded_at
            age_seconds = max(0, (now - recorded_at).total_seconds())
        else:
            age_seconds = None
        telemetry_status = (
            "live"
            if age_seconds is not None and age_seconds <= 300
            else ("stale" if recorded_at else "missing")
        )
        route_id = (
            row.get("route_id")
            or vehicle.get("route_id")
            or "Unassigned"
        )
        route = routes.get(route_id, {})
        reported_status = row.get("status") or "UNKNOWN"
        buses.append(
            {
                "id": bus_id,
                "display_name": vehicle.get("display_name") or bus_id,
                "route_id": route_id,
                "route_name": route.get("name") or f"Route {route_id}",
                "origin": route.get("origin"),
                "destination": route.get("destination"),
                "capacity": vehicle.get("capacity"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "speed": row.get("speed"),
                "heading": row.get("heading"),
                "operational_status": (
                    reported_status if telemetry_status == "live" else "UNKNOWN"
                ),
                "last_reported_operational_status": reported_status,
                "telemetry_status": telemetry_status,
                "recorded_at": (
                    recorded_at.isoformat() if recorded_at else row.get("recorded_at")
                ),
                "age_seconds": round(age_seconds) if age_seconds is not None else None,
            }
        )

    status_order = {"live": 0, "stale": 1, "missing": 2}
    buses.sort(key=lambda bus: (status_order[bus["telemetry_status"]], bus["id"]))
    live_count = sum(bus["telemetry_status"] == "live" for bus in buses)
    stale_count = sum(bus["telemetry_status"] == "stale" for bus in buses)
    missing_count = sum(bus["telemetry_status"] == "missing" for bus in buses)
    return {
        "status": (
            "live"
            if live_count
            else ("stale" if stale_count else ("missing" if buses else "empty"))
        ),
        "schema_ready": vehicle_rows is not None and route_rows is not None,
        "buses": buses,
        "summary": {
            "total": len(buses),
            "live": live_count,
            "stale": stale_count,
            "missing": missing_count,
            "latest_observation": latest.isoformat() if latest else None,
        },
        "generated_at": now.isoformat(),
    }


def create_incident(
    incident_type,
    severity,
    location,
    description,
    bus_id=None,
    route_id=None,
    recommendation=None,
):
    incident_id = str(uuid.uuid4())
    stamp = datetime.now(timezone.utc).isoformat()
    incident_body = {
        "incident_id": incident_id,
        "incident_type": incident_type,
        "severity": severity,
        "location": location,
        "description": description,
        "status": "New",
        "timestamp": stamp,
        "source": "UrbanIQ-api",
    }
    alert_body = {
        "alert_id": str(uuid.uuid4()),
        "incident_id": incident_id,
        "alert_type": incident_type,
        "severity": severity,
        "location": location,
        "message": description,
        "status": "New",
        "detected_at": stamp,
    }
    if bus_id:
        incident_body["bus_id"] = bus_id
        alert_body["bus_id"] = bus_id
    if route_id:
        incident_body["route_id"] = route_id
        alert_body["route_id"] = route_id
    if recommendation:
        alert_body["recommendation"] = recommendation
    rows = insert_with_schema_fallback(
        "incidents",
        incident_body,
        {"bus_id", "route_id"},
    )
    insert_with_schema_fallback(
        "alerts",
        alert_body,
        {"bus_id", "route_id", "recommendation"},
        prefer="return=minimal",
    )
    return map_incident(rows[0] if rows else {
        "incident_id": incident_id,
        "incident_type": incident_type,
        "severity": severity,
        "location": location,
        "description": description,
        "status": "New",
        "timestamp": stamp,
        "source": "UrbanIQ-api",
    })


def build_cors_origins(configured=""):
    origins = [
        "https://hyderabad-urban-intelligence.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    origins.extend(
        origin
        for origin in (value.strip() for value in configured.split(","))
        if origin and origin != "*" and origin not in origins
    )
    return origins


cors_origins = build_cors_origins(os.getenv("CORS_ORIGINS", ""))
CORS(app, resources={r"/api/*": {"origins": cors_origins}})


@app.before_request
def reject_unapproved_browser_origins():
    origin = request.headers.get("Origin")
    if (
        origin
        and request.path.startswith("/api/")
        and origin not in cors_origins
    ):
        return jsonify({"error": "Origin is not allowed."}), 403


HYDERABAD_POINT = (17.3850, 78.4867)
_last_traffic_persist = 0
_traffic_cache = {"payload": None, "fetched_at": 0.0}


def supabase_headers():
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not key:
        return None
    key = key.strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def persist_traffic_snapshot(payload, route_id=None, bus_id=None):
    """Write into the existing traffic_realtime table (service role)."""
    global _last_traffic_persist
    headers = supabase_headers()
    base = supabase_base()
    if not headers or not base:
        return

    now = time.time()
    if now - _last_traffic_persist < 60:
        return

    current = payload["current_speed"]
    free_flow = payload["free_flow_speed"]
    speed_ratio = round(1 - (current / free_flow), 4) if free_flow else 0
    row = {
        "road_id": "hyd-tomtom-live",
        "road_name": "Hyderabad TomTom Segment",
        "latitude": HYDERABAD_POINT[0],
        "longitude": HYDERABAD_POINT[1],
        "current_speed": current,
        "free_flow_speed": free_flow,
        "current_travel_time": payload.get("current_travel_time"),
        "free_flow_travel_time": payload.get("free_flow_travel_time"),
        "speed_ratio": speed_ratio,
        "delay_ratio": speed_ratio,
        "congestion_score": payload["congestion_score"],
        "congestion_level": payload["congestion_level"],
        "confidence": payload.get("confidence"),
        "road_closure": payload.get("road_closure", False),
        "source": "TomTom",
    }
    if route_id:
        row["route_id"] = route_id
    if bus_id:
        row["bus_id"] = bus_id

    try:
        insert_with_schema_fallback(
            "traffic_realtime",
            row,
            {"route_id", "bus_id"},
            prefer="return=minimal",
        )
        _last_traffic_persist = now
    except Exception as exc:
        print("SUPABASE TRAFFIC INSERT ERROR:", repr(exc))


def persist_vehicle_density(bus_id, route_id, person_count, vehicle_count):
    if not bus_id:
        return
    body = {
        "bus_id": bus_id,
        # frame is NOT NULL; a still upload is a single frame.
        "frame": 0,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "bus_count": 0,
        "congestion_level": (
            "HIGH" if vehicle_count >= 15 else "MODERATE" if vehicle_count >= 8 else "LOW"
        ),
        "source": "UrbanIQ_vehicle_detect",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if route_id:
        body["route_id"] = route_id
    try:
        insert_with_schema_fallback(
            "vehicle_density",
            body,
            {"route_id"},
            prefer="return=minimal",
        )
    except Exception as exc:
        print("VEHICLE DENSITY INSERT ERROR:", repr(exc))


def persist_route_condition(
    route_id,
    source,
    congestion_score=None,
    pothole_count=None,
    waterlogging_count=None,
):
    if not route_id:
        return
    body = {
        "route_id": route_id,
        "source": source,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "congestion_score": congestion_score,
        "pothole_count": pothole_count,
        "waterlogging_count": waterlogging_count,
    }
    try:
        supabase_request(
            "POST",
            "route_conditions",
            json_body=body,
            prefer="return=minimal",
        )
    except RuntimeError as exc:
        if "PGRST205" not in str(exc):
            print("ROUTE CONDITION INSERT ERROR:", repr(exc))


def fetch_traffic_history(limit=30):
    headers = supabase_headers()
    base = supabase_base()
    if not headers or not base:
        return []

    try:
        response = requests.get(
            f"{base.rstrip('/')}/rest/v1/traffic_realtime",
            headers={**headers, "Prefer": "count=exact"},
            params={
                "select": "congestion_score,congestion_level,current_speed,recorded_at",
                "order": "recorded_at.desc",
                "limit": str(limit),
            },
            timeout=10,
        )
        if not response.ok:
            print("SUPABASE TRAFFIC HISTORY:", response.status_code)
            return []
        rows = response.json()
        rows.reverse()
        return [
            {
                "congestion_score": row.get("congestion_score"),
                "congestion_level": row.get("congestion_level"),
                "current_speed": row.get("current_speed"),
                "captured_at": row.get("recorded_at"),
            }
            for row in rows
        ]
    except Exception as exc:
        print("SUPABASE TRAFFIC HISTORY ERROR:", repr(exc))
        return []


def parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def linear_demand_prediction(points, horizon_minutes=60):
    """Fit a small rolling linear model to minute-level YOLO person counts."""
    first_time = points[0][0]
    x_values = [(stamp - first_time).total_seconds() / 60 for stamp, _ in points]
    y_values = [count for _, count in points]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values)
        )
        / denominator
        if denominator
        else 0
    )
    intercept = y_mean - slope * x_mean
    target_x = x_values[-1] + horizon_minutes
    prediction = max(0, min(100, intercept + slope * target_x))

    total_variance = sum((value - y_mean) ** 2 for value in y_values)
    residual_variance = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x_values, y_values)
    )
    fit_score = (
        max(0, 1 - (residual_variance / total_variance))
        if total_variance
        else 0.5
    )
    return prediction, fit_score


def demand_level(score):
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MODERATE"
    return "LOW"


def fetch_demand_forecast():
    """
    Build a demand proxy from YOLO person counts.

    A forecast is withheld until there are at least six minute buckets spanning
    30 minutes and the latest observation is recent. This prevents a demo-sized
    frame burst from being presented as a trained passenger forecast.
    """
    rows = supabase_request(
        "GET",
        "vehicle_density",
        params={
            "select": "bus_id,person_count,vehicle_count,recorded_at,source",
            "order": "recorded_at.desc",
            "limit": "1000",
        },
    )
    buses = supabase_request(
        "GET",
        "buses",
        params={"select": "bus_id,route_id,status", "limit": "500"},
    )
    configured_vehicles = optional_supabase_rows(
        "fleet_vehicles",
        params={"select": "bus_id,route_id", "active": "eq.true", "limit": "500"},
    )
    route_by_bus = {
        row.get("bus_id"): row.get("route_id") or "Unassigned"
        for row in (configured_vehicles if configured_vehicles is not None else buses)
        if row.get("bus_id")
    }

    minute_values = defaultdict(list)
    valid_stamps = []
    eligible_frames = 0
    for row in rows:
        stamp = parse_timestamp(row.get("recorded_at"))
        bus_id = row.get("bus_id")
        if (
            not stamp
            or not bus_id
            or bus_id == "OPERATOR-UPLOAD"
            or bus_id not in route_by_bus
        ):
            continue
        eligible_frames += 1
        stamp = stamp.astimezone(timezone.utc)
        minute = stamp.replace(second=0, microsecond=0)
        minute_values[(bus_id, minute)].append(float(row.get("person_count") or 0))
        valid_stamps.append(stamp)

    now = datetime.now(timezone.utc)
    grouped_by_bus = defaultdict(list)
    for (bus_id, minute), values in minute_values.items():
        grouped_by_bus[bus_id].append((minute, sum(values) / len(values)))

    route_forecasts = []
    for bus_id, points in grouped_by_bus.items():
        points.sort(key=lambda item: item[0])
        coverage_minutes = (
            (points[-1][0] - points[0][0]).total_seconds() / 60
            if len(points) > 1
            else 0
        )
        age_minutes = max(0, (now - points[-1][0]).total_seconds() / 60)
        ready = len(points) >= 6 and coverage_minutes >= 30 and age_minutes <= 30
        route = route_by_bus.get(bus_id, "Unassigned")
        if ready:
            predicted_people, fit_score = linear_demand_prediction(points)
            confidence = round(
                min(
                    95,
                    45
                    + min(20, len(points) * 2)
                    + min(15, coverage_minutes / 4)
                    + fit_score * 15,
                )
            )
            score = round(min(100, predicted_people))
            route_forecasts.append(
                {
                    "bus_id": bus_id,
                    "route_id": route,
                    "predicted_people": round(predicted_people),
                    "demand_score": score,
                    "demand_level": demand_level(score),
                    "confidence": confidence,
                    "observations": len(points),
                    "coverage_minutes": round(coverage_minutes, 1),
                    "status": "live",
                }
            )
        else:
            route_forecasts.append(
                {
                    "bus_id": bus_id,
                    "route_id": route,
                    "status": "collecting",
                    "observations": len(points),
                    "coverage_minutes": round(coverage_minutes, 1),
                    "last_seen_minutes_ago": round(age_minutes, 1),
                }
            )

    live_routes = [
        forecast for forecast in route_forecasts if forecast["status"] == "live"
    ]
    latest_stamp = max(valid_stamps).isoformat() if valid_stamps else None
    quality = {
        "raw_frames": eligible_frames,
        "excluded_frames": len(rows) - eligible_frames,
        "minute_buckets": len(minute_values),
        "buses_observed": len(grouped_by_bus),
        "latest_observation": latest_stamp,
        "minimum_required": "6 minute buckets over 30 minutes; latest within 30 minutes",
        "signal": "YOLO person_count demand proxy",
    }

    if not live_routes:
        return {
            "status": "collecting",
            "model": "rolling-linear-demand-v1",
            "message": (
                "Not enough recent temporal coverage for a trustworthy forecast."
            ),
            "data_quality": quality,
            "routes": route_forecasts,
            "generated_at": now.isoformat(),
        }

    total_weight = sum(max(1, row["observations"]) for row in live_routes)
    predicted_people = round(
        sum(row["predicted_people"] * row["observations"] for row in live_routes)
        / total_weight
    )
    confidence = round(
        sum(row["confidence"] * row["observations"] for row in live_routes)
        / total_weight
    )
    demand_score = round(
        sum(row["demand_score"] * row["observations"] for row in live_routes)
        / total_weight
    )
    return {
        "status": "live",
        "model": "rolling-linear-demand-v1",
        "predicted_people": predicted_people,
        "demand_score": demand_score,
        "demand_level": demand_level(demand_score),
        "confidence": confidence,
        "recommendation": (
            "Increase fleet capacity on high-demand routes."
            if demand_score >= 80
            else "Current fleet capacity is adequate; continue monitoring."
        ),
        "data_quality": quality,
        "routes": sorted(
            live_routes, key=lambda row: row["demand_score"], reverse=True
        ),
        "generated_at": now.isoformat(),
    }


def latest_traffic_snapshot():
    rows = supabase_request(
        "GET",
        "traffic_realtime",
        params={
            "select": (
                "congestion_score,congestion_level,current_speed,"
                "free_flow_speed,recorded_at"
            ),
            "order": "recorded_at.desc",
            "limit": "1",
        },
    )
    if not rows:
        return None
    row = rows[0]
    stamp = parse_timestamp(row.get("recorded_at"))
    age_seconds = (
        max(0, (datetime.now(timezone.utc) - stamp).total_seconds())
        if stamp
        else None
    )
    return {
        **row,
        "age_seconds": round(age_seconds) if age_seconds is not None else None,
        "status": (
            "live"
            if age_seconds is not None and age_seconds <= 600
            else "stale"
        ),
    }


def fetch_routes():
    route_rows = optional_supabase_rows(
        "fleet_routes",
        params={"select": "*", "active": "eq.true", "order": "route_id.asc"},
    )
    condition_rows = optional_supabase_rows(
        "route_conditions",
        params={
            "select": "*",
            "order": "observed_at.desc",
            "limit": "1000",
        },
    )
    schema_ready = route_rows is not None and condition_rows is not None

    if route_rows is None:
        fleet = fetch_fleet()
        route_rows = [
            {
                "route_id": route_id,
                "name": route_name,
                "origin": None,
                "destination": None,
            }
            for route_id, route_name in sorted(
                {
                    (bus["route_id"], bus["route_name"])
                    for bus in fleet["buses"]
                    if bus["route_id"] != "Unassigned"
                }
            )
        ]
    condition_rows = condition_rows or []

    latest_condition = {}
    for row in condition_rows:
        route_id = row.get("route_id")
        if not route_id:
            continue
        aggregate = latest_condition.setdefault(
            route_id,
            {
                "observed_at": None,
                "congestion_score": None,
                "congestion_observed_at": None,
                "pothole_count": None,
                "pothole_observed_at": None,
                "waterlogging_count": None,
                "waterlogging_observed_at": None,
            },
        )
        if aggregate["observed_at"] is None:
            aggregate["observed_at"] = row.get("observed_at")
        for field in ("congestion_score", "pothole_count", "waterlogging_count"):
            if aggregate[field] is None and row.get(field) is not None:
                aggregate[field] = row.get(field)
                aggregate[f"{field.split('_')[0]}_observed_at"] = row.get(
                    "observed_at"
                )

    now = datetime.now(timezone.utc)
    routes = []
    for route in route_rows:
        route_id = route.get("route_id")
        condition = latest_condition.get(route_id)
        observed_at = parse_timestamp(condition.get("observed_at")) if condition else None
        signal_stamps = [
            parse_timestamp(condition.get(field))
            for field in (
                "congestion_observed_at",
                "pothole_observed_at",
                "waterlogging_observed_at",
            )
        ] if condition else []
        signal_ages = [
            max(0, (now - stamp.astimezone(timezone.utc)).total_seconds())
            if stamp
            else None
            for stamp in signal_stamps
        ]
        signal_names = ("congestion", "pothole", "waterlogging")
        fresh_signals = [
            name
            for name, value, age in zip(
                signal_names,
                (
                    condition.get("congestion_score") if condition else None,
                    condition.get("pothole_count") if condition else None,
                    condition.get("waterlogging_count") if condition else None,
                ),
                signal_ages,
            )
            if value is not None and age is not None and age <= 900
        ]
        condition_status = (
            "live"
            if len(fresh_signals) >= 2
            else "partial"
            if fresh_signals
            else ("stale" if condition else "missing")
        )
        routes.append(
            {
                "route_id": route_id,
                "name": route.get("name") or f"Route {route_id}",
                "origin": route.get("origin"),
                "destination": route.get("destination"),
                "condition_status": condition_status,
                "congestion_score": (
                    condition.get("congestion_score") if condition else None
                ),
                "pothole_count": (
                    condition.get("pothole_count") if condition else None
                ),
                "waterlogging_count": (
                    condition.get("waterlogging_count") if condition else None
                ),
                "fresh_signals": fresh_signals,
                "evidence_count": len(fresh_signals),
                "observed_at": (
                    observed_at.isoformat() if observed_at else None
                ),
            }
        )

    scored = []
    for route in routes:
        if route["condition_status"] != "live":
            continue
        components = []
        if "congestion" in route["fresh_signals"]:
            components.append(float(route["congestion_score"]))
        if "pothole" in route["fresh_signals"]:
            components.append(min(100, int(route["pothole_count"]) * 10))
        if "waterlogging" in route["fresh_signals"]:
            components.append(min(100, int(route["waterlogging_count"]) * 15))
        score = sum(components) / len(components)
        scored.append({**route, "score": round(score, 1)})
    scored.sort(key=lambda row: row["score"])

    return {
        "status": (
            "live"
            if len(scored) >= 2
            else ("collecting" if schema_ready else "schema_required")
        ),
        "schema_ready": schema_ready,
        "routes": routes,
        "recommended": scored[0] if len(scored) >= 2 else None,
        "alternatives": scored[1:] if len(scored) >= 2 else [],
        "message": (
            None
            if len(scored) >= 2
            else "At least two routes need two recent observed condition signals."
        ),
        "generated_at": now.isoformat(),
    }


def fetch_fitness():
    route_rows = optional_supabase_rows(
        "fitness_routes",
        params={
            "select": "*",
            "active": "eq.true",
            "verified": "eq.true",
            "order": "activity_type.asc",
        },
    )
    facility_rows = optional_supabase_rows(
        "sports_facilities",
        params={
            "select": "*",
            "active": "eq.true",
            "verified": "eq.true",
            "order": "name.asc",
        },
    )
    traffic = latest_traffic_snapshot()
    traffic_live = bool(traffic and traffic["status"] == "live")
    if traffic_live:
        score = float(traffic.get("congestion_score") or 0)
        safety = "SAFE" if score < 60 else "CAUTION"
    else:
        safety = "NO DATA"

    routes = [
        {
            "id": row.get("id"),
            "route_code": row.get("route_code"),
            "name": row.get("name"),
            "activity_type": row.get("activity_type"),
            "distance_km": row.get("distance_km"),
            "duration_minutes": row.get("duration_minutes"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "safety": safety,
            "traffic_status": traffic.get("status") if traffic else "empty",
            "congestion_score": (
                traffic.get("congestion_score") if traffic_live else None
            ),
        }
        for row in (route_rows or [])
    ]
    facilities = [
        {
            "id": row.get("id"),
            "facility_code": row.get("facility_code"),
            "name": row.get("name"),
            "facility_type": row.get("facility_type"),
            "activities": row.get("activities") or [],
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
        }
        for row in (facility_rows or [])
    ]
    schema_ready = route_rows is not None and facility_rows is not None
    return {
        "status": (
            "live"
            if schema_ready and routes
            else "catalog_only"
            if schema_ready and facilities
            else ("empty" if schema_ready else "schema_required")
        ),
        "schema_ready": schema_ready,
        "traffic": traffic,
        "routes": routes,
        "facilities": facilities,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_overview():
    fleet_data = fetch_fleet()
    traffic = latest_traffic_snapshot()
    demand = fetch_demand_forecast()
    alerts = fetch_alerts()
    models, _ = model_readiness()
    services = {
        "fleet": fleet_data["status"],
        "traffic": traffic["status"] if traffic else "empty",
        "vision": (
            "live"
            if all(models.get(name) for name in ("vehicle", "garbage", "pothole"))
            else "degraded"
        ),
        "demand": demand["status"],
    }
    critical_healthy = (
        services["fleet"] == "live"
        and services["traffic"] == "live"
        and services["vision"] == "live"
    )
    return {
        "status": "operational" if critical_healthy else "degraded",
        "services": services,
        "models": models,
        "fleet": fleet_data["summary"],
        "active_alerts": sum(
            str(alert.get("status", "")).lower()
            not in {"resolved", "closed", "dismissed"}
            for alert in alerts
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def traffic_simulation(vehicle_count):
    """
    Score a what-if vehicle count against the live traffic snapshot.

    The score itself only needs the measured road speed, so the simulation runs
    whenever that snapshot is fresh. Observed vehicle density is an optional
    side-by-side comparison and carries its own status rather than blocking the
    scenario, so a stale camera history never presents as a stale forecast.
    """
    traffic = latest_traffic_snapshot()
    if not traffic or traffic["status"] != "live":
        return {
            "status": "offline",
            "error": "A recent traffic snapshot is required.",
        }, 409

    density_rows = supabase_request(
        "GET",
        "vehicle_density",
        params={
            "select": "vehicle_count,recorded_at",
            "order": "recorded_at.desc",
            "limit": "100",
        },
    )
    now = datetime.now(timezone.utc)
    observations = []
    for row in density_rows:
        stamp = parse_timestamp(row.get("recorded_at"))
        if stamp:
            observations.append(
                (stamp.astimezone(timezone.utc), float(row.get("vehicle_count") or 0))
            )

    recent_counts = [
        count
        for stamp, count in observations
        if (now - stamp).total_seconds() <= 600
    ]
    if recent_counts:
        # Vehicles are whole objects, so the mean is reported as a count.
        baseline_count = round(sum(recent_counts) / len(recent_counts))
        baseline = {
            "status": "live",
            "vehicle_count": baseline_count,
            "sample_size": len(recent_counts),
        }
    elif observations:
        newest_stamp, newest_count = max(observations, key=lambda item: item[0])
        baseline_count = round(newest_count)
        baseline = {
            "status": "stale",
            "vehicle_count": baseline_count,
            "sample_size": 1,
            "age_minutes": round((now - newest_stamp).total_seconds() / 60),
        }
    else:
        baseline_count = None
        baseline = {"status": "missing"}

    predicted_score = calculate_congestion(
        vehicle_count,
        float(traffic.get("current_speed") or 0),
    )
    if predicted_score < 30:
        level = "Low"
    elif predicted_score < 60:
        level = "Moderate"
    elif predicted_score < 80:
        level = "High"
    else:
        level = "Severe"
    return {
        "status": "live",
        "vehicles": vehicle_count,
        "baseline_vehicle_count": baseline_count,
        "baseline": baseline,
        "current_score": traffic.get("congestion_score"),
        "score": predicted_score,
        "level": level,
        "generated_at": now.isoformat(),
    }, 200


def calculate_congestion(vehicle_count, average_speed):
    """
    Calculate traffic congestion score from 0 to 100.
    Higher score = more congestion.
    """

    vehicle_score = min((vehicle_count / 100) * 60, 60)

    speed_score = max((40 - average_speed) / 40 * 40, 0)

    score = vehicle_score + speed_score

    return round(min(score, 100), 2)

@app.route("/api/incidents")
def get_incidents():
    try:
        return jsonify(fetch_incidents())
    except Exception as exc:
        print("INCIDENTS ERROR:", repr(exc))
        return jsonify({"error": "Could not load incidents"}), 503


@app.route("/api/alerts")
def get_alerts():
    try:
        return jsonify(fetch_alerts())
    except Exception as exc:
        print("ALERTS ERROR:", repr(exc))
        return jsonify({"error": "Could not load alerts"}), 503


@app.route("/api/fleet")
def fleet():
    try:
        return jsonify(fetch_fleet())
    except Exception as exc:
        print("FLEET ERROR:", repr(exc))
        return jsonify({
            "status": "offline",
            "error": "Could not load fleet telemetry from Supabase",
        }), 503


@app.route("/api/fleet/telemetry", methods=["POST"])
def ingest_fleet_telemetry():
    expected_token = (os.getenv("FLEET_INGEST_TOKEN") or "").strip()
    if not expected_token:
        return jsonify({
            "error": "Telemetry ingestion is disabled until FLEET_INGEST_TOKEN is set",
        }), 503
    if request.headers.get("X-Ingest-Token", "") != expected_token:
        return jsonify({"error": "Invalid ingestion token"}), 401

    payload = request.get_json(silent=True)
    readings = payload if isinstance(payload, list) else [payload]
    if not readings or payload is None:
        return jsonify({"error": "Send a reading object or a list of readings"}), 400
    if len(readings) > 200:
        return jsonify({"error": "Send at most 200 readings per request"}), 400

    try:
        catalog = known_vehicle_routes()
    except Exception as exc:
        print("TELEMETRY CATALOG ERROR:", repr(exc))
        return jsonify({"error": "Could not read the vehicle catalog"}), 503

    observed_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for index, reading in enumerate(readings):
        try:
            rows.append(build_telemetry_row(reading, catalog, observed_at))
        except ValueError as exc:
            return jsonify({"error": f"reading {index}: {exc}"}), 400

    # PostgREST rejects a batch whose objects have differing key sets, so pad
    # every row with the optional fields the others reported.
    columns = {column for row in rows for column in row}
    rows = [
        {column: row.get(column) for column in columns}
        for row in rows
    ]

    try:
        # buses holds one row per vehicle (bus_id is unique), so repeat readings
        # from the same device update that row instead of colliding with it.
        supabase_request(
            "POST",
            "buses",
            params={"on_conflict": "bus_id"},
            json_body=rows,
            prefer="resolution=merge-duplicates,return=minimal",
        )
    except Exception as exc:
        print("TELEMETRY INSERT ERROR:", repr(exc))
        return jsonify({"error": "Could not store telemetry in Supabase"}), 503

    return jsonify({
        "status": "accepted",
        "accepted": len(rows),
        "recorded_at": observed_at,
    }), 201


@app.route("/api/demand-forecast")
def demand_forecast():
    try:
        return jsonify(fetch_demand_forecast())
    except Exception as exc:
        print("DEMAND FORECAST ERROR:", repr(exc))
        return jsonify({
            "status": "offline",
            "error": "Could not build demand forecast from Supabase data",
        }), 503


@app.route("/api/overview")
def overview():
    try:
        return jsonify(fetch_overview())
    except Exception as exc:
        print("OVERVIEW ERROR:", repr(exc))
        return jsonify({
            "status": "offline",
            "error": "Could not build the operations overview",
        }), 503


@app.route("/api/routes")
def route_catalog():
    try:
        return jsonify(fetch_routes())
    except Exception as exc:
        print("ROUTES ERROR:", repr(exc))
        return jsonify({
            "status": "offline",
            "error": "Could not load route signals",
        }), 503


@app.route("/api/fitness")
def fitness():
    try:
        return jsonify(fetch_fitness())
    except Exception as exc:
        print("FITNESS ERROR:", repr(exc))
        return jsonify({
            "status": "offline",
            "error": "Could not load fitness routes and facilities",
        }), 503


@app.route("/api/traffic-simulation", methods=["POST"])
def simulate_traffic():
    payload = request.get_json(silent=True) or {}
    try:
        vehicle_count = int(payload.get("vehicle_count"))
    except (TypeError, ValueError):
        return jsonify({"error": "vehicle_count must be an integer"}), 400
    if vehicle_count < 1 or vehicle_count > 500:
        return jsonify({"error": "vehicle_count must be between 1 and 500"}), 400
    try:
        result, status_code = traffic_simulation(vehicle_count)
        return jsonify(result), status_code
    except Exception as exc:
        print("TRAFFIC SIMULATION ERROR:", repr(exc))
        return jsonify({
            "status": "offline",
            "error": "Could not load the live simulation baseline",
        }), 503


# -----------------------------------
# HOME
# -----------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Hyderabad Urban Intelligence Backend is running"
    })


@app.route("/api/health")
def health():
    models, _ = model_readiness()
    core_ready = all(
        models.get(name) for name in ("vehicle", "garbage", "pothole")
    )
    return jsonify({
        "status": "ok" if core_ready else "degraded",
        "models": models,
        "failed_models": [name for name, ready in models.items() if not ready],
    }), 200 if core_ready else 503


# -----------------------------------
# VEHICLE DETECTION
# -----------------------------------

@app.route("/api/vehicle-detect", methods=["POST"])
def vehicle_detect():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    temp_path = save_upload(request.files["image"], "vehicle")

    results = get_vehicle_model().predict(
        source=str(temp_path),
        conf=0.25
    )

    vehicles = []
    person_count = 0

    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id == 0:
                person_count += 1
                continue

            if class_id in vehicle_classes:

                confidence = float(box.conf[0])

                vehicles.append({
                    "label": vehicle_classes[class_id],
                    "confidence": round(
                        confidence * 100,
                        2
                    ),
                    "box": [
                        round(x, 2)
                        for x in box.xyxy[0].tolist()
                    ]
                })

    persist_vehicle_density(
        # Stills uploaded from the console are real observations but come from an
        # operator device rather than a bus camera, so they are labelled as such.
        request.form.get("bus_id") or "OPERATOR-UPLOAD",
        request.form.get("route_id"),
        person_count,
        len(vehicles),
    )
    return jsonify({
        "vehicle_count": len(vehicles),
        "person_count": person_count,
        "vehicles": vehicles
    })
# -----------------------------------
# NUMBER PLATE DETECTION + OCR
# -----------------------------------

@app.route("/api/plate-detect", methods=["POST"])
def plate_detect():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    temp_path = save_upload(request.files["image"], "plate")

    results = get_plate_model().predict(
        source=str(temp_path),
        conf=0.25
    )

    plates = []

    import cv2

    image_cv = cv2.imread(str(temp_path))

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = [
                int(x)
                for x in box.xyxy[0].tolist()
            ]

            plate_crop = image_cv[y1:y2, x1:x2]

            ocr_results = get_plate_reader().readtext(
            plate_crop
            )

            plate_text = ""
            ocr_confidence = 0

            if ocr_results:

                best_result = max(
                    ocr_results,
                    key=lambda item: item[2]
                )

                plate_text = best_result[1]
                ocr_confidence = best_result[2] * 100

            plates.append({
                "plate": plate_text,
                "detection_confidence": round(
                    confidence * 100, 2
                ),
                "ocr_confidence": round(
                    ocr_confidence, 2
                ),
                "box": [x1, y1, x2, y2]
            })

    return jsonify({
        "plate_count": len(plates),
        "plates": plates
    })
# -----------------------------------
# RASH DRIVING DETECTION
# -----------------------------------

@app.route("/api/rash-driving", methods=["POST"])
def rash_driving():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    temp_path = save_upload(request.files["image"], "rash")

    results = get_vehicle_model().predict(
    source=str(temp_path),
    conf=0.25
    )

    detected_vehicles = []

    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id in vehicle_classes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = [
                    int(x)
                    for x in box.xyxy[0].tolist()
                ]

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                detected_vehicles.append({
                    "label": vehicle_classes[class_id],
                    "confidence": round(
                        confidence * 100, 2
                    ),
                    "center": [
                        center_x,
                        center_y
                    ],
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                })

        motion_results = []

    for vehicle in detected_vehicles:

        vehicle_id = vehicle["label"]

        current_x, current_y = vehicle["center"]

        previous = previous_vehicle_positions.get(vehicle_id)

        motion_score = 0

        if previous:
            previous_x, previous_y = previous["center"]
            previous_time = previous["time"]

            distance = (
                (current_x - previous_x) ** 2
                + (current_y - previous_y) ** 2
            ) ** 0.5

            elapsed_time = time.time() - previous_time

            if elapsed_time > 0:
                motion_score = distance / elapsed_time

        previous_vehicle_positions[vehicle_id] = {
            "center": vehicle["center"],
            "time": time.time()
        }

        motion_results.append({
            **vehicle,
            "motion_score": round(motion_score, 2)
        })

    return jsonify({
        "vehicle_count": len(motion_results),
        "vehicles": motion_results,
        "timestamp": time.time()
    })
# -----------------------------------
# TRAFFIC DATA
# -----------------------------------

@app.route("/api/traffic")
def traffic():
    global _traffic_cache
    try:
        route_id = request.args.get("route_id")
        bus_id = request.args.get("bus_id")
        cache_age = time.monotonic() - _traffic_cache["fetched_at"]
        if _traffic_cache["payload"] is not None and cache_age <= 15:
            payload = {**_traffic_cache["payload"], "cache_age_seconds": round(cache_age)}
            persist_traffic_snapshot(payload, route_id=route_id, bus_id=bus_id)
            persist_route_condition(
                route_id,
                source="TomTom",
                congestion_score=payload["congestion_score"],
            )
            return jsonify(payload)

        api_key = os.getenv("TOMTOM_API_KEY")

        if not api_key:
            return jsonify({
                "error": "TomTom API key not found"
            }), 500
        api_key = api_key.strip()

        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/10/json"

        params = {
            "key": api_key,
            "point": f"{HYDERABAD_POINT[0]},{HYDERABAD_POINT[1]}",
            "unit": "KMPH"
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            return jsonify({
                "error": "TomTom API request failed",
                "status": response.status_code
            }), response.status_code

        tomtom_data = response.json()["flowSegmentData"]

        current_speed = tomtom_data["currentSpeed"]
        free_flow_speed = tomtom_data["freeFlowSpeed"]

        if free_flow_speed > 0:
            congestion_score = round(
                (1 - (current_speed / free_flow_speed)) * 100,
                2
            )
        else:
            congestion_score = 0

        congestion_score = max(0, min(congestion_score, 100))

        if congestion_score < 25:
            level = "Low"
        elif congestion_score < 50:
            level = "Moderate"
        elif congestion_score < 75:
            level = "High"
        else:
            level = "Severe"

        payload = {
            "city": "Hyderabad",
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "congestion_score": congestion_score,
            "congestion_level": level,
            "current_travel_time": tomtom_data["currentTravelTime"],
            "free_flow_travel_time": tomtom_data["freeFlowTravelTime"],
            "confidence": tomtom_data["confidence"],
            "road_closure": tomtom_data["roadClosure"],
            "data_source": "TomTom Traffic Flow"
        }
        _traffic_cache = {
            "payload": payload,
            "fetched_at": time.monotonic(),
        }
        persist_traffic_snapshot(payload, route_id=route_id, bus_id=bus_id)
        persist_route_condition(
            route_id,
            source="TomTom",
            congestion_score=congestion_score,
        )
        return jsonify(payload)

    except requests.exceptions.RequestException as e:
        print("TOMTOM CONNECTION ERROR:", repr(e))

        return jsonify({
            "error": "Could not connect to TomTom Traffic API"
        }), 503

    except Exception as e:
        print("TRAFFIC ERROR:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/traffic-history")
def traffic_history():
    return jsonify(fetch_traffic_history())


@app.route("/api/garbage-detect", methods=["POST"])
def garbage_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_path = save_upload(request.files["image"], "garbage")

    try:
        from src.garbage_ai.garbage_detector import detect_garbage

        detections = detect_garbage(str(image_path))
        alert_created = False

        if detections:
            try:
                create_incident(
                    incident_type="Garbage",
                    severity="Medium",
                    location="Hyderabad",
                    description=f"{len(detections)} garbage item(s) detected by AI.",
                    bus_id=request.form.get("bus_id"),
                    route_id=request.form.get("route_id"),
                    recommendation="Schedule cleaning for the detected area.",
                )
                alert_created = True
            except Exception as exc:
                print("GARBAGE INCIDENT ERROR:", repr(exc))

        return jsonify({
            "detections": detections,
            "alert_created": alert_created,
        })
    except Exception as error:
        app.logger.exception("Garbage detection failed")
        return jsonify({"error": str(error)}), 500
    finally:
        image_path.unlink(missing_ok=True)
    # -----------------------------------
# POTHOLE DETECTION
# -----------------------------------

@app.route("/api/pothole-detect", methods=["POST"])
def pothole_detect():

    try:
        from src.pothole_ai.pothole_detector import detect_potholes

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        image_path = save_upload(request.files["image"], "pothole")

        try:
            detections = detect_potholes(str(image_path))
            alert_created = False
            if detections:
                severity = "High" if len(detections) >= 3 else "Medium"
                route_id = request.form.get("route_id")
                try:
                    create_incident(
                        incident_type="Pothole",
                        severity=severity,
                        location="Hyderabad",
                        description=f"{len(detections)} pothole(s) detected by AI.",
                        bus_id=request.form.get("bus_id"),
                        route_id=route_id,
                        recommendation="Inspect and schedule road repair.",
                    )
                    alert_created = True
                except Exception as exc:
                    print("POTHOLE INCIDENT ERROR:", repr(exc))
                persist_route_condition(
                    route_id,
                    source="UrbanIQ_pothole_detect",
                    pothole_count=len(detections),
                )

            defect_count = len(detections)
            road_risk = (
                "HIGH"
                if defect_count >= 3
                else "MODERATE" if defect_count else "LOW"
            )
            return jsonify({
                "detections": detections,
                "road_risk": road_risk,
                "alert_created": alert_created,
            })
        finally:
            image_path.unlink(missing_ok=True)

    except Exception as e:

        print("POTHOLE ERROR:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500

        # -----------------------------------
# POTHOLE RESULT
# -----------------------------------

@app.route("/api/pothole-result")
def pothole_result():
    return jsonify({
        "status": "deprecated",
        "message": (
            "Annotated images are rendered client-side from /api/pothole-detect "
            "boxes; uploaded files are deleted after analysis."
        ),
    }), 410
# -----------------------------------
# START FLASK
# -----------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
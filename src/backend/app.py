import sys
from pathlib import Path
import os
import requests
import uuid
import time
import tempfile
from datetime import datetime

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

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Vehicle detection model
# AI models are loaded only when required
vehicle_model = None
plate_model = None
plate_reader = None
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

app = Flask(__name__)
garbage_alerts = []
# Rash driving tracking
previous_vehicle_positions = {}
rash_driving_alerts = []


def save_upload(upload, prefix):
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    handle, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
    os.close(handle)
    upload.save(path)
    return Path(path)

def supabase_base():
    base = os.getenv("SUPABASE_URL")
    return base.rstrip("/") if base else None


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


def create_incident(incident_type, severity, location, description):
    incident_id = str(uuid.uuid4())
    stamp = datetime.now().isoformat()
    rows = supabase_request(
        "POST",
        "incidents",
        json_body={
            "incident_id": incident_id,
            "incident_type": incident_type,
            "severity": severity,
            "location": location,
            "description": description,
            "status": "New",
            "timestamp": stamp,
            "source": "UrbanIQ-api",
        },
    )
    supabase_request(
        "POST",
        "alerts",
        json_body={
            "alert_id": str(uuid.uuid4()),
            "incident_id": incident_id,
            "alert_type": incident_type,
            "severity": severity,
            "location": location,
            "message": description,
            "status": "New",
            "detected_at": stamp,
        },
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


CORS(app)

HYDERABAD_POINT = (17.3850, 78.4867)
_last_traffic_persist = 0


def supabase_headers():
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not key:
        return None
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def persist_traffic_snapshot(payload):
    """Write into the existing traffic_realtime table (service role)."""
    global _last_traffic_persist
    headers = supabase_headers()
    base = os.getenv("SUPABASE_URL")
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

    try:
        response = requests.post(
            f"{base.rstrip('/')}/rest/v1/traffic_realtime",
            headers=headers,
            json=row,
            timeout=10,
        )
        if response.ok:
            _last_traffic_persist = now
        else:
            print("SUPABASE TRAFFIC INSERT:", response.status_code, response.text[:200])
    except Exception as exc:
        print("SUPABASE TRAFFIC INSERT ERROR:", repr(exc))


def fetch_traffic_history(limit=30):
    headers = supabase_headers()
    base = os.getenv("SUPABASE_URL")
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
    model_paths = {
        "vehicle": PROJECT_ROOT / "yolo11n.pt",
        "garbage": PROJECT_ROOT / "src" / "garbage_ai" / "best.pt",
        "pothole": PROJECT_ROOT / "weights" / "pothole2v.pt",
    }
    return jsonify({
        "status": "ok",
        "models": {
            name: path.is_file() for name, path in model_paths.items()
        },
    })


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

    return jsonify({
        "vehicle_count": len(vehicles),
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
    try:
        api_key = os.getenv("TOMTOM_API_KEY")

        if not api_key:
            return jsonify({
                "error": "TomTom API key not found"
            }), 500

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
        persist_traffic_snapshot(payload)
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

        if detections:
            try:
                create_incident(
                    incident_type="Garbage",
                    severity="Medium",
                    location="Hyderabad",
                    description=f"{len(detections)} garbage item(s) detected by AI.",
                )
            except Exception as exc:
                print("GARBAGE INCIDENT ERROR:", repr(exc))
            garbage_alerts.append({
                "id": str(uuid.uuid4()),
                "type": "Garbage",
                "message": f"{len(detections)} garbage object(s) detected",
                "location": "Hyderabad",
                "timestamp": datetime.now().isoformat()
            })

        return jsonify({"detections": detections})
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
            if detections:
                severity = "High" if len(detections) >= 3 else "Medium"
                try:
                    create_incident(
                        incident_type="Pothole",
                        severity=severity,
                        location="Hyderabad",
                        description=f"{len(detections)} pothole(s) detected by AI."
                    )
                except Exception as exc:
                    print("POTHOLE INCIDENT ERROR:", repr(exc))

            return jsonify({"detections": detections})
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

    project_root = Path(__file__).resolve().parents[2]

    result_folder = project_root / "runs" / "detect" / "predict"

    image_path = result_folder / "Pothole.jpg"

    if not image_path.exists():
        return jsonify({
            "error": "Pothole result image not found"
        }), 404

    return send_file(
        str(image_path),
        mimetype="image/jpeg"
    )
# -----------------------------------
# START FLASK
# -----------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
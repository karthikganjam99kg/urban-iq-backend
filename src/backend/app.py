import sys
from pathlib import Path
import os
import requests
import random
import uuid
import json
import time
import shutil
import tempfile
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

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
SOURCE_DATA_FILE = Path(__file__).resolve().parent / "urban_data.json"
DATA_FILE = (
    Path("/tmp/urban_data.json") if os.getenv("VERCEL") else SOURCE_DATA_FILE
)


def ensure_data_file():
    if DATA_FILE != SOURCE_DATA_FILE and not DATA_FILE.exists():
        shutil.copyfile(SOURCE_DATA_FILE, DATA_FILE)


def save_upload(upload, prefix):
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    handle, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
    os.close(handle)
    upload.save(path)
    return Path(path)

def load_data():
    ensure_data_file()
    if not DATA_FILE.exists():
        return {
            "incidents": [],
            "alerts": [],
            "road_conditions": [],
            "traffic_history": []
        }

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    ensure_data_file()
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_incident(incident_type, severity, location, description):
    data = load_data()

    incident = {
        "id": str(uuid.uuid4()),
        "type": incident_type,
        "severity": severity,
        "location": location,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "status": "New"
    }

    data["incidents"].append(incident)

    alert = {
        "id": str(uuid.uuid4()),
        "incident_id": incident["id"],
        "type": incident_type,
        "severity": severity,
        "location": location,
        "message": description,
        "timestamp": incident["timestamp"],
        "status": "New"
    }

    data["alerts"].append(alert)

    save_data(data)

    return incident
CORS(app)


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
    data = load_data()
    return jsonify(data["incidents"])


@app.route("/api/alerts")
def get_alerts():
    data = load_data()
    return jsonify(data["alerts"])
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
            "point": "17.3850,78.4867",
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

        return jsonify({
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
        })

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
@app.route("/api/garbage-detect", methods=["POST"])
def garbage_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_path = save_upload(request.files["image"], "garbage")

    try:
        from src.garbage_ai.garbage_detector import detect_garbage

        detections = detect_garbage(str(image_path))

        if detections:
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

                create_incident(
                    incident_type="Pothole",
                    severity=severity,
                    location="Hyderabad",
                    description=f"{len(detections)} pothole(s) detected by AI."
                )

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
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv("POTHOLE_MODEL_PATH", PROJECT_ROOT / "weights" / "pothole2v.pt")
)

model = None


def get_pothole_model():
    global model

    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Pothole model not found: {MODEL_PATH}")

        from ultralytics import YOLO

        model = YOLO(str(MODEL_PATH))

    return model

def detect_potholes(image_path):
    results = get_pothole_model().predict(
        source=image_path,
        conf=0.25,
        save=False,
        imgsz=320
    )

    detections = []

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": "pothole",
                "confidence": round(confidence * 100, 2),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    return detections


if __name__ == "__main__":
    print("Pothole AI detector is ready.")
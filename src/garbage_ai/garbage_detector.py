import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv(
        "GARBAGE_MODEL_PATH",
        PROJECT_ROOT / "src" / "garbage_ai" / "best.pt",
    )
)
model = None


def get_garbage_model():
    global model

    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Garbage model not found: {MODEL_PATH}")

        from ultralytics import YOLO

        model = YOLO(str(MODEL_PATH))

    return model

# COCO classes that can represent visible waste
GARBAGE_CLASSES = {
   "glass",
   "paper",
   "plastic",
   "trash"
}


def detect_garbage(image_path):

    results = get_garbage_model().predict(
        source=image_path,
        conf=0.10
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            label = result.names[class_id]

            if label in GARBAGE_CLASSES:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": "garbage",
                    "object": label,
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
    print("Garbage AI detector is ready.")
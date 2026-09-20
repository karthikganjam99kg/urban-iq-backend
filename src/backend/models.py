"""Lazy model registry shared by health checks and inference endpoints."""

import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_vehicle_model = None
_plate_model = None
_plate_reader = None
_readiness_lock = threading.Lock()
_readiness_cache = {"checked_at": 0.0, "ready": {}, "errors": {}}


def get_vehicle_model():
    global _vehicle_model
    if _vehicle_model is None:
        from ultralytics import YOLO

        _vehicle_model = YOLO(str(PROJECT_ROOT / "weights" / "vehicle.pt"))
    return _vehicle_model


def get_plate_model():
    global _plate_model
    if _plate_model is None:
        from ultralytics import YOLO

        _plate_model = YOLO(str(PROJECT_ROOT / "weights" / "license_plate.pt"))
    return _plate_model


def get_plate_reader():
    global _plate_reader
    if _plate_reader is None:
        import easyocr

        _plate_reader = easyocr.Reader(["en"], gpu=False)
    return _plate_reader


def model_readiness():
    """Load each model once and cache readiness for short health-check bursts."""
    global _readiness_cache
    from src.garbage_ai.garbage_detector import get_garbage_model
    from src.pothole_ai.pothole_detector import get_pothole_model

    with _readiness_lock:
        cache_age = time.monotonic() - _readiness_cache["checked_at"]
        if _readiness_cache["ready"] and cache_age <= 30:
            return (
                dict(_readiness_cache["ready"]),
                dict(_readiness_cache["errors"]),
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

        _readiness_cache = {
            "checked_at": time.monotonic(),
            "ready": ready,
            "errors": errors,
        }
        return dict(ready), dict(errors)

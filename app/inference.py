import base64
import binascii
import os
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"


def demo_mode() -> bool:
    return os.getenv("DEMO_MODE") == "1"


# Topología estándar de BlazePose (35 conexiones)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


class ImagenInvalida(ValueError):
    pass


@lru_cache(maxsize=1)
def get_detector():
    base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.PoseLandmarkerOptions(base_options=base)
    return vision.PoseLandmarker.create_from_options(opts)


def close_detector():
    if get_detector.cache_info().currsize:
        get_detector().close()
        get_detector.cache_clear()


def _decode(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64.split(",")[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ImagenInvalida("base64 inválido")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImagenInvalida("no se pudo decodificar la imagen")
    return bgr


def _demo_landmarks() -> list[dict]:
    return [{"x": 0.5, "y": (i + 1) / 34, "z": 0.0, "visibility": 1.0}
            for i in range(33)]


def estimate_pose(b64: str) -> list[dict]:
    bgr = _decode(b64)          # valida la imagen también en modo demo
    if demo_mode():
        return _demo_landmarks()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = get_detector().detect(img)
    if not res.pose_landmarks:
        return []
    return [{"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility}
            for p in res.pose_landmarks[0]]
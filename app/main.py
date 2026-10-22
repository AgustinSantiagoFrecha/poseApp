import base64
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pydantic import BaseModel

app = FastAPI()

base = python.BaseOptions(model_asset_path="app/pose_landmarker_lite.task")
opts = vision.PoseLandmarkerOptions(base_options=base)
detector = vision.PoseLandmarker.create_from_options(opts)


class FrameIn(BaseModel):
    image: str  # imagen en base64


@app.post("/api/pose")
def pose(data: FrameIn):
    b64 = data.image.split(",")[-1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"landmarks": []}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(img)
    if not res.pose_landmarks:
        return {"landmarks": []}
    return {
        "landmarks": [
            {"x": p.x, "y": p.y, "visibility": p.visibility}
            for p in res.pose_landmarks[0]
        ]
    }
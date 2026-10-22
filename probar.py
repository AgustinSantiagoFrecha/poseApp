import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base = python.BaseOptions(model_asset_path="app/pose_landmarker_lite.task")
opts = vision.PoseLandmarkerOptions(base_options=base)

with vision.PoseLandmarker.create_from_options(opts) as detector:
    img = mp.Image.create_from_file("foto.jpg")
    res = detector.detect(img)
    if not res.pose_landmarks:
        print("No se detectó ninguna persona")
    else:
        for i, p in enumerate(res.pose_landmarks[0]):
            print(i, round(p.x, 3), round(p.y, 3), round(p.visibility, 2))
            
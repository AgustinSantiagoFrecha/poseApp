import base64
import json
import urllib.request

with open("foto.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/pose",
    data=json.dumps({"image": b64}).encode(),
    headers={"Content-Type": "application/json"},
)
res = json.load(urllib.request.urlopen(req))
print("Landmarks recibidos:", len(res["landmarks"]))
print(res["landmarks"][:2])
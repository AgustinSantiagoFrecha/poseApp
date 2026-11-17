from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import CONNECTIONS, ImagenInvalida, close_detector, estimate_pose

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_detector()          # libera el modelo al apagar


app = FastAPI(lifespan=lifespan)


class FrameIn(BaseModel):
    image: str = Field(max_length=5_000_000)   # ~3,7 MB de imagen


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/pose")
def pose(data: FrameIn):
    try:
        landmarks = estimate_pose(data.image)
    except ImagenInvalida as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="error en la inferencia")
    return {"landmarks": landmarks, "connections": CONNECTIONS}


# Siempre al final, después de los endpoints
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
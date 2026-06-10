from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes.jobs import router as jobs_router

app = FastAPI(title="Audio to Sheet Music API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(jobs_router)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
app.mount(
    "/artifacts",
    StaticFiles(directory=PROJECT_ROOT / "backend" / "storage"),
    name="artifacts",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

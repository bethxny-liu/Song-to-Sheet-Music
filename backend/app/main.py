from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes.jobs import router as jobs_router
from backend.app.config import CORS_ORIGINS, STORAGE_DIR

app = FastAPI(title="Audio to Sheet Music API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(jobs_router)
app.mount(
    "/artifacts",
    StaticFiles(directory=STORAGE_DIR),
    name="artifacts",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

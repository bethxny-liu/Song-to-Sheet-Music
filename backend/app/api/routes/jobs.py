import asyncio
from functools import lru_cache, partial
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.app.config import STORAGE_DIR
from backend.app.schemas import ConversionOptions, ConversionResult
from backend.app.services.conversion_service import ConversionService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@lru_cache
def get_conversion_service() -> ConversionService:
    return ConversionService(storage_dir=STORAGE_DIR)


@router.post("/convert", response_model=ConversionResult)
async def convert_audio(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    composer: str = Form("Unknown"),
    tempo_bpm: int = Form(90),
    instrument_name: str = Form("piano"),
    layout: str = Form("melody"),
    isolate_piano: bool = Form(False),
    service: ConversionService = Depends(get_conversion_service),
) -> ConversionResult:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    suffix = Path(file.filename).suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        content = await file.read()
        temp.write(content)
        temp_file_path = Path(temp.name)

    options = ConversionOptions(
        title=title,
        composer=composer,
        tempo_bpm=tempo_bpm,
        instrument_name=instrument_name,
        layout=layout if layout in ("melody", "grand") else "melody",
        isolate_piano=isolate_piano,
    )
    base_url = str(request.base_url).rstrip("/")
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            partial(service.convert, temp_file_path, options, base_url),
        )
    finally:
        temp_file_path.unlink(missing_ok=True)

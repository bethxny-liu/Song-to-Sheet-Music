import asyncio
import logging
from functools import lru_cache, partial
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from algo.audio import AudioInputError
from services.app.config import MAX_UPLOAD_BYTES, STORAGE_DIR
from services.app.schemas import ConversionOptions, ConversionResult
from services.app.conversion import ConversionService

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)


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
    service: ConversionService = Depends(get_conversion_service),
) -> ConversionResult:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    try:
        options = ConversionOptions(
            title=title, composer=composer, tempo_bpm=tempo_bpm,
            instrument_name=instrument_name,
        )
    except ValidationError as exc:
        details = [
            {"loc": ["body", *error["loc"]], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        raise HTTPException(status_code=422, detail=details) from exc

    temp_file_path = None
    try:
        suffix = Path(file.filename).suffix[:16] or ".wav"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_file_path = Path(temp.name)
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload must be 25 MB or smaller.")
                temp.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        base_url = str(request.base_url).rstrip("/")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, partial(service.convert, temp_file_path, options, base_url),
            )
        except AudioInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Audio conversion failed")
            raise HTTPException(
                status_code=500,
                detail="Conversion failed. Try a shorter, clearer recording or a different audio file.",
            ) from exc
    finally:
        if temp_file_path is not None:
            temp_file_path.unlink(missing_ok=True)

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.app.schemas import ConversionOptions, ConversionResult
from backend.app.services.conversion_service import ConversionService

router = APIRouter(prefix="/jobs", tags=["jobs"])
PROJECT_ROOT = Path(__file__).resolve().parents[4]
service = ConversionService(storage_dir=PROJECT_ROOT / "backend" / "storage")


@router.post("/convert", response_model=ConversionResult)
async def convert_audio(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    composer: str = Form("Unknown"),
    tempo_bpm: int = Form(90),
    instrument_name: str = Form("piano"),
    layout: str = Form("melody"),
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
    )
    return service.convert(temp_file_path, options, str(request.base_url).rstrip("/"))

"""FastAPI application entrypoint for the CLARITY backend."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.gemini_report import generate_report_with_image
from app.heatmap_gen import generate_heatmap
from app.inference import run_prediction
from app.models import MODEL_LABELS, get_predictor


app = FastAPI(title="CLARITY", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif"}


def _validate_file(file: Optional[UploadFile]) -> None:
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")


def _handle_prediction_error(exc: ValueError) -> None:
    message = str(exc)
    status = 503 if "not available" in message else 400
    raise HTTPException(status_code=status, detail=message) from exc


@app.get("/")
def root():
    return {"message": "CLARITY API v2.0", "models": list(MODEL_LABELS.keys())}


@app.get("/health")
def health():
    densenet = get_predictor("densenet121")
    resnet = get_predictor("resnet152")
    device = densenet.device if densenet else settings.DEVICE

    return {
        "status": "healthy",
        "device": str(device),
        "densenet_loaded": densenet is not None,
        "resnet_loaded": resnet is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("densenet121")):
    _validate_file(file)
    image_data = await file.read()

    try:
        result = run_prediction(image_data, model)
    except ValueError as exc:
        _handle_prediction_error(exc)

    return {
        "success": True,
        "model_used": result.model_used,
        "predictions": result.predictions,
        "positive_findings": result.positive_findings,
        "positive_labels": result.positive_labels,
        "confidence": result.confidence,
    }


@app.post("/predict/report")
async def predict_report(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    patient_id: str = Form(default=""),
    email: str = Form(default=""),
    model: str = Form("densenet121"),
):
    _validate_file(file)
    image_data = await file.read()

    try:
        result = run_prediction(image_data, model)
    except ValueError as exc:
        _handle_prediction_error(exc)

    patient_info = {
        "name": name,
        "age": age,
        "gender": gender,
        "patient_id": patient_id,
        "email": email,
    }

    report, warning = generate_report_with_image(
        patient_info=patient_info,
        predictions=result.predictions,
        model_used=result.model_used,
        image_data=image_data,
    )

    return {
        "success": True,
        "patient_info": patient_info,
        "predictions": result.predictions,
        "positive_findings": result.positive_findings,
        "positive_labels": result.positive_labels,
        "model_used": result.model_used,
        "report": report,
        "warning": warning,
    }


@app.post("/predict/heatmap")
async def predict_heatmap(
    file: UploadFile = File(...),
    model: str = Form("densenet121"),
    method: str = Form("gradcam"),
    layer: Optional[str] = Form(None),
):
    _validate_file(file)
    image_data = await file.read()

    try:
        heatmap_payload = generate_heatmap(
            image_data=image_data,
            model_name=model,
            method=method,
            layer=layer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return heatmap_payload


@app.get("/config/layers/{model_name}")
def get_layers(model_name: str):
    if model_name == "densenet121":
        return {"model": MODEL_LABELS["densenet121"], "layers": settings.DENSENET121_LAYERS}
    if model_name == "resnet152":
        return {"model": MODEL_LABELS["resnet152"], "layers": settings.RESNET152_LAYERS}
    raise HTTPException(status_code=400, detail="Invalid model name")


@app.get("/config/methods/{model_name}")
def get_methods(model_name: str):
    if model_name == "densenet121":
        return {
            "model": MODEL_LABELS["densenet121"],
            "methods": settings.HEATMAP_METHODS_DENSENET,
        }
    if model_name == "resnet152":
        return {
            "model": MODEL_LABELS["resnet152"],
            "methods": settings.HEATMAP_METHODS_RESNET,
        }
    raise HTTPException(status_code=400, detail="Invalid model name")

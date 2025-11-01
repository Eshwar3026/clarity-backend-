"""Inference utilities for running model predictions on uploaded images."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.models import MODEL_LABELS, get_predictor


@dataclass
class PredictionResult:
    """Structured prediction output used across API endpoints."""

    model_key: str
    model_used: str
    predictions: Dict[str, float]
    positive_findings: List[Dict[str, float]]
    positive_labels: List[str]
    probabilities: np.ndarray

    @property
    def confidence(self) -> float:
        return float(np.max(self.probabilities))

    @property
    def top_index(self) -> int:
        return int(np.argmax(self.probabilities))


def _ensure_model_key(model_name: str) -> str:
    if model_name not in MODEL_LABELS:
        raise ValueError("Invalid model name")
    return model_name


def _load_image(image_data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unable to decode image data") from exc


def _build_predictions(probabilities: Sequence[float]) -> Dict[str, float]:
    return {
        disease: float(prob)
        for disease, prob in zip(settings.LABEL_COLS, probabilities)
    }


def _build_positive_findings(probabilities: Sequence[float]) -> List[Dict[str, float]]:
    positive = [
        {"disease": disease, "probability": float(prob)}
        for disease, prob in zip(settings.LABEL_COLS, probabilities)
        if prob >= settings.CONFIDENCE_THRESHOLD
    ]
    positive.sort(key=lambda item: item["probability"], reverse=True)
    return positive


def run_prediction(image_data: bytes, model_name: str) -> PredictionResult:
    """Run inference for the requested model and return structured output."""

    model_key = _ensure_model_key(model_name)
    predictor = get_predictor(model_key)
    if predictor is None:
        raise ValueError(f"{model_key} model is not available")

    image = _load_image(image_data)
    input_tensor = predictor.transforms(image).unsqueeze(0).to(predictor.device)

    with torch.no_grad():
        logits = predictor.model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = _build_predictions(probabilities)
    positive_findings = _build_positive_findings(probabilities)
    positive_labels = [item["disease"] for item in positive_findings]

    return PredictionResult(
        model_key=model_key,
        model_used=MODEL_LABELS[model_key],
        predictions=predictions,
        positive_findings=positive_findings,
        positive_labels=positive_labels,
        probabilities=probabilities,
    )
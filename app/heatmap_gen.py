"""Utilities for generating heatmaps for model explanations."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.config import settings
from app.models import MODEL_LABELS, get_predictor


class HeatmapGenerationError(RuntimeError):
    """Raised when heatmap generation fails."""


def _ensure_model_key(model_name: str) -> str:
    if model_name not in MODEL_LABELS:
        raise ValueError("Invalid model name")
    return model_name


def _resolve_method(model_key: str, method: str) -> str:
    if model_key == "densenet121":
        allowed = settings.HEATMAP_METHODS_DENSENET
    else:
        allowed = settings.HEATMAP_METHODS_RESNET

    if method not in allowed:
        raise ValueError("Invalid heatmap method for the selected model")
    return method


def _resolve_layer(model_key: str, layer: Optional[str]) -> str:
    candidates = (
        settings.DENSENET121_LAYERS
        if model_key == "densenet121"
        else settings.RESNET152_LAYERS
    )

    if layer is None:
        return candidates[-1]
    if layer not in candidates:
        raise ValueError("Invalid layer for the selected model")
    return layer


def _load_image(image_data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unable to decode image data") from exc


def _build_predictions(probabilities: Sequence[float]) -> dict:
    return {
        disease: float(prob)
        for disease, prob in zip(settings.LABEL_COLS, probabilities)
    }


def _encode_heatmap(heatmap_array: np.ndarray) -> str:
    heatmap_uint8 = np.uint8(np.clip(heatmap_array * 255.0, 0, 255))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(colored.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _generate_shap_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    top_index: int,
) -> Optional[np.ndarray]:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise HeatmapGenerationError("SHAP is not installed") from exc

    try:
        background = torch.randn(8, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE, device=device)
        model.eval()
        explainer = shap.DeepExplainer(model, background)

        input_for_shap = input_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            shap_values = explainer.shap_values(input_for_shap, check_additivity=False)

        if isinstance(shap_values, list):
            shap_array = np.array(shap_values)
            heatmap_raw = np.abs(shap_array[top_index, 0, :, :, :])
        else:
            if isinstance(shap_values, torch.Tensor):
                shap_array = shap_values.detach().cpu().numpy()
            else:
                shap_array = np.array(shap_values)
            heatmap_raw = np.abs(shap_array[0, :, :, :])

        heatmap = np.mean(heatmap_raw, axis=0)
        maximum = float(heatmap.max())
        minimum = float(heatmap.min())
        if maximum > minimum:
            heatmap = (heatmap - minimum) / (maximum - minimum)
        else:
            heatmap = np.ones_like(heatmap) * 0.5
        return heatmap
    except Exception as exc:  # noqa: BLE001
        raise HeatmapGenerationError(str(exc)) from exc


def generate_heatmap(
    image_data: bytes,
    model_name: str,
    method: str,
    layer: Optional[str],
) -> dict:
    """Generate a heatmap for the specified model, method, and layer."""

    model_key = _ensure_model_key(model_name)
    method_key = _resolve_method(model_key, method)
    layer_key = _resolve_layer(model_key, layer)

    predictor = get_predictor(model_key)
    if predictor is None:
        raise ValueError(f"{model_key} model is not available")

    image = _load_image(image_data)
    input_tensor = predictor.transforms(image).unsqueeze(0).to(predictor.device)

    with torch.no_grad():
        logits = predictor.model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = _build_predictions(probabilities)
    top_index = int(np.argmax(probabilities))
    top_disease = settings.LABEL_COLS[top_index]

    heatmap_array: Optional[np.ndarray] = None
    method_used = method_key

    if method_key == "shap":
        try:
            heatmap_array = _generate_shap_heatmap(
                predictor.model,
                input_tensor,
                predictor.device,
                top_index,
            )
        except HeatmapGenerationError:
            heatmap_array = None
            method_used = "gradcam"

    if heatmap_array is None:
        target_layer = dict(predictor.model.named_modules())[layer_key]

        if method_used == "gradcam":
            cam_class = GradCAM
        elif method_used == "gradcam_pp":
            cam_class = GradCAMPlusPlus
        elif method_used == "layercam":
            cam_class = LayerCAM
        elif method_used == "scorecam":
            cam_class = ScoreCAM
        else:
            cam_class = GradCAM

        cam = cam_class(model=predictor.model, target_layers=[target_layer])
        cam_input = input_tensor.detach().clone().requires_grad_(True)
        targets = [ClassifierOutputTarget(top_index)]
        heatmap_array = cam(input_tensor=cam_input, targets=targets)[0]

    if heatmap_array.ndim > 2:
        heatmap_array = np.mean(heatmap_array, axis=tuple(range(2, heatmap_array.ndim)))

    if heatmap_array.shape != (settings.IMAGE_SIZE, settings.IMAGE_SIZE):
        heatmap_array = cv2.resize(
            heatmap_array,
            (settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

    heatmap_array = np.clip(heatmap_array, 0, 1)
    heatmap_base64 = _encode_heatmap(heatmap_array)

    return {
        "success": True,
        "model_used": MODEL_LABELS[model_key],
        "method_used": method_used,
        "layer_used": layer_key,
        "predictions": predictions,
        "heatmap_image": heatmap_base64,
        "top_disease": top_disease,
        "top_probability": float(probabilities[top_index]),
    }

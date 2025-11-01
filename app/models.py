from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "densenet121": "DenseNet121",
    "resnet152": "ResNet152",
}

class DenseNet121Predictor:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transforms = self._get_transforms()
    
    def _load_model(self):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            len(settings.LABEL_COLS),
        )
        checkpoint = torch.load(settings.DENSENET121_PATH, map_location=self.device)
        model.load_state_dict(checkpoint)
        
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        import torch.nn.functional as F

        def new_forward(x):
            features = model.features(x)
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = model.classifier(out)
            return out
        
        model.forward = new_forward

        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        return T.Compose([
            T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

class ResNet152Predictor:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transforms = self._get_transforms()
    
    def _load_model(self):
        model = models.resnet152(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(settings.LABEL_COLS))
        checkpoint = torch.load(settings.RESNET152_PATH, map_location=self.device)
        model.load_state_dict(checkpoint)

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        return T.Compose([
            T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
_densenet_predictor: Optional[DenseNet121Predictor] = None
_densenet_failed = False
_resnet_predictor: Optional[ResNet152Predictor] = None
_resnet_failed = False


def _get_or_create_densenet() -> Optional[DenseNet121Predictor]:
    global _densenet_predictor, _densenet_failed

    if _densenet_predictor or _densenet_failed:
        return _densenet_predictor

    try:
        _densenet_predictor = DenseNet121Predictor()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load DenseNet121 model: %s", exc)
        _densenet_failed = True
        _densenet_predictor = None
    return _densenet_predictor


def _get_or_create_resnet() -> Optional[ResNet152Predictor]:
    global _resnet_predictor, _resnet_failed

    if _resnet_predictor or _resnet_failed:
        return _resnet_predictor

    try:
        _resnet_predictor = ResNet152Predictor()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load ResNet152 model: %s", exc)
        _resnet_failed = True
        _resnet_predictor = None
    return _resnet_predictor


def get_predictor(model_name: str) -> Optional[DenseNet121Predictor | ResNet152Predictor]:
    if model_name == "densenet121":
        return _get_or_create_densenet()
    if model_name == "resnet152":
        return _get_or_create_resnet()
    return None


def get_model(model_name: str):
    predictor = get_predictor(model_name)
    return predictor.model if predictor else None


def get_transform(model_name: str = "densenet121"):
    predictor = get_predictor(model_name)
    return predictor.transforms if predictor else None
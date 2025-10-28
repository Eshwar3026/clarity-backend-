import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
from app.config import settings

class ModelPredictor:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transforms = self._get_transforms()
    
    def _load_model(self):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, len(settings.LABEL_COLS))
        checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
        model.load_state_dict(checkpoint)
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
    
    def predict(self, image_path: str, threshold: float = settings.CONFIDENCE_THRESHOLD):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        predictions = {
            disease: float(prob) 
            for disease, prob in zip(settings.LABEL_COLS, probs)
        }
        
        positive_findings = [
            {"disease": disease, "probability": float(prob)}
            for disease, prob in zip(settings.LABEL_COLS, probs)
            if prob >= threshold
        ]
        positive_findings.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'predictions': predictions,
            'positive_findings': positive_findings,
            'confidence': float(np.max(probs))
        }

predictor = ModelPredictor()
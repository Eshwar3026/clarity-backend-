import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from app.models import densenet_predictor, resnet_predictor
from app.config import settings

class HeatmapGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == "densenet121":
            self.predictor = densenet_predictor
            self.available_methods = settings.HEATMAP_METHODS_DENSENET
            self.available_layers = settings.DENSENET121_LAYERS
        else:
            self.predictor = resnet_predictor
            self.available_methods = settings.HEATMAP_METHODS_RESNET
            self.available_layers = settings.RESNET152_LAYERS
    
    def generate(self, image_path: str, method: str, layer: str):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.predictor.transforms(image).unsqueeze(0).to(self.predictor.device)
        
        with torch.no_grad():
            logits = self.predictor.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        heatmap = None
        
        if method == "shap" and self.model_name == "densenet121":
            try:
                import shap
                background = torch.randn(10, 3, 224, 224).to(self.predictor.device)
                explainer = shap.DeepExplainer(self.predictor.model, background)
                shap_values = explainer.shap_values(input_tensor, check_additivity=False)
                
                top_idx = np.argmax(probs)
                
                if isinstance(shap_values, list):
                    shap_array = np.array(shap_values)
                    heatmap_raw = np.abs(shap_array[top_idx, 0, :, :, :])
                else:
                    heatmap_raw = np.abs(shap_values[0, :, :, :])
                
                heatmap = np.mean(heatmap_raw, axis=0)
                
                if heatmap.max() > heatmap.min():
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                else:
                    heatmap = np.ones_like(heatmap) * 0.5
            
            except Exception as e:
                print(f"SHAP failed: {str(e)}, falling back to GradCAM")
                method = "gradcam"
        
        if heatmap is None:
            try:
                target_layer = dict(self.predictor.model.named_modules())[layer]
                
                if method == "gradcam":
                    cam = GradCAM(model=self.predictor.model, target_layers=[target_layer])
                elif method == "gradcam_pp":
                    cam = GradCAMPlusPlus(model=self.predictor.model, target_layers=[target_layer])
                elif method == "layercam":
                    cam = LayerCAM(model=self.predictor.model, target_layers=[target_layer])
                elif method == "scorecam":
                    cam = ScoreCAM(model=self.predictor.model, target_layers=[target_layer])
                else:
                    cam = GradCAM(model=self.predictor.model, target_layers=[target_layer])
                
                top_idx = np.argmax(probs)
                targets = [ClassifierOutputTarget(top_idx)]
                heatmap = cam(input_tensor=input_tensor, targets=targets)[0]
            
            except Exception as e:
                raise Exception(f"Heatmap generation failed: {str(e)}")
        
        if len(heatmap.shape) > 2:
            heatmap = np.mean(heatmap, axis=tuple(range(2, len(heatmap.shape))))
        
        if heatmap.shape != (224, 224):
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heatmap_color = heatmap_color / 255.0
        
        heatmap_pil = Image.fromarray((heatmap_color * 255).astype(np.uint8))
        buffer = BytesIO()
        heatmap_pil.save(buffer, format='PNG')
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        top_idx = np.argmax(probs)
        top_disease = settings.LABEL_COLS[top_idx]
        
        predictions = {
            disease: float(prob) 
            for disease, prob in zip(settings.LABEL_COLS, probs)
        }
        
        return {
            'heatmap_image': heatmap_base64,
            'predictions': predictions,
            'top_disease': top_disease,
            'top_probability': float(probs[top_idx])
        }
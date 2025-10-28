import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from app.inference import predictor
from app.config import settings

class GradCAMGenerator:
    def __init__(self):
        self.target_layers = [predictor.model.features[-1]]
        self.cam = GradCAM(model=predictor.model, target_layers=self.target_layers)
    
    def generate(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        input_tensor = predictor.transforms(image).unsqueeze(0).to(predictor.device)
        
        with torch.no_grad():
            logits = predictor.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0]
        
        img_rgb = np.array(image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))) / 255.0
        cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        
        cam_pil = Image.fromarray(cam_image)
        buffer = BytesIO()
        cam_pil.save(buffer, format='PNG')
        cam_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        top_idx = np.argmax(probs)
        top_disease = settings.LABEL_COLS[top_idx]
        
        predictions = {
            disease: float(prob) 
            for disease, prob in zip(settings.LABEL_COLS, probs)
        }
        
        return {
            'gradcam_image': cam_base64,
            'predictions': predictions,
            'top_disease': top_disease,
            'top_probability': float(probs[top_idx])
        }

gradcam_gen = GradCAMGenerator()

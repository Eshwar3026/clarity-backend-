# app/heatmap.py

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import io
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from app.models import get_model, get_transform, sigmoid, DEVICE, CLASS_NAMES
from app.config import settings


async def generate_heatmap(file, model_name: str, method: str, layer: str):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if model_name == "densenet121":
            if method not in settings.HEATMAP_METHODS_DENSENET:
                raise HTTPException(status_code=400, detail="Invalid method for DenseNet121")
            if layer is None:
                layer = settings.DENSENET121_LAYERS[-1]
            elif layer not in settings.DENSENET121_LAYERS:
                raise HTTPException(status_code=400, detail="Invalid layer for DenseNet121")
            model_used = "DenseNet121"
            
        elif model_name == "resnet152":
            if method not in settings.HEATMAP_METHODS_RESNET:
                raise HTTPException(status_code=400, detail="Invalid method for ResNet152")
            if layer is None:
                layer = settings.RESNET152_LAYERS[-1]
            elif layer not in settings.RESNET152_LAYERS:
                raise HTTPException(status_code=400, detail="Invalid layer for ResNet152")
            model_used = "ResNet152"
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        target_model = get_model(model_name)
        
        if target_model is None:
            raise HTTPException(status_code=500, detail=f"{model_used} not loaded")
        
        if method == "shap" and model_name != "densenet121":
            raise HTTPException(status_code=400, detail="SHAP only available for DenseNet121")
        
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        transform = get_transform()
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        input_tensor = input_tensor.detach().clone()
        
        with torch.no_grad():
            logits = target_model(input_tensor)
            probs = sigmoid(logits.detach().cpu().numpy()[0])
        
        top_idx = int(np.argmax(probs))
        top_disease = CLASS_NAMES[top_idx]
        
        heatmap = None
        
        if method == "shap":
            try:
                import shap
                
                input_for_shap = input_tensor.detach().clone()
                input_for_shap.requires_grad = True
                
                background = torch.randn(10, 3, 224, 224).to(DEVICE)
                target_model.eval()
                explainer = shap.DeepExplainer(target_model, background)
                
                with torch.enable_grad():
                    shap_values = explainer.shap_values(input_for_shap, check_additivity=False)
                
                if isinstance(shap_values, list):
                    shap_array = np.array(shap_values)
                    heatmap_raw = np.abs(shap_array[top_idx, 0, :, :, :])
                else:
                    if isinstance(shap_values, torch.Tensor):
                        shap_array = shap_values.detach().cpu().numpy()
                    else:
                        shap_array = np.array(shap_values)
                    heatmap_raw = np.abs(shap_array[0, :, :, :])
                
                heatmap = np.mean(heatmap_raw, axis=0)
                
                if heatmap.max() > heatmap.min():
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                else:
                    heatmap = np.ones_like(heatmap) * 0.5
            
            except Exception as e:
                method = "gradcam"
                heatmap = None
        
        if heatmap is None:
            try:
                target_layer = dict(target_model.named_modules())[layer]
                
                if method == "gradcam":
                    cam = GradCAM(model=target_model, target_layers=[target_layer])
                elif method == "gradcam_pp":
                    cam = GradCAMPlusPlus(model=target_model, target_layers=[target_layer])
                elif method == "layercam":
                    cam = LayerCAM(model=target_model, target_layers=[target_layer])
                elif method == "scorecam":
                    cam = ScoreCAM(model=target_model, target_layers=[target_layer])
                else:
                    cam = GradCAM(model=target_model, target_layers=[target_layer])
                
                targets = [ClassifierOutputTarget(top_idx)]
                input_for_cam = input_tensor.detach().clone()
                input_for_cam.requires_grad = True
                heatmap = cam(input_tensor=input_for_cam, targets=targets)[0]
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")
        
        if len(heatmap.shape) > 2:
            heatmap = np.mean(heatmap, axis=tuple(range(2, len(heatmap.shape))))
        
        if heatmap.shape != (224, 224):
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        heatmap_pil = Image.fromarray(heatmap_color.astype(np.uint8))
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format='PNG')
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        predictions = {disease: float(prob) for disease, prob in zip(CLASS_NAMES, probs)}
        
        return {
            "success": True,
            "model_used": model_used,
            "method_used": method,
            "layer_used": layer,
            "predictions": predictions,
            "heatmap_image": heatmap_base64,
            "top_disease": top_disease,
            "top_probability": float(probs[top_idx])
        }
        
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "message": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

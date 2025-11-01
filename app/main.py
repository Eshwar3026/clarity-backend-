# app/main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import io
import traceback
import base64
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from app.config import settings

app = FastAPI(title="CLARITY", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"

model_densenet = None
model_resnet = None
transform = None

CLASS_NAMES = settings.LABEL_COLS


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def load_models():
    global model_densenet, model_resnet, transform
    
    dn_path = settings.DENSENET121_PATH
    rn_path = settings.RESNET152_PATH
    
    try:
        model_densenet = models.densenet121(weights=None)
        model_densenet.classifier = nn.Linear(model_densenet.classifier.in_features, len(CLASS_NAMES))
        model_densenet.load_state_dict(torch.load(dn_path, map_location=DEVICE))
        for module in model_densenet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        model_densenet.eval().to(DEVICE)
    except Exception as e:
        model_densenet = None
    
    try:
        model_resnet = models.resnet152(weights=None)
        model_resnet.fc = nn.Linear(model_resnet.fc.in_features, len(CLASS_NAMES))
        model_resnet.load_state_dict(torch.load(rn_path, map_location=DEVICE))
        for module in model_resnet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        model_resnet.eval().to(DEVICE)
    except Exception as e:
        model_resnet = None
    
    transform = T.Compose([
        T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/")
def root():
    return {"message": "CLARITY API v2.0", "models": ["densenet121", "resnet152"]}


@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "device": str(DEVICE),
        "densenet_loaded": model_densenet is not None,
        "resnet_loaded": model_resnet is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("densenet121")):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        if model == "densenet121":
            if model_densenet is None:
                raise HTTPException(status_code=500, detail="DenseNet121 not loaded")
            with torch.no_grad():
                logits = model_densenet(input_tensor)
            model_used = "DenseNet121"
            
        elif model == "resnet152":
            if model_resnet is None:
                raise HTTPException(status_code=500, detail="ResNet152 not loaded")
            with torch.no_grad():
                logits = model_resnet(input_tensor)
            model_used = "ResNet152"
        else:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        probs = sigmoid(logits.detach().cpu().numpy()[0])
        predictions = {disease: float(prob) for disease, prob in zip(CLASS_NAMES, probs)}
        
        positive_findings = [
            {"disease": disease, "probability": float(prob)}
            for disease, prob in zip(CLASS_NAMES, probs)
            if prob >= settings.CONFIDENCE_THRESHOLD
        ]
        positive_findings.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            "success": True,
            "model_used": model_used,
            "predictions": predictions,
            "positive_findings": positive_findings,
            "confidence": float(np.max(probs))
        }
        
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "message": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.post("/predict/report")
async def predict_report(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    patient_id: str = Form(default=""),
    email: str = Form(default=""),
    model: str = Form("densenet121")
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file")
        
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        if model == "densenet121":
            if model_densenet is None:
                raise HTTPException(status_code=500, detail="DenseNet121 not loaded")
            with torch.no_grad():
                logits = model_densenet(input_tensor)
            model_used = "DenseNet121"
        elif model == "resnet152":
            if model_resnet is None:
                raise HTTPException(status_code=500, detail="ResNet152 not loaded")
            with torch.no_grad():
                logits = model_resnet(input_tensor)
            model_used = "ResNet152"
        else:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        probs = sigmoid(logits.detach().cpu().numpy()[0])
        predictions = {disease: float(prob) for disease, prob in zip(CLASS_NAMES, probs)}
        
        positive_findings = [disease for disease, prob in zip(CLASS_NAMES, probs) if prob >= settings.CONFIDENCE_THRESHOLD]
        
        report = f"""
CLARITY Medical Report
{'='*60}
Patient: {name} | Age: {age} | Gender: {gender}
Patient ID: {patient_id or 'N/A'} | Email: {email or 'N/A'}
Model: {model_used}

Positive Findings: {', '.join(positive_findings) if positive_findings else 'None'}

Detailed Predictions:
{chr(10).join([f'- {d}: {p:.2%}' for d, p in predictions.items() if p >= 0.5])}
        """.strip()
        
        return {
            "success": True,
            "patient_info": {"name": name, "age": age, "gender": gender, "patient_id": patient_id, "email": email},
            "predictions": predictions,
            "model_used": model_used,
            "report": report
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.post("/predict/heatmap")
async def predict_heatmap(
    file: UploadFile = File(...),
    model: str = Form("densenet121"),
    method: str = Form("gradcam"),
    layer: str = Form(None)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if model == "densenet121":
            if method not in settings.HEATMAP_METHODS_DENSENET:
                raise HTTPException(status_code=400, detail=f"Invalid method for DenseNet121")
            if layer is None:
                layer = settings.DENSENET121_LAYERS[-1]
            elif layer not in settings.DENSENET121_LAYERS:
                raise HTTPException(status_code=400, detail=f"Invalid layer for DenseNet121")
            model_used = "DenseNet121"
            target_model = model_densenet
            
        elif model == "resnet152":
            if method not in settings.HEATMAP_METHODS_RESNET:
                raise HTTPException(status_code=400, detail=f"Invalid method for ResNet152")
            if layer is None:
                layer = settings.RESNET152_LAYERS[-1]
            elif layer not in settings.RESNET152_LAYERS:
                raise HTTPException(status_code=400, detail=f"Invalid layer for ResNet152")
            model_used = "ResNet152"
            target_model = model_resnet
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        if target_model is None:
            raise HTTPException(status_code=500, detail=f"{model_used} not loaded")
        
        if method == "shap" and model != "densenet121":
            raise HTTPException(status_code=400, detail="SHAP is only available for DenseNet121")
        
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = target_model(input_tensor)
            probs = sigmoid(logits.detach().cpu().numpy()[0])
        
        top_idx = np.argmax(probs)
        top_disease = CLASS_NAMES[top_idx]
        
        if method == "shap":
            try:
                import shap
                background = torch.randn(10, 3, 224, 224).to(DEVICE)
                explainer = shap.DeepExplainer(target_model, background)
                shap_values = explainer.shap_values(input_tensor, check_additivity=False)
                
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
                
            except Exception:
                method = "gradcam"
        
        if method != "shap":
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
            heatmap = cam(input_tensor=input_tensor, targets=targets)[0]
        
        if len(heatmap.shape) > 2:
            heatmap = np.mean(heatmap, axis=tuple(range(2, len(heatmap.shape))))
        
        if heatmap.shape != (224, 224):
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray((heatmap_color).astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
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


@app.get("/config/layers/{model_name}")
def get_layers(model_name: str):
    try:
        if model_name == "densenet121":
            return {"model": "DenseNet121", "layers": settings.DENSENET121_LAYERS}
        elif model_name == "resnet152":
            return {"model": "ResNet152", "layers": settings.RESNET152_LAYERS}
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.get("/config/methods/{model_name}")
def get_methods(model_name: str):
    try:
        if model_name == "densenet121":
            return {"model": "DenseNet121", "methods": settings.HEATMAP_METHODS_DENSENET}
        elif model_name == "resnet152":
            return {"model": "ResNet152", "methods": settings.HEATMAP_METHODS_RESNET}
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})

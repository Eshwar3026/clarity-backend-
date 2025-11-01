from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from app.config import settings
from app.schemas import *
from app.models import densenet_predictor, resnet_predictor
from app.heatmap_gen import HeatmapGenerator
from app.gemini_report import generate_report
from app.utils import save_uploaded_file, validate_image_file
import os

app = FastAPI(title="CLARITY", version="2.0.0")

@app.get("/")
def root():
    return {"message": "CLARITY API v2.0", "models": ["densenet121", "resnet152"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("densenet121")):
    try:
        if not validate_image_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        filepath = save_uploaded_file(file)
        
        if model == "densenet121":
            result = densenet_predictor.predict(filepath)
            model_used = "DenseNet121"
        elif model == "resnet152":
            result = resnet_predictor.predict(filepath)
            model_used = "ResNet152"
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        os.remove(filepath)
        
        return PredictionResponse(
            success=True,
            model_used=model_used,
            predictions=result['predictions'],
            positive_findings=result['positive_findings'],
            confidence=result['confidence']
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.post("/predict/report")
async def predict_report(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    patient_id: str = Form(None),
    email: str = Form(None),
    model: str = Form("densenet121")
):
    try:
        if not validate_image_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        filepath = save_uploaded_file(file)
        
        if model == "densenet121":
            result = densenet_predictor.predict(filepath)
            model_used = "DenseNet121"
        elif model == "resnet152":
            result = resnet_predictor.predict(filepath)
            model_used = "ResNet152"
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        patient_info = {
            "name": name,
            "age": age,
            "gender": gender,
            "patient_id": patient_id,
            "email": email
        }
        
        report = generate_report(patient_info, result['predictions'], model_used)
        
        os.remove(filepath)
        
        return ReportResponse(
            success=True,
            patient_info=patient_info,
            predictions=result['predictions'],
            model_used=model_used,
            report=report
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.post("/predict/heatmap")
async def predict_heatmap(
    file: UploadFile = File(...),
    model: str = Form("densenet121"),
    method: str = Form("gradcam"),
    layer: str = Form(None)
):
    try:
        if not validate_image_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        if model == "densenet121":
            if method not in settings.HEATMAP_METHODS_DENSENET:
                raise HTTPException(status_code=400, detail=f"Invalid method for DenseNet121. Allowed: {settings.HEATMAP_METHODS_DENSENET}")
            if layer is None:
                layer = settings.DENSENET121_LAYERS[-1]
            model_used = "DenseNet121"
        elif model == "resnet152":
            if method not in settings.HEATMAP_METHODS_RESNET:
                raise HTTPException(status_code=400, detail=f"Invalid method for ResNet152. Allowed: {settings.HEATMAP_METHODS_RESNET}")
            if layer is None:
                layer = settings.RESNET152_LAYERS[-1]
            model_used = "ResNet152"
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        filepath = save_uploaded_file(file)
        
        generator = HeatmapGenerator(model)
        result = generator.generate(filepath, method, layer)
        
        os.remove(filepath)
        
        return HeatmapResponse(
            success=True,
            model_used=model_used,
            method_used=method,
            layer_used=layer,
            predictions=result['predictions'],
            heatmap_image=result['heatmap_image'],
            top_disease=result['top_disease'],
            top_probability=result['top_probability']
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

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
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

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
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )
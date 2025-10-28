from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import ValidationError

from app.config import settings
from app.schemas import PredictionResponse, ReportResponse, GradCAMResponse, PatientInfo, ReportRequest
from app.inference import predictor
from app.gradcam import gradcam_gen
from app.gemini_report import generate_report
from app.utils import save_upload_file, cleanup_file, validate_image

app = FastAPI(
    title="CLARITY - Chest X-Ray Diagnosis API",
    description="AI-powered chest X-ray analysis with DenseNet121",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("✅ Model loaded")
    print(f"✅ Device: {predictor.device}")

@app.get("/")
async def root():
    return {
        "message": "CLARITY API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/predict/report", "/predict/gradcam", "/docs"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        file_path = save_upload_file(file, file.filename)
        
        if not validate_image(file_path):
            cleanup_file(file_path)
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = predictor.predict(file_path, threshold=settings.CONFIDENCE_THRESHOLD)
        cleanup_file(file_path)
        
        return PredictionResponse(
            success=True,
            predictions=result['predictions'],
            positive_findings=result['positive_findings'],
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/report", response_model=ReportResponse)
async def predict_report(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    patient_id: str = Form(None),
    email: str = Form(None)
):
    try:
        file_path = save_upload_file(file, file.filename)
        
        if not validate_image(file_path):
            cleanup_file(file_path)
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = predictor.predict(file_path, threshold=settings.CONFIDENCE_THRESHOLD)
        
        patient_info = PatientInfo(
            name=name,
            age=age,
            gender=gender,
            patient_id=patient_id,
            email=email
        )
        
        report = generate_report(patient_info, result['predictions'])
        cleanup_file(file_path)
        
        return ReportResponse(
            success=True,
            patient_info=patient_info,
            predictions=result['predictions'],
            report=report
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/gradcam", response_model=GradCAMResponse)
async def predict_gradcam(file: UploadFile = File(...)):
    try:
        file_path = save_upload_file(file, file.filename)
        
        if not validate_image(file_path):
            cleanup_file(file_path)
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = gradcam_gen.generate(file_path)
        cleanup_file(file_path)
        
        return GradCAMResponse(
            success=True,
            predictions=result['predictions'],
            gradcam_image=result['gradcam_image'],
            top_disease=result['top_disease'],
            top_probability=result['top_probability']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

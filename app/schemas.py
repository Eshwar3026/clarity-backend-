from pydantic import BaseModel
from typing import Optional, Dict, List

class PatientInfo(BaseModel):
    name: str
    age: int
    gender: str
    patient_id: Optional[str] = None
    email: Optional[str] = None

class PredictionResponse(BaseModel):
    success: bool
    predictions: Dict[str, float]
    positive_findings: List[Dict[str, float]]
    confidence: float
    message: Optional[str] = None

class ReportRequest(BaseModel):
    patient_info: PatientInfo
    predictions: Dict[str, float]

class ReportResponse(BaseModel):
    success: bool
    patient_info: PatientInfo
    predictions: Dict[str, float]
    report: str
    message: Optional[str] = None

class GradCAMResponse(BaseModel):
    success: bool
    predictions: Dict[str, float]
    gradcam_image: str
    top_disease: str
    top_probability: float
    message: Optional[str] = None
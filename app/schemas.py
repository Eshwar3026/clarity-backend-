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
    model_used: str
    predictions: Dict[str, float]
    positive_findings: List[Dict[str, float]]
    confidence: float
    message: Optional[str] = None

class ReportRequest(BaseModel):
    patient_info: PatientInfo
    predictions: Dict[str, float]
    model_used: str

class ReportResponse(BaseModel):
    success: bool
    patient_info: PatientInfo
    predictions: Dict[str, float]
    model_used: str
    report: str
    message: Optional[str] = None

class HeatmapRequest(BaseModel):
    model_name: str
    method: str
    layer: str

class HeatmapResponse(BaseModel):
    success: bool
    model_used: str
    method_used: str
    layer_used: str
    predictions: Dict[str, float]
    heatmap_image: str
    top_disease: str
    top_probability: float
    message: Optional[str] = None
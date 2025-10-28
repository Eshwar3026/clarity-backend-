import google.generativeai as genai
from app.config import settings
from app.schemas import PatientInfo

genai.configure(api_key=settings.GEMINI_API_KEY)

def generate_report(patient_info: PatientInfo, predictions: dict) -> str:
    positive_findings = [
        f"- {disease}: {prob*100:.1f}% confidence"
        for disease, prob in predictions.items()
        if prob >= 0.3
    ]
    
    if not positive_findings:
        positive_findings = ["- No significant findings detected"]
    
    findings_text = "\n".join(positive_findings)
    
    prompt = f"""Generate a professional medical report for:

Patient Name: {patient_info.name}
Age: {patient_info.age}
Gender: {patient_info.gender}

AI-Detected Findings:
{findings_text}

Structure the report with:
1. PATIENT DEMOGRAPHICS
2. CLINICAL FINDINGS
3. IMPRESSION
4. RECOMMENDATIONS
5. DISCLAIMER: "This is an AI-assisted analysis and requires radiologist verification."

Use professional medical terminology and be concise."""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {str(e)}"
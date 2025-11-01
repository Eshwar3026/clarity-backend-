import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

def generate_report(patient_info, predictions, model_used):
    positive_findings = [
        disease for disease, prob in predictions.items()
        if prob >= settings.CONFIDENCE_THRESHOLD
    ]
    
    findings_text = ", ".join(positive_findings) if positive_findings else "No significant findings"
    
    prompt = f"""
You are a medical AI assistant. Generate a professional medical report based on the following data:

PATIENT INFORMATION:
- Name: {patient_info.get('name', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}
- Patient ID: {patient_info.get('patient_id', 'N/A')}

MODEL ANALYSIS:
- Model Used: {model_used}
- Positive Findings: {findings_text}

PREDICTION SCORES:
{chr(10).join([f"- {disease}: {score:.2%}" for disease, score in predictions.items() if score >= settings.CONFIDENCE_THRESHOLD])}

Generate a concise, professional medical report summary based on these findings. Include:
1. Patient demographics
2. Findings from the analysis
3. Clinical significance
4. Recommendations for further evaluation if needed

Keep the report clear and professional for medical use.
"""
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text
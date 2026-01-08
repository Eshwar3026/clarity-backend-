# app/gemini_report.py

import base64
import logging
import math
import re
import time
from typing import Optional, Tuple

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from app.config import settings

logger = logging.getLogger(__name__)

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY is not configured.")

_QUOTA_RETRY_PATTERN = re.compile(r"retry in (?P<seconds>\\d+(?:\\.\\d+)?)s", re.IGNORECASE)
_quota_cooldown_until = 0.0


def _seconds_until_quota_reset() -> int:
    remaining = _quota_cooldown_until - time.monotonic()
    if remaining <= 0:
        return 0
    return int(math.ceil(remaining))


def _in_quota_cooldown() -> bool:
    return _seconds_until_quota_reset() > 0


def _schedule_quota_cooldown(delay_hint: Optional[float]) -> int:
    global _quota_cooldown_until
    cooldown = delay_hint if delay_hint and delay_hint > 0 else settings.GEMINI_RATE_LIMIT_COOLDOWN
    cooldown = max(int(math.ceil(cooldown)), settings.GEMINI_RATE_LIMIT_COOLDOWN)
    _quota_cooldown_until = time.monotonic() + cooldown
    return cooldown


def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
    retry_info = getattr(exc, "retry_info", None)
    retry_delay = getattr(retry_info, "retry_delay", None)
    if retry_delay is not None:
        seconds = getattr(retry_delay, "seconds", 0)
        nanos = getattr(retry_delay, "nanos", 0)
        total = seconds + nanos / 1_000_000_000
        if total > 0:
            return total

    message = str(exc)
    match = _QUOTA_RETRY_PATTERN.search(message)
    if match:
        try:
            return float(match.group("seconds"))
        except ValueError:
            return None

    return None


def _build_quota_warning(retry_after: Optional[int]) -> str:
    if retry_after and retry_after > 0:
        return (
            f"Gemini quota exceeded for model {settings.GEMINI_MODEL}; "
            f"retry after ~{retry_after}s. Showing rule-based summary instead."
        )
    return (
        f"Gemini quota exceeded for model {settings.GEMINI_MODEL}; "
        "showing rule-based summary until quota resets."
    )


def _format_findings(predictions: dict, limit: int = 5) -> list:
    ranked = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    lines = []
    for disease, score in ranked[:limit]:
        lines.append(f"- {disease}: {score:.1%}")
    return lines


def _build_fallback_report(patient_info: dict, predictions: dict, model_used: str) -> str:
    name = patient_info.get("name") or "N/A"
    age = patient_info.get("age") or "N/A"
    gender = patient_info.get("gender") or "N/A"
    patient_id = patient_info.get("patient_id") or "N/A"

    positive_findings = [
        (disease, prob) for disease, prob in predictions.items()
        if prob >= settings.CONFIDENCE_THRESHOLD
    ]
    positive_findings.sort(key=lambda item: item[1], reverse=True)

    if positive_findings:
        summary_intro = "Model-detected findings ranked by confidence:"
        highlighted = [
            f"• {disease} ({prob:.1%} confidence)"
            for disease, prob in positive_findings[:3]
        ]
    else:
        summary_intro = "The model did not detect findings above the confidence threshold."
        highlighted = [
            "• Continue correlating with clinical context and consider radiologist review if symptoms persist."
        ]

    ranked_lines = _format_findings(predictions)
    ranked_section = "\n".join(ranked_lines) if ranked_lines else "- No model predictions available."

    report_sections = [
        "CLINICAL SUMMARY",
        f"Patient: {name} | Age: {age} | Gender: {gender} | ID: {patient_id}",
        f"Model Used: {model_used}",
        "",
        summary_intro,
        *highlighted,
        "",
        "Detailed model probabilities:",
        ranked_section,
        "",
        "Recommendations:",
        "- Correlate with patient history and presenting symptoms.",
        "- Consider additional imaging or specialist referral as clinically indicated.",
    ]

    return "\n".join(report_sections)


def image_to_base64(image_data: bytes) -> str:
    """Convert image bytes to base64"""
    return base64.standard_b64encode(image_data).decode("utf-8")


def generate_report_with_image(
    patient_info: dict,
    predictions: dict,
    model_used: str,
    image_data: bytes
) -> Tuple[str, Optional[str]]:
    """Generate report with image analysis using Gemini API"""
    
    fallback_report = _build_fallback_report(patient_info, predictions, model_used)

    if not settings.GEMINI_API_KEY:
        warning = "Gemini API key is not configured; returning a rule-based summary instead."
        return fallback_report, warning

    if _in_quota_cooldown():
        warning = _build_quota_warning(_seconds_until_quota_reset())
        return fallback_report, warning

    if _in_quota_cooldown():
        warning = _build_quota_warning(_seconds_until_quota_reset())
        return fallback_report, warning

    try:
        positive_findings = [
            disease for disease, prob in predictions.items()
            if prob >= settings.CONFIDENCE_THRESHOLD
        ]
        findings_text = ", ".join(positive_findings) if positive_findings else "No significant findings"

        high_confidence_scores = [
            f"- {disease}: {score:.2%}"
            for disease, score in predictions.items()
            if score >= settings.CONFIDENCE_THRESHOLD
        ]
        high_confidence_section = "\n".join(high_confidence_scores) or "- None above threshold"

        image_base64 = image_to_base64(image_data)

        prompt_text = f"""
You are a professional medical AI assistant. Analyze this medical image and generate a comprehensive medical report.

PATIENT INFORMATION:
- Name: {patient_info.get('name', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}
- Patient ID: {patient_info.get('patient_id', 'N/A')}

AI MODEL ANALYSIS:
- Model Used: {model_used}
- Primary Finding: {findings_text}
- Confidence Scores: 
{high_confidence_section}

Please analyze the provided medical image and:
1. Describe the image findings in detail
2. Correlate with the AI model's predictions
3. Provide clinical significance of the findings
4. Recommend appropriate follow-up actions
5. Highlight any areas of concern requiring specialist attention

Generate a professional, concise medical report suitable for clinical use.
"""

        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        response = model.generate_content([
            prompt_text,
            {
                "mime_type": "image/jpeg",
                "data": image_base64
            }
        ])

        text = getattr(response, "text", None)

        if text:
            return text, None

        logger.warning("Gemini API response did not include text; using fallback summary.")
        warning = "Gemini response was empty; returning a rule-based summary instead."
        return fallback_report, warning
    except google_exceptions.ResourceExhausted as exc:
        logger.warning("Gemini quota exhausted: %s", exc)
        cooldown = _schedule_quota_cooldown(_extract_retry_delay_seconds(exc))
        warning = _build_quota_warning(cooldown)
        return fallback_report, warning
    except Exception as exc:
        logger.exception("Gemini report generation failed: %s", exc)
        warning = "Gemini report generation failed; showing fallback summary instead."
        return fallback_report, warning


def generate_report(
    patient_info: dict,
    predictions: dict,
    model_used: str
) -> Tuple[str, Optional[str]]:
    """Generate report without image (fallback)"""
    fallback_report = _build_fallback_report(patient_info, predictions, model_used)

    if not settings.GEMINI_API_KEY:
        warning = "Gemini API key is not configured; returning a rule-based summary instead."
        return fallback_report, warning

    try:
        positive_findings = [
            disease for disease, prob in predictions.items()
            if prob >= settings.CONFIDENCE_THRESHOLD
        ]
        findings_text = ", ".join(positive_findings) if positive_findings else "No significant findings"

        high_confidence_scores = [
            f"- {disease}: {score:.2%}"
            for disease, score in predictions.items()
            if score >= settings.CONFIDENCE_THRESHOLD
        ]
        high_confidence_section = "\n".join(high_confidence_scores) or "- None above threshold"

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
{high_confidence_section}

Generate a concise, professional medical report summary based on these findings. Include:
1. Patient demographics
2. Findings from the analysis
3. Clinical significance
4. Recommendations for further evaluation if needed

Keep the report clear and professional for medical use.
"""

        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)

        if text:
            return text, None

        logger.warning("Gemini API response did not include text; using fallback summary.")
        warning = "Gemini response was empty; returning a rule-based summary instead."
        return fallback_report, warning
    except google_exceptions.ResourceExhausted as exc:
        logger.warning("Gemini quota exhausted: %s", exc)
        cooldown = _schedule_quota_cooldown(_extract_retry_delay_seconds(exc))
        warning = _build_quota_warning(cooldown)
        return fallback_report, warning
    except Exception as exc:
        logger.exception("Gemini report generation failed: %s", exc)
        warning = "Gemini report generation failed; showing fallback summary instead."
        return fallback_report, warning

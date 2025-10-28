import os
import shutil
from pathlib import Path
from app.config import settings

def save_upload_file(uploaded_file, filename: str) -> str:
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return file_path

def cleanup_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

def validate_image(file_path: str) -> bool:
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    ext = Path(file_path).suffix.lower()
    return ext in valid_extensions

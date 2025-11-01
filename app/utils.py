import os
import uuid
from datetime import datetime

from app.config import settings

def get_upload_file_size(upload_file) -> int:
    """Return the size of the uploaded file without consuming its stream."""
    file_obj = upload_file.file
    current_position = file_obj.tell()
    file_obj.seek(0, os.SEEK_END)
    size = file_obj.tell()
    file_obj.seek(current_position)
    return size


def save_uploaded_file(upload_file) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{upload_file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    upload_file.file.seek(0)
    with open(filepath, "wb") as f:
        f.write(upload_file.file.read())
    
    return filepath

def validate_image_file(filename: str) -> bool:
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.splitext(filename)[1].lower() in allowed_extensions
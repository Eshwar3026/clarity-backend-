import os
import uuid
from datetime import datetime

def save_uploaded_file(upload_file) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{upload_file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(upload_file.file.read())
    
    return filepath

def validate_image_file(filename: str) -> bool:
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.splitext(filename)[1].lower() in allowed_extensions
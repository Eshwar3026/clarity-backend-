import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/best_densenet121_auc_0.9115.pth")
    IMAGE_SIZE = 224
    DEVICE = os.getenv("DEVICE", "cuda")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    LABEL_COLS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
        'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ]
    
    CONFIDENCE_THRESHOLD = 0.5
    UPLOAD_DIR = "./uploads"
    OUTPUT_DIR = "./outputs"
    
    MAX_FILE_SIZE = 50 * 1024 * 1024

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

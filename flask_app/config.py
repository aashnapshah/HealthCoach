"""
Flask Application Configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Flask application configuration"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    PORT = int(os.environ.get('PORT', 8000))
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    ENV = os.environ.get('FLASK_ENV', 'development')
    
    # Hugging Face Configuration
    HF_MODEL = os.environ.get('HF_MODEL', 'meta-llama/Llama-3.1-8B')
    HF_TOKEN = os.environ.get('HF_TOKEN', None)  # Required for gated models like Llama
    HF_CACHE_DIR = os.environ.get('HF_CACHE_DIR', '/n/data1/hms/dbmi/manrai/aashna/HealthCoachV2/cache/huggingface')  # Optional: specify cache location
    
    # RunPod Configuration (Optional)
    RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
    RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID')
    RUNPOD_ENDPOINT_URL = f"https://api.runpod.ai/v2/{os.environ.get('RUNPOD_ENDPOINT_ID', 'YOUR_ENDPOINT_ID')}/run"
    
    # Static Files
    STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

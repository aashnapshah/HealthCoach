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
    
    # Ollama Configuration (for health checks)
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', '127.0.0.1')
    OLLAMA_PORT = os.environ.get('OLLAMA_PORT', '11434')
    OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama2')
    
    # RunPod Configuration (Optional)
    RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
    RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID')
    RUNPOD_ENDPOINT_URL = f"https://api.runpod.ai/v2/{os.environ.get('RUNPOD_ENDPOINT_ID', 'YOUR_ENDPOINT_ID')}/run"
    
    # Static Files
    STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

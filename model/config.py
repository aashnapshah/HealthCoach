"""
RAG and Model Configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Hugging Face cache to project directory (must be set before any HF imports)
HF_CACHE_PATH = '/n/data1/hms/dbmi/manrai/aashna/HealthCoachV2/cache/huggingface'
os.environ.setdefault('HF_HOME', HF_CACHE_PATH)
os.environ.setdefault('TRANSFORMERS_CACHE', HF_CACHE_PATH)
os.environ.setdefault('HF_DATASETS_CACHE', HF_CACHE_PATH)

class ModelConfig:
    """Configuration for RAG and LLM models"""
    
    # Inference Mode: 'local' or 'runpod'
    INFERENCE_MODE = os.environ.get('INFERENCE_MODE', 'local')  # Set to 'runpod' to use RunPod API
    
    # RunPod Configuration (when INFERENCE_MODE='runpod')
    RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY', None)
    RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID', None)
    
    # Hugging Face Configuration (when INFERENCE_MODE='local')
    # Good model options:
    # - 'mistralai/Mistral-7B-Instruct-v0.2' (recommended, good quality)
    # - 'meta-llama/Llama-2-7b-chat-hf' (requires HF token approval)
    # - 'microsoft/phi-2' (smaller, faster)
    # - 'google/flan-t5-xl' (good for structured output)
    # - 'meta-llama/Llama-3.1-8B-Instruct' (very powerful, but slower)
    HF_MODEL = os.environ.get('HF_MODEL', 'meta-llama/Llama-3.1-8B')
    HF_TOKEN = os.environ.get('HF_TOKEN', None)  # Required for gated models like Llama
    HF_CACHE_DIR = os.environ.get('HF_CACHE_DIR', '/n/data1/hms/dbmi/manrai/aashna/HealthCoachV2/cache/huggingface')  # Optional: specify cache location
    
    # Model parameters
    MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 1024))
    TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.1))
    DEVICE = os.environ.get('DEVICE', 'auto')  # 'auto', 'cuda', 'cpu'
    
    # RAG Configuration
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 512))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 128))
    RETRIEVAL_K = int(os.environ.get('RETRIEVAL_K', 4))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


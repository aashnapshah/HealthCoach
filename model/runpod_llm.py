"""
Simple RunPod LLM Wrapper
Makes API calls to RunPod instead of loading models locally
"""
import requests
import os
from typing import Optional

class RunPodLLM:
    """Wrapper that mimics LangChain LLM interface but calls RunPod API"""
    
    def __init__(self, 
                 endpoint_id: str = None,
                 api_key: str = None,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1):
        """
        Initialize RunPod LLM client
        
        Args:
            endpoint_id: Your RunPod endpoint ID
            api_key: Your RunPod API key
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        self.endpoint_id = endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID')
        self.api_key = api_key or os.environ.get('RUNPOD_API_KEY')
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        if not self.endpoint_id or not self.api_key:
            raise ValueError("Must provide RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY")
        
        self.url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
    
    def invoke(self, prompt: str) -> str:
        """
        Send prompt to RunPod and get response
        Compatible with LangChain's invoke interface
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "prompt": prompt,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the generated text from RunPod response
            if "output" in result:
                return result["output"].get("response", "")
            elif "error" in result:
                raise Exception(f"RunPod error: {result['error']}")
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to call RunPod API: {e}")
    
    def __call__(self, prompt: str) -> str:
        """Allow direct calling of the instance"""
        return self.invoke(prompt)


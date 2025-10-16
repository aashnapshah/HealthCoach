"""
RunPod Handler for HealthCoach Model
This file handles inference requests from RunPod
"""
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """
    Load the fine-tuned model and tokenizer
    This runs once when the serverless function starts
    """
    global model, tokenizer
    
    # Get model path from environment or use default
    model_name = os.environ.get("MODEL_NAME", "microsoft/phi-2")  # Default to phi-2 or your model
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("Model loaded successfully!")

def generate_response(message, context=None, max_new_tokens=512, temperature=0.7):
    """
    Generate a response given a message and optional context
    """
    global model, tokenizer
    
    # Build prompt with context if provided
    if context and context.get('patient_info'):
        prompt = f"""You are a helpful healthcare AI assistant. You have access to the following patient information:

{context['patient_info']}

Patient: {message}

HealthCoach AI:"""
    else:
        prompt = f"""You are a helpful healthcare AI assistant.

Patient: {message}

HealthCoach AI:"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove prompt)
    response = response.split("HealthCoach AI:")[-1].strip()
    
    return response

def handler(event):
    """
    RunPod serverless handler function
    This is called for each inference request
    """
    try:
        # Extract input from event
        job_input = event.get("input", {})
        
        # Support both 'prompt' (for LangChain) and 'message' (for custom)
        prompt = job_input.get("prompt", "")
        message = job_input.get("message", "")
        text = prompt or message
        
        context = job_input.get("context", {})
        max_new_tokens = job_input.get("max_new_tokens", 1024)
        temperature = job_input.get("temperature", 0.1)
        
        if not text:
            return {"error": "No prompt or message provided"}
        
        # For simple prompts (from LangChain), use direct generation
        if prompt and not context:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.1,
                    do_sample=True if temperature > 0 else False,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        else:
            # For messages with context, use the existing function
            response = generate_response(
                message=text,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        
        return {
            "response": response,
            "context": context  # Return context for maintaining conversation state
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Load model on startup
    load_model()
    
    # Start the RunPod serverless handler
    runpod.serverless.start({"handler": handler})


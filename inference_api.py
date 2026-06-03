"""
FastAPI inference server for fine-tuned LLM
Deploy your model and serve predictions via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fine-Tuned LLM API",
    description="REST API for fine-tuned language model inference",
    version="1.0.0"
)

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class PredictionResponse(BaseModel):
    input: str
    output: str
    generation_time: float
    tokens_generated: int

# Global variables for model
model = None
tokenizer = None
device = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer, device
    
    logger.info("Loading fine-tuned model...")
    
    try:
        model_path = "./my_finetuned_model"  # Update with your model path
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model = model.to(device)
        
        logger.info("✓ Model loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fine-Tuned LLM API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "ready" if model else "loading",
        "device": device,
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate text based on input prompt
    
    Args:
        text: Input prompt
        max_length: Maximum length of generated text
        temperature: Controls randomness (0.0 = deterministic, 1.0 = random)
        top_p: Nucleus sampling parameter
    
    Returns:
        PredictionResponse with generated text and metadata
    """
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer.encode(request.text, return_tensors="pt").to(device)
        input_length = inputs.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - input_length
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
        
        return PredictionResponse(
            input=request.text,
            output=generated_text,
            generation_time=generation_time,
            tokens_generated=tokens_generated
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(requests: list[PredictionRequest]):
    """
    Generate text for multiple prompts (batch inference)
    """
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            # Call predict endpoint for each
            result = await predict(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction failed for: {req.text}")
            results.append({
                "input": req.text,
                "error": str(e)
            })
    
    return results

@app.get("/model-info")
async def model_info():
    """Get model information"""
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "vocab_size": tokenizer.vocab_size,
        "model_type": model.config.model_type if hasattr(model, 'config') else "unknown"
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {
        "error": "Invalid input",
        "detail": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("""
    🚀 Fine-Tuned LLM API Server
    ============================
    Starting FastAPI server...
    
    Endpoints:
    - GET  / (health check)
    - POST /predict (single inference)
    - POST /batch-predict (batch inference)
    - GET  /model-info (model details)
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

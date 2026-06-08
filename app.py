from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(
    title="Dialogue Summarizer API",
    description="Summarize customer-agent conversations using FLAN-T5-base",
    version="2.0"
)

print("🚀 Loading FLAN-T5-base model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
device = torch.device("cpu")
model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

class SummarizeRequest(BaseModel):
    dialogue: str
    strategy: str = "few_shot"

@app.post("/summarize")
def summarize_dialogue(request: SummarizeRequest):
    """
    Summarize a customer-agent dialogue
    
    Strategies:
    - zero_shot: No examples
    - few_shot: With examples
    """
    try:
        if not request.dialogue or len(request.dialogue) < 10:
            return {
                "error": "Dialogue too short (minimum 10 characters)",
                "status": 400
            }
        
        # Create prompt based on strategy
        if request.strategy == "few_shot":
            prompt = f"""Summarize this customer-agent dialogue in 1-2 sentences.
Include: (1) customer's main issue, (2) agent's action/response.

Example 1:
Dialogue: Customer: I want to cancel. Agent: Can I help first?
Summary: Customer wants to cancel. Agent offered assistance.

Example 2:
Dialogue: Customer: Billing issue. Agent: I'll fix it.
Summary: Customer reported billing problem. Agent committed to resolution.

NOW SUMMARIZE THIS:
Dialogue: {request.dialogue}
Summary:"""
        
        else:  # zero_shot
            prompt = f"""Summarize this dialogue briefly (1-2 sentences):
{request.dialogue}
Summary:"""
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                num_beams=2,
                temperature=0.7,
                early_stopping=True
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "summary": summary.strip(),
            "strategy": request.strategy,
            "status": 200,
            "model": "google/flan-t5-base",
            "dialogue_length": len(request.dialogue.split())
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": 500
        }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "google/flan-t5-base (local)",
        "version": "2.0",
        "ready": True
    }

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "name": "Dialogue Summarizer API",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "summarize": "/summarize (POST)",
            "docs": "/docs"
        },
        "model": "google/flan-t5-base",
        "deployed_on": "Hugging Face Spaces"
    }

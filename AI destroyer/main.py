"""
AI Chatbot Backend — FastAPI + Hugging Face Transformers
Model: GPT-2 by default (swap for any HF model e.g. mistralai/Mistral-7B-Instruct-v0.2)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model config ──────────────────────────────────────────────────────────────
# Swap MODEL_NAME for a more powerful model when ready:
#   "mistralai/Mistral-7B-Instruct-v0.2"   (7B, needs ~16GB RAM / GPU)
#   "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  (1.1B, runs on CPU)
#   "openai-community/gpt2"                (117M, quick demo)
MODEL_NAME = os.getenv("MODEL_NAME", "openai-community/gpt2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant. Answer questions clearly and concisely."
)

# ── Load model at startup ─────────────────────────────────────────────────────
generator = None

@app.on_event("startup")
async def load_model():
    global generator
    logger.info(f"Loading model: {MODEL_NAME}")
    device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    logger.info("Model loaded successfully.")

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE

class ChatResponse(BaseModel):
    reply: str
    model: str

# ── Helper: build prompt from history ─────────────────────────────────────────
def build_prompt(messages: List[Message]) -> str:
    """
    Formats conversation history into a prompt string.
    For instruction-tuned models (Mistral, TinyLlama, etc.) use their
    specific chat template instead — see HuggingFace model card.
    """
    prompt = f"System: {SYSTEM_PROMPT}\n\n"
    for msg in messages:
        if msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"
    return prompt

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    prompt = build_prompt(request.messages)

    try:
        outputs = generator(
            prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        # Extract only the new reply (after the prompt)
        reply = generated[len(prompt):].strip()
        # Cut off at next "User:" turn if model keeps going
        if "\nUser:" in reply:
            reply = reply.split("\nUser:")[0].strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(reply=reply, model=MODEL_NAME)
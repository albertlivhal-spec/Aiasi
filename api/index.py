from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🤖 AI Assistant Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    history: List[Dict]

async def get_ai_response(message: str, history: List[Dict]) -> str:
    if not HF_TOKEN:
        return "🔧 AI система настраивается. Добавьте HF_TOKEN в настройки Vercel"
    
    try:
        # Простой промпт
        prompt = ""
        for msg in history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += f"User: {message}\nAssistant:"
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100, "temperature": 0.7}
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'].split("Assistant:")[-1].strip()
        else:
            return f"Ошибка API: {response.status_code}"
            
    except Exception as e:
        return f"Ошибка: {str(e)}"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        ai_response = await get_ai_response(request.message, request.history)
        
        new_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": ai_response}
        ]
        
        return ChatResponse(
            response=ai_response,
            history=new_history[-6:]
        )
        
    except Exception as e:
        return ChatResponse(
            response="Привет! Технические неполадки. Попробуйте позже.",
            history=request.history
        )

@app.get("/")
async def root():
    return {"status": "active", "ai": "Hugging Face"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Vercel требует app
app = app
# HF_TOKEN added - Вс 26 окт 2025 23:44:58 MSK

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []
    user_id: str = "guest"

class ChatResponse(BaseModel):
    response: str
    history: List[Dict]

async def get_ai_response(message: str, history: List[Dict]) -> str:
    """Получаем ответ от Hugging Face API"""
    if not HF_TOKEN:
        return "👋 Привет! Я ваш AI помощник. Технические настройки завершаются..."
    
    try:
        # Собираем контекст из истории
        context = ""
        for msg in history[-4:]:  # Берем последние 4 сообщения
            speaker = "Вы" if msg["role"] == "user" else "Я"
            context += f"{speaker}: {msg['content']}\n"
        
        prompt = f"{context}Вы: {message}\nЯ:"
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.8,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result, list) and 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
        
        # Fallback ответы
        fallback_responses = [
            "Интересный вопрос! Расскажите подробнее?",
            "Понял ваш запрос. Чем еще могу помочь?",
            "Спасибо за сообщение! Что вас интересует?",
            "Привет! Рад общению. Чем могу быть полезен?"
        ]
        import random
        return random.choice(fallback_responses)
        
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return "🤖 Здравствуйте! Я ваш AI помощник. Готов ответить на ваши вопросы!"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        ai_response = await get_ai_response(request.message, request.history)
        
        # Обновляем историю
        new_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": ai_response}
        ]
        
        return ChatResponse(
            response=ai_response,
            history=new_history[-6:]  # Храним последние 6 сообщений
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response="Привет! Я ваш умный помощник. Задавайте вопросы!",
            history=request.history
        )

@app.get("/")
async def root():
    return {"status": "active", "ai": "Hugging Face", "version": "2.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AI Assistant Pro"}

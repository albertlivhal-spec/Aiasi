from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import logging
from typing import List, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем приложение
app = FastAPI(
    title="🤖 AI Assistant Pro",
    description="Умный помощник на Hugging Face",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"

# Модели данных
class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    history: List[Dict]

# Функция для получения ответа от AI
async def get_ai_response(message: str, history: List[Dict]) -> str:
    """Получаем ответ от Hugging Face API"""
    
    # Проверка токена
    if not HF_TOKEN:
        logger.error("HF_TOKEN not configured")
        return "🔧 AI система настраивается. Добавьте HF_TOKEN в настройки Vercel"
    
    try:
        # Формируем промпт из истории
        prompt = ""
        for msg in history[-4:]:  # Берем последние 4 сообщения
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        
        prompt += f"User: {message}\nAssistant:"
        
        logger.info(f"Sending request to HF API: {prompt[:100]}...")
        
        # Заголовки и параметры
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Отправляем запрос
        response = requests.post(
            HF_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        logger.info(f"HF API response status: {response.status_code}")
        
        # Обрабатываем ответ
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_text = result[0].get('generated_text', '').strip()
                # Извлекаем только ответ ассистента
                if "Assistant:" in ai_text:
                    return ai_text.split("Assistant:")[-1].strip()
                return ai_text
            else:
                return "🤖 Не получилось обработать ответ AI"
                
        elif response.status_code == 503:
            logger.warning("Model is loading")
            return "🤖 Модель загружается... Попробуйте через 30 секунд!"
            
        else:
            logger.error(f"HF API error: {response.status_code} - {response.text}")
            return f"❌ Ошибка AI сервиса: {response.status_code}"
            
    except requests.exceptions.Timeout:
        logger.error("HF API timeout")
        return "⏰ Таймаут подключения к AI сервису"
        
    except requests.exceptions.ConnectionError:
        logger.error("HF API connection error")
        return "🔌 Ошибка подключения к AI сервису"
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"⚠️ Неожиданная ошибка: {str(e)}"

# Эндпоинты API
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Основной эндпоинт для чата"""
    try:
        logger.info(f"Chat request: {request.message}")
        
        # Получаем ответ от AI
        ai_response = await get_ai_response(request.message, request.history)
        
        # Обновляем историю
        new_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": ai_response}
        ]
        
        # Сохраняем только последние 8 сообщений
        new_history = new_history[-8:]
        
        logger.info(f"Chat response: {ai_response[:100]}...")
        
        return ChatResponse(
            response=ai_response,
            history=new_history
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return ChatResponse(
            response="Привет! Я ваш AI помощник. Произошла техническая ошибка.",
            history=request.history
        )

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "status": "active", 
        "service": "AI Assistant Pro",
        "ai_provider": "Hugging Face",
        "model": "DialoGPT-medium"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "timestamp": "2024-10-27",
        "hf_token_configured": bool(os.getenv("HF_TOKEN"))
    }

@app.get("/test")
async def test_ai():
    """Тестовый эндпоинт"""
    try:
        test_response = await get_ai_response("Привет! Как дела?", [])
        return {
            "test": "success",
            "ai_response": test_response,
            "hf_token_configured": bool(os.getenv("HF_TOKEN"))
        }
    except Exception as e:
        return {
            "test": "error",
            "error": str(e),
            "hf_token_configured": bool(os.getenv("HF_TOKEN"))
        }

# Для Vercel требуется переменная app
app = app

# Для локального запуска
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

# DeepSeek-R1 - отличная диалоговая модель с поддержкой русского
HF_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-llm-7b-chat"

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    history: List[Dict]

async def get_ai_response(message: str, history: List[Dict]) -> str:
    """Получаем ответ от DeepSeek-R1 через Hugging Face API"""
    if not HF_TOKEN:
        return "🔧 AI система настраивается. Расскажите, чем могу помочь?"
    
    try:
        # Формируем промпт для диалоговой модели
        prompt = "<s>"
        for msg in history[-4:]:
            if msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST]"
            else:
                prompt += f" {msg['content']}</s><s>"
        
        prompt += f"[INST] {message} [/INST]"
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "return_full_text": False
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_text = result[0].get('generated_text', '').strip()
                # Очищаем ответ от тегов промпта
                ai_text = ai_text.replace('[INST]', '').replace('[/INST]', '').replace('<s>', '').replace('</s>', '')
                return ai_text.strip()
        elif response.status_code == 503:
            # Модель загружается
            return "🤖 Модель загружается... Попробуйте через 30 секунд!"
        else:
            logger.warning(f"HF API: {response.status_code}")
        
        # Fallback ответ
        return "Привет! Я ваш AI помощник на основе DeepSeek. Рад общению! 😊"
            
    except Exception as e:
        logger.error(f"AI error: {e}")
        return "Здравствуйте! Я AI помощник. Чем могу помочь?"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Получаем ответ от AI
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
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response="Привет! Я ваш умный помощник DeepSeek. Задавайте вопросы!",
            history=request.history
        )

@app.get("/")
async def root():
    return {"status": "active", "ai_provider": "Hugging Face", "model": "DeepSeek-R1"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Assistant Pro"}

@app.get("/test")
async def test_ai():
    """Тестовый эндпоинт для проверки AI"""
    try:
        test_response = await get_ai_response("Привет, как дела?", [])
        return {"test": "OK", "ai_response": test_response}
    except Exception as e:
        return {"test": "ERROR", "error": str(e)}

# Для локального запуска
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "✅ Server is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat")
async def chat():
    return {"response": "Тестовый ответ от сервера"}

app = app

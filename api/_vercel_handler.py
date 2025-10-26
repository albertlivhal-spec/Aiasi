import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(__file__))

try:
    from chat import app
except ImportError:
    # Альтернативный способ импорта
    import importlib.util
    spec = importlib.util.spec_from_file_location("chat", os.path.join(os.path.dirname(__file__), "chat.py"))
    chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat)
    app = chat.app

# Vercel требует именно "app"
app = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

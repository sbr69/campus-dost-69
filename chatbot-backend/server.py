"""
SC-CSE Chatbot Server Entry Point.

This module serves as the entry point for the chatbot server.
All application logic is organized in the `app` package.
"""
from app.main import app

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    
    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )

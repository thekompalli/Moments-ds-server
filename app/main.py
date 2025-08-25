# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.speech.endpoints import router as speech_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="MOMents Data Science Server",
    description="API for MOMents app",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(speech_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to MOMents Data Science API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
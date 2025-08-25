# app/api/speech/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class SpeechProcessRequest(BaseModel):
    user_id: str
    trimester: int = Field(..., ge=1, le=3)
    pregnancy_week: int = Field(..., ge=1, le=42)
    current_date: Optional[str] = None

    class Config:
        json_schema_extra = {  
            "example": {
                "user_id": "user-123",
                "trimester": 2,
                "pregnancy_week": 20,
                "current_date": "2025-03-27"
            }
        }

class TextParseRequest(BaseModel):
    user_id: str
    trimester: int = Field(..., ge=1, le=3)
    pregnancy_week: int = Field(..., ge=1, le=42)
    current_date: Optional[str] = None
    text_input: str

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "trimester": 2,
                "pregnancy_week": 20,
                "current_date": "2025-03-27",
                "text_input": "I have an appointment at 9 a.m. and need to take medicine at 2 p.m."
            }
        }

class TaskEntity(BaseModel):
    task_title: str
    description: Optional[str] = None
    due_date: Optional[str] = None
    due_time: Optional[str] = None
    category: str  # must-do, optional, self-care
    energy_level: str  # high, medium, low
    estimated_minutes: int
    confidence: float

class SpeechProcessResponse(BaseModel):
    transcription: str
    detected_tasks: List[TaskEntity]
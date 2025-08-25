# app/api/speech/endpoints.py (Updated)
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from datetime import datetime
from typing import List, Optional
import httpx

from app.api.speech.schemas import SpeechProcessRequest, SpeechProcessResponse, TaskEntity, TextParseRequest
from app.services.eden_ai_service import EdenAIService
from app.core.nlp.task_splitter import TaskSplitter
from app.core.nlp.entity_extractor import EntityExtractor
from app.core.categorization.classifier import TaskCategorizer
from app.core.energy.estimator import EnergyEstimator
from app.services.prompt_service import get_prompt_service

from app.config import get_settings

from pydantic import BaseModel

router = APIRouter(prefix="/speech", tags=["Speech Processing"])
logger = logging.getLogger(__name__)



class TextRequest(BaseModel):
    text: str

# Simple dependency injection for services
def get_prompt_service_dependency():
    return get_prompt_service()

def get_eden_ai_service():
    return EdenAIService()

def get_task_splitter():
    return TaskSplitter()

def get_entity_extractor():
    return EntityExtractor()

def get_task_categorizer():
    return TaskCategorizer()

def get_energy_estimator():
    return EnergyEstimator()

@router.post("/process", response_model=SpeechProcessResponse)
async def process_speech(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    trimester: int = Form(...),
    pregnancy_week: int = Form(...),
    current_date: Optional[str] = Form(None),
    eden_ai_service: EdenAIService = Depends(get_eden_ai_service),
    task_splitter: TaskSplitter = Depends(get_task_splitter),
    entity_extractor: EntityExtractor = Depends(get_entity_extractor),
    task_categorizer: TaskCategorizer = Depends(get_task_categorizer),
    energy_estimator: EnergyEstimator = Depends(get_energy_estimator)
):
    """
    Process speech audio to extract tasks using AI.
    
    This endpoint:
    1. Converts audio to text using Eden AI's speech-to-text
    2. Splits text into individual tasks using Eden AI's LLM
    3. Extracts entities from each task
    4. Categorizes each task
    5. Estimates energy level for each task
    """
    try:
        # Read the audio file content
        file_content = await file.read()
        
        # Process the audio with Eden AI
        transcription = await eden_ai_service.transcribe_audio(file_content)
        
        if not transcription or transcription.strip() == "":
            raise HTTPException(status_code=400, detail="No speech detected in the audio")
        
        # Set the current date if not provided
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Split transcription into individual tasks
        task_texts = await task_splitter.split_text_into_tasks(transcription)

        # Log the tasks found for debugging
        logger.info(f"Found {len(task_texts)} tasks: {task_texts}")

        if not task_texts:
            logger.info(f"No tasks detected in transcription: {transcription}")
            return SpeechProcessResponse(
                transcription=transcription,
                detected_tasks=[]
            )
        
        # Process each task
        detected_tasks = []
        for task_text in task_texts:
            try:
                # Extract entities
                entities = await entity_extractor.extract_entities(task_text, current_date)
                logger.info(f"Extracted entities: {entities}")
                
                # Create a base task with extracted information
                task_title = entities.get("task_title", task_text[:50])
                
                # Default values in case subsequent steps fail
                category = "must-do"
                category_confidence = 0.8
                energy_level = "medium" 
                energy_confidence = 0.8
                estimated_minutes = 30
                
                try:
                    # Categorize the task
                    category_result = await task_categorizer.categorize_task(
                        task_title=task_title,
                        task_description=entities.get("description", ""),
                        trimester=trimester,
                        pregnancy_week=pregnancy_week
                    )
                    logger.info(f"Category result: {category_result}")
                    
                    # Extract category values safely
                    if category_result and isinstance(category_result, dict):
                        category = category_result.get("category", category)
                        category_confidence = category_result.get("confidence", category_confidence)
                except Exception as cat_e:
                    logger.error(f"Error in task categorization: {str(cat_e)}, using defaults")
                
                try:
                    # Estimate energy level
                    energy_result = await energy_estimator.estimate_energy(
                        task_title=task_title,
                        task_description=entities.get("description", ""),
                        estimated_minutes=estimated_minutes,
                        trimester=trimester
                    )
                    logger.info(f"Energy result: {energy_result}")
                    
                    # Extract energy values safely
                    if energy_result and isinstance(energy_result, dict):
                        energy_level = energy_result.get("energy_level", energy_level)
                        energy_confidence = energy_result.get("confidence", energy_confidence)
                        estimated_minutes = energy_result.get("estimated_minutes", estimated_minutes)
                        
                        # Special case handling for tablet-taking
                        if "take tablet" in task_title.lower() or "take medicine" in task_title.lower():
                            if estimated_minutes > 10:
                                estimated_minutes = 5
                                logger.info(f"Adjusted time for taking tablets to {estimated_minutes} minutes")
                except Exception as energy_e:
                    logger.error(f"Error in energy estimation: {str(energy_e)}, using defaults")
                
                # Create the task entity
                task = TaskEntity(
                    task_title=task_title,
                    description=entities.get("description"),
                    due_date=entities.get("due_date"),
                    due_time=entities.get("due_time"),
                    category=category,
                    energy_level=energy_level,
                    estimated_minutes=estimated_minutes,
                    confidence=min(category_confidence, energy_confidence)
                )
                
                detected_tasks.append(task)
            except Exception as task_e:
                logger.error(f"Error processing task '{task_text}': {str(task_e)}")
                # Continue with next task - don't let one bad task fail everything
        
        # Log the processing (in background to not block response)
        background_tasks.add_task(
            logger.info,
            f"Processed speech for user {user_id}: {len(detected_tasks)} tasks detected from '{transcription}'"
        )
        
        return SpeechProcessResponse(
            transcription=transcription,
            detected_tasks=detected_tasks
        )
    
    except Exception as e:
        logger.error(f"Error processing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing speech: {str(e)}")

@router.post("/parse_text", response_model=SpeechProcessResponse)
async def parse_text(
    request: TextParseRequest,
    task_splitter: TaskSplitter = Depends(get_task_splitter),
    entity_extractor: EntityExtractor = Depends(get_entity_extractor),
    task_categorizer: TaskCategorizer = Depends(get_task_categorizer),
    energy_estimator: EnergyEstimator = Depends(get_energy_estimator)
):
    """
    Parse text input to extract tasks using LLM approach.
    """
    try:
        # Set the current date if not provided
        current_date = request.current_date
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Extract tasks using OpenAI
        task_texts = await task_splitter.split_text_into_tasks(request.text_input)
        logger.info(f"Extracted tasks: {task_texts}")
        
        # Process each task
        detected_tasks = []
        for task_text in task_texts:
            # Extract entities
            entities = await entity_extractor.extract_entities(task_text, current_date)
            logger.info(f"Extracted entities: {entities}")
            
            # Create a base task with extracted information
            task_title = entities.get("task_title", task_text[:50])
            
            # Categorize the task
            category_result = await task_categorizer.categorize_task(
                task_title=task_title,
                task_description=entities.get("description", ""),
                trimester=request.trimester,
                pregnancy_week=request.pregnancy_week
            )
            logger.info(f"Category result: {category_result}")
            
            # Estimate energy level
            energy_result = await energy_estimator.estimate_energy(
                task_title=task_title,
                task_description=entities.get("description", ""),
                estimated_minutes=30,  # Default, will be updated by the LLM
                trimester=request.trimester
            )
            logger.info(f"Energy result: {energy_result}")
            
            # Use the LLM's time estimate if available
            estimated_minutes = energy_result.get("estimated_minutes", 30)
            
            # Create the task entity
            task = TaskEntity(
                task_title=task_title,
                description=entities.get("description"),
                due_date=entities.get("due_date"),
                due_time=entities.get("due_time"),
                category=category_result["category"],
                energy_level=energy_result["energy_level"],
                estimated_minutes=estimated_minutes,
                confidence=min(category_result.get("confidence", 0.8), energy_result.get("confidence", 0.8))
            )
            
            detected_tasks.append(task)
        
        return SpeechProcessResponse(
            transcription=request.text_input,
            detected_tasks=detected_tasks
        )
    
    except Exception as e:
        logger.error(f"Error in text parsing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing text: {str(e)}")
    
@router.post("/debug_tasks")
async def debug_task_splitting(request: TextRequest):
    """
    Debug endpoint to analyze task extraction from a text.
    """
    text = request.text
    
    # Initialize a fresh new instance of the TaskSplitter
    task_splitter = TaskSplitter()
    
    # Try both extraction methods directly
    try:
        # Add an explicit model parameter - try with OpenAI model
        headers = {
            "Authorization": f"Bearer {task_splitter.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Analyze this text and identify all the tasks or appointments mentioned:

"{text}"

Format your response as a JSON array of task strings:
[
  "Appointment at 9 a.m. in the morning",
  "Take tablets around 2 p.m.",
  "Go to yoga at 6 p.m. for one hour",
  "Do homework before 10 p.m."
]
"""
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            payload = {
                "model": "openai/gpt-4o", # Try with OpenAI instead of Gemini
                "temperature": 0.1,
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            response = await client.post(task_splitter.llm_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                return {"error": f"LLM API error: {response.text}"}
            
            result = response.json()
            assistant_message = result.get('generated_text', '')
            
            # Try to parse tasks from the response
            primary_tasks = task_splitter._parse_tasks_from_response(assistant_message)
            
            # Return the results
            return {
                "direct_prompt": prompt,
                "openai_response": assistant_message,
                "extracted_tasks": primary_tasks,
                "text": text
            }
            
    except Exception as e:
        logger.error(f"Error in direct extraction: {str(e)}")
        return {"error": str(e)}
    

# Add this to app/api/speech/endpoints.py

@router.get("/test_eden_ai")
async def test_eden_ai():
    """
    Test if the Eden AI API is working correctly.
    """
    try:
        # Get settings to access the API key
        settings = get_settings()
        api_key = settings.eden_ai_api_key
        
        # Check if API key exists
        if not api_key:
            return {"status": "error", "message": "Eden AI API key not configured"}
        
        # Prepare a simple request to the Eden AI LLM API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try both Gemini and OpenAI models
        results = {}
        
        # Test with Gemini
        async with httpx.AsyncClient(timeout=10.0) as client:
            gemini_payload = {
                "model": "google/gemini-2.5-pro-exp-03-25",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Say hello world"
                            }
                        ]
                    }
                ]
            }
            
            gemini_url = "https://api.edenai.run/v2/llm/chat"
            gemini_response = await client.post(gemini_url, json=gemini_payload, headers=headers)
            
            results["gemini"] = {
                "status_code": gemini_response.status_code,
                "response": gemini_response.text if gemini_response.status_code != 200 else gemini_response.json()
            }
        
        # Test with OpenAI
        async with httpx.AsyncClient(timeout=10.0) as client:
            openai_payload = {
                "model": "openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Say hello world"
                            }
                        ]
                    }
                ]
            }
            
            openai_url = "https://api.edenai.run/v2/llm/chat"
            openai_response = await client.post(openai_url, json=openai_payload, headers=headers)
            
            results["openai"] = {
                "status_code": openai_response.status_code,
                "response": openai_response.text if openai_response.status_code != 200 else openai_response.json()
            }
        
        # Test the Speech-to-Text API
        async with httpx.AsyncClient(timeout=10.0) as client:
            # We can't actually test this without an audio file, so we'll just check
            # if the API documentation endpoint works
            stt_url = "https://api.edenai.run/v2/info/providers"
            stt_response = await client.get(stt_url, headers=headers)
            
            results["api_info"] = {
                "status_code": stt_response.status_code,
                "response": "Success" if stt_response.status_code == 200 else stt_response.text
            }
        
        return {
            "status": "success", 
            "message": "API Test Results",
            "api_key_provided": bool(api_key),
            "api_key_first_chars": api_key[:6] + "..." if api_key else None,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error testing Eden AI API: {str(e)}")
        return {
            "status": "error", 
            "message": f"Exception occurred: {str(e)}"
        }
# app/services/eden_ai_service.py (updated version)
import httpx
import asyncio  # Add this import
import logging
import time
import json
from fastapi import HTTPException
from app.config import get_settings

logger = logging.getLogger(__name__)

class EdenAIService:
    """Service for interacting with Eden AI APIs"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.eden_ai_api_key
        self.speech_to_text_url = self.settings.speech_to_text_url
        self.result_url_template = self.settings.speech_to_text_result_url
        self.default_providers = self.settings.default_providers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def transcribe_audio(self, file_content: bytes, language: str = "en", providers: list = None):
        """
        Transcribe audio file using Eden AI's speech-to-text API
        
        Args:
            file_content: Binary content of the audio file
            language: Language code (default: en)
            providers: List of providers to use (default: ["google"])
            
        Returns:
            Transcribed text
        """
        if not providers:
            providers = ["openai"]
        
        # Prepare the providers string
        providers_str = ",".join(providers)
        
        # Prepare files and data for multipart upload
        files = {
            "file": file_content,
        }
        
        data = {
            "providers": providers_str,
            "language": language,
            "show_original_response": False,
            "speakers": 2,
            "profanity_filter": False,
            "convert_to_wav": True  # For better compatibility
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Submit the job
                logger.info(f"Submitting audio transcription job to Eden AI with providers: {providers_str}")
                response = await client.post(
                    self.speech_to_text_url, 
                    files=files,
                    data=data,
                    headers=self.headers
                )
                
                # Check for error responses
                if response.status_code >= 400:
                    logger.error(f"Failed to start transcription job: {response.text}")
                    raise HTTPException(status_code=500, detail=f"Eden AI API error: {response.text}")
                
                # Get the job ID
                job_data = response.json()
                public_id = job_data.get('public_id')
                
                if not public_id:
                    raise HTTPException(status_code=500, detail="No job ID returned from Eden AI")
                
                logger.info(f"Transcription job started with ID: {public_id}")
                
                # Poll for the result
                result_url = self.result_url_template.format(public_id=public_id)
                
                # Poll for the job result (with timeout after 60 seconds)
                max_attempts = 30
                for attempt in range(max_attempts):
                    await asyncio.sleep(2)  # Use asyncio.sleep instead of httpx.AsyncClient().sleep
                    
                    result_response = await client.get(result_url, headers=self.headers)
                    
                    if result_response.status_code != 200:
                        logger.warning(f"Polling attempt {attempt+1}: Status {result_response.status_code}")
                        continue
                    
                    result_data = result_response.json()
                    status = result_data.get('status')
                    
                    if status == 'finished':
                        # Extract the transcription from OpenAI instead of Google
                        openai_result = result_data.get('results', {}).get('openai', {})
                        text = openai_result.get('text', '')
                        return text
                    
                    if status == 'failed':
                        error_msg = result_data.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"Transcription job failed: {error_msg}")
                        raise HTTPException(status_code=500, detail=f"Transcription failed: {error_msg}")
                    
                    # If still processing, log status and continue polling
                    logger.info(f"Job {public_id} status: {status} (attempt {attempt+1})")
                
                raise HTTPException(status_code=500, detail="Transcription timed out after 60 seconds")
                
        except Exception as e:
            logger.error(f"Error communicating with Eden AI: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error communicating with Eden AI: {str(e)}")
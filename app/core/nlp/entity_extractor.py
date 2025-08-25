# app/core/nlp/entity_extractor.py
import httpx
import logging
import json
from typing import Dict, Any
from fastapi import HTTPException

from app.config import get_settings
from app.services.prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts entities from task text using pure LLM approach.
    """
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.eden_ai_api_key
        self.llm_url = "https://api.edenai.run/v2/llm/chat"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "openai/gpt-4o"
        self.prompt_service = get_prompt_service()
    
    async def extract_entities(self, task_text: str, current_date: str) -> Dict[str, Any]:
        """
        Extract entities from task text using LLM.
        
        Args:
            task_text: The task text to analyze
            current_date: Current date for relative date resolution
            
        Returns:
            Dictionary of extracted entities
        """
        try:
            prompt = self.prompt_service.format_prompt(
                'entity_extractor', 
                'entity_extraction_prompt', 
                task_text=task_text, 
                current_date=current_date
            )
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                payload = {
                    "model": self.model,
                    "temperature": 0.1,
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
                
                response = await client.post(self.llm_url, json=payload, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error(f"LLM API error: {response.text}")
                    return self._create_default_entities(task_text)
                
                result = response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                logger.info(f"LLM entity extraction response: {assistant_message}")
                
                # Parse entities from response
                try:
                    # Look for JSON object in response
                    start_idx = assistant_message.find('{')
                    end_idx = assistant_message.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = assistant_message[start_idx:end_idx]
                        extracted_entities = json.loads(json_str)
                        
                        # Validate entities
                        entities = {
                            "task_title": extracted_entities.get("task_title", task_text[:50]),
                            "description": extracted_entities.get("description"),
                            "due_date": extracted_entities.get("due_date"),
                            "due_time": extracted_entities.get("due_time")
                        }
                        
                        # Log the extracted entities
                        logger.info(f"Extracted entities: {entities}")
                        return entities
                    
                    # Fallback to default if JSON parsing fails
                    return self._create_default_entities(task_text)
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {assistant_message}")
                    return self._create_default_entities(task_text)
                
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return self._create_default_entities(task_text)
    
    def _create_default_entities(self, task_text: str) -> Dict[str, Any]:
        """Create default entities when extraction fails."""
        return {
            "task_title": task_text[:50],
            "description": None,
            "due_date": None,
            "due_time": None
        }
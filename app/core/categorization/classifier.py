# app/core/categorization/classifier.py
import httpx
import logging
import json
from typing import Dict, Any
from fastapi import HTTPException

from app.config import get_settings
from app.services.prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

class TaskCategorizer:
    """
    Categorizes tasks using pure LLM approach.
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
    
    async def categorize_task(self, 
                              task_title: str, 
                              task_description: str, 
                              trimester: int, 
                              pregnancy_week: int) -> Dict[str, Any]:
        """
        Categorize a task using LLM.
        
        Args:
            task_title: The title of the task
            task_description: Description or additional details
            trimester: Current pregnancy trimester (1-3)
            pregnancy_week: Current pregnancy week (1-42)
            
        Returns:
            Dictionary with category and confidence
        """
        try:
            # If task_description is None, use a default value for formatting
            task_description = task_description if task_description else 'None provided'
            
            prompt = self.prompt_service.format_prompt(
                'task_categorizer', 
                'categorization_prompt',
                task_title=task_title,
                task_description=task_description,
                trimester=trimester,
                pregnancy_week=pregnancy_week
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
                    return {"category": "must-do", "confidence": 0.8}
                
                result = response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                logger.info(f"LLM categorization response: {assistant_message}")
                
                # Parse the JSON response
                try:
                    # Look for JSON object in the response
                    start_idx = assistant_message.find('{')
                    end_idx = assistant_message.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = assistant_message[start_idx:end_idx]
                        category_info = json.loads(json_str)
                        
                        # Verify the category is one of the expected values
                        category = category_info.get("category", "").lower()
                        if category not in ["must-do", "self-care", "optional"]:
                            category = "must-do"  # Default to must-do if unexpected category
                        
                        confidence = category_info.get("confidence", 0.8)
                        
                        # Ensure confidence is a float between 0 and 1
                        try:
                            confidence = float(confidence)
                            confidence = max(0.0, min(1.0, confidence))
                        except:
                            confidence = 0.8
                        
                        result = {
                            "category": category,
                            "confidence": confidence
                        }
                        
                        # Add explanation if available
                        if "explanation" in category_info:
                            result["explanation"] = category_info["explanation"]
                            
                        return result
                    
                    # Fallback if JSON parsing fails
                    return {"category": "must-do", "confidence": 0.8}
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {assistant_message}")
                    return {"category": "must-do", "confidence": 0.8}
                
        except Exception as e:
            logger.error(f"Error in task categorization: {str(e)}")
            return {"category": "must-do", "confidence": 0.8}
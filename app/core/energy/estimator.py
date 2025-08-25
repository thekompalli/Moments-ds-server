# app/core/energy/estimator.py
import httpx
import logging
import json
from typing import Dict, Any
from fastapi import HTTPException

from app.config import get_settings
from app.services.prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

class EnergyEstimator:
    """
    Estimates energy level required for tasks using pure LLM approach.
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
    
    async def estimate_energy(self, 
                             task_title: str, 
                             task_description: str, 
                             estimated_minutes: int, 
                             trimester: int) -> Dict[str, Any]:
        """
        Estimate energy level required for a task using LLM.
        
        Args:
            task_title: The title of the task
            task_description: Description or additional details
            estimated_minutes: Estimated time in minutes
            trimester: Current pregnancy trimester (1-3)
            
        Returns:
            Dictionary with energy level and confidence
        """
        try:
            # If task_description is None, use a default value for formatting
            task_description = task_description if task_description else 'None provided'
            
            prompt = self.prompt_service.format_prompt(
                'energy_estimator', 
                'energy_estimation_prompt',
                task_title=task_title,
                task_description=task_description,
                estimated_minutes=estimated_minutes,
                trimester=trimester
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
                    return {"energy_level": "medium", "confidence": 0.8}
                
                result = response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                logger.info(f"LLM energy estimation response: {assistant_message}")
                
                # Parse the JSON response
                try:
                    # Look for JSON object in the response
                    start_idx = assistant_message.find('{')
                    end_idx = assistant_message.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = assistant_message[start_idx:end_idx]
                        energy_info = json.loads(json_str)
                        
                        # Verify the energy level is one of the expected values
                        energy_level = energy_info.get("energy_level", "").lower()
                        if energy_level not in ["high", "medium", "low"]:
                            energy_level = "medium"  # Default to medium if unexpected value
                        
                        confidence = energy_info.get("confidence", 0.8)
                        
                        # Ensure confidence is a float between 0 and 1
                        try:
                            confidence = float(confidence)
                            confidence = max(0.0, min(1.0, confidence))
                        except:
                            confidence = 0.8
                        
                        # Use the LLM's estimated minutes if available
                        try:
                            minutes = int(energy_info.get("estimated_minutes", estimated_minutes))
                            # Sanity check - ensure minutes is reasonable
                            if minutes <= 0 or minutes > 480:  # Max 8 hours
                                minutes = estimated_minutes
                        except:
                            minutes = estimated_minutes
                        
                        result = {
                            "energy_level": energy_level,
                            "confidence": confidence,
                            "estimated_minutes": minutes
                        }
                        
                        return result
                    
                    # Fallback if JSON parsing fails
                    return {"energy_level": "medium", "confidence": 0.8}
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {assistant_message}")
                    return {"energy_level": "medium", "confidence": 0.8}
                
        except Exception as e:
            logger.error(f"Error in energy estimation: {str(e)}")
            return {"energy_level": "medium", "confidence": 0.8}
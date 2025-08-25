# app/core/nlp/task_splitter.py
import httpx
import logging
import json
from typing import List
from fastapi import HTTPException

from app.config import get_settings
from app.services.prompt_service import get_prompt_service

logger = logging.getLogger(__name__)

class TaskSplitter:
    """
    Splits text input into individual tasks using Eden AI's LLM capabilities.
    """
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.eden_ai_api_key
        self.llm_url = "https://api.edenai.run/v2/llm/chat"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Using OpenAI model for more reliable formatting
        self.model = "openai/gpt-4o"
        self.prompt_service = get_prompt_service()
    
    async def split_text_into_tasks(self, text: str) -> List[str]:
        """
        Split a text into individual tasks using LLM.
        
        Args:
            text: Input text containing one or more tasks
            
        Returns:
            List of individual task strings
        """
        try:
            # First attempt - primary extraction
            tasks = await self._extract_tasks_primary(text)
            if tasks:
                logger.info(f"Primary extraction succeeded, found {len(tasks)} tasks")
                return tasks
            
            # If first attempt fails, check if text contains tasks
            contains_tasks = await self._does_text_contain_tasks(text)
            if not contains_tasks:
                logger.info(f"Text does not contain tasks: {text}")
                return []
            
            # If it contains tasks but primary extraction failed, try alternative approach
            logger.info("Primary extraction failed, trying alternative approach")
            tasks = await self._extract_tasks_alternative(text)
            if tasks:
                logger.info(f"Alternative extraction succeeded, found {len(tasks)} tasks")
                return tasks
            
            # If all LLM approaches fail, but we know there are tasks, return the whole text as one task
            logger.warning(f"All LLM extraction attempts failed for: {text}")
            return [text]
            
        except Exception as e:
            logger.error(f"Error in task splitting: {str(e)}")
            return []
    
    async def _extract_tasks_primary(self, text: str) -> List[str]:
        """Primary task extraction method using LLM."""
        prompt = self.prompt_service.format_prompt('task_splitter', 'primary_prompt', text=text)
        
        async with httpx.AsyncClient(timeout=20.0) as client:
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
            
            try:
                response = await client.post(self.llm_url, json=payload, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error(f"LLM API error: {response.text}")
                    return []
                
                result = response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Debug the response
                logger.info(f"LLM response: {assistant_message}")
                
                # Parse the response
                return self._parse_tasks_from_response(assistant_message)
                
            except Exception as e:
                logger.error(f"Error in primary task extraction: {str(e)}")
                return []
    
    async def _extract_tasks_alternative(self, text: str) -> List[str]:
        """Alternative task extraction method using LLM with simpler prompt."""
        prompt = self.prompt_service.format_prompt('task_splitter', 'alternative_prompt', text=text)
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            payload = {
                "model": self.model,
                "temperature": 0.2,
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
            
            try:
                response = await client.post(self.llm_url, json=payload, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error(f"LLM API error in alternative approach: {response.text}")
                    return []
                
                result = response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Debug the response
                logger.info(f"Alternative LLM response: {assistant_message}")
                
                # Parse the response - looking for a numbered list format
                return self._parse_tasks_from_numbered_list(assistant_message)
                
            except Exception as e:
                logger.error(f"Error in alternative task extraction: {str(e)}")
                return []
    
    async def _does_text_contain_tasks(self, text: str) -> bool:
        """Check if the text contains tasks using LLM."""
        prompt = self.prompt_service.format_prompt('task_splitter', 'task_detection_prompt', text=text)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
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
            
            try:
                response = await client.post(self.llm_url, json=payload, headers=self.headers)
                
                if response.status_code != 200:
                    return False
                
                result = response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Log the response
                logger.info(f"Task detection response: {assistant_message}")
                
                return "CONTAINS_TASKS" in assistant_message.upper()
                
            except Exception as e:
                logger.error(f"Error checking if text contains tasks: {str(e)}")
                return False
    
    def _parse_tasks_from_response(self, response_text: str) -> List[str]:
        """Parse tasks from LLM response, looking for JSON array."""
        try:
            # Look for a JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    tasks = json.loads(json_str)
                    if isinstance(tasks, list):
                        return [task for task in tasks if task and isinstance(task, str)]
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e} for string: {json_str}")
            
            # If no valid JSON array found, try to parse as text
            return self._parse_tasks_from_numbered_list(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing tasks from response: {str(e)}")
            return []
    
    def _parse_tasks_from_numbered_list(self, response_text: str) -> List[str]:
        """Parse tasks from text with numbered list format."""
        try:
            tasks = []
            # Split text by newlines to find list items
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for lines starting with numbers (1. Task) or bullet points
                if (line[0].isdigit() and '. ' in line[:5]) or line.startswith('- ') or line.startswith('* '):
                    # Extract task content after the prefix
                    if line[0].isdigit() and '. ' in line[:5]:
                        task = line[line.find('. ') + 2:].strip()
                    else:
                        task = line[2:].strip()
                    
                    if task:  # Only add non-empty tasks
                        tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error parsing numbered list: {str(e)}")
            return []
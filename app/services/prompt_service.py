# app/services/prompt_service.py
import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptService:
    """
    Service for managing and formatting prompts used by various LLM components.
    """
    
    def __init__(self, prompts_file_path: str = None):
        """
        Initialize the prompt service.
        
        Args:
            prompts_file_path: Path to the prompts JSON file. If None, uses default path.
        """
        if prompts_file_path is None:
            # Default path - assuming it's in the project root
            self.prompts_file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'prompts.json'
            )
        else:
            self.prompts_file_path = prompts_file_path
            
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from the JSON file."""
        try:
            with open(self.prompts_file_path, 'r') as f:
                prompts = json.load(f)
            logger.info(f"Successfully loaded prompts from {self.prompts_file_path}")
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts from {self.prompts_file_path}: {str(e)}")
            # Return empty dictionary if file doesn't exist or has errors
            return {}
    
    def reload_prompts(self) -> None:
        """Reload prompts from the JSON file."""
        self.prompts = self._load_prompts()
    
    def get_prompt(self, component: str, prompt_name: str) -> str:
        """
        Get a prompt by component and name.
        
        Args:
            component: Component name (e.g., 'task_splitter', 'entity_extractor')
            prompt_name: Specific prompt name
            
        Returns:
            The prompt template as a string
        """
        try:
            return self.prompts.get(component, {}).get(prompt_name, "")
        except Exception as e:
            logger.error(f"Error retrieving prompt {component}.{prompt_name}: {str(e)}")
            return ""
    
    def format_prompt(self, component: str, prompt_name: str, **kwargs) -> str:
        """
        Get and format a prompt with the provided parameters.
        
        Args:
            component: Component name
            prompt_name: Specific prompt name
            **kwargs: Parameters to format the prompt with
            
        Returns:
            The formatted prompt
        """
        prompt_template = self.get_prompt(component, prompt_name)
        
        if not prompt_template:
            logger.warning(f"Prompt template {component}.{prompt_name} not found")
            return ""
            
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required parameter {e} for prompt {component}.{prompt_name}")
            return prompt_template
        except Exception as e:
            logger.error(f"Error formatting prompt {component}.{prompt_name}: {str(e)}")
            return prompt_template

# Create a singleton instance for global use
_prompt_service_instance = None

def get_prompt_service():
    """Get or create the prompt service singleton instance."""
    global _prompt_service_instance
    if _prompt_service_instance is None:
        _prompt_service_instance = PromptService()
    return _prompt_service_instance
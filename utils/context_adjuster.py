"""
Context Adjuster

This module provides a simple utility to adjust company context based on user feedback.
It uses the OpenRouter API to refine the context according to specific user instructions.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def adjust_company_context(current_context: Dict[str, Any], user_feedback: str) -> Optional[Dict[str, Any]]:
    """
    Adjust company context based on user feedback.
    
    Args:
        current_context: The current company context dictionary
        user_feedback: User's feedback on what needs to be adjusted
        
    Returns:
        Adjusted company context dictionary or None if adjustment failed
    """
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        raise ValueError("OpenRouter API key is required for context adjustment")
    
    # Extract relevant information from current context
    company_name = current_context.get("name", "Unknown")
    company_description = current_context.get("description", "")
    company_url = current_context.get("url", "")
    target_geography = current_context.get("target_geography", "Global")
    
    # Create prompt for the model
    prompt = f"""
    I need you to adjust the company context for {company_name} ({company_url}) based on user feedback.
    
    CURRENT CONTEXT:
    Company Name: {company_name}
    Target Geography: {target_geography}
    
    Description:
    {company_description}
    
    USER FEEDBACK:
    {user_feedback}
    
    Please adjust the company context to address the user's feedback. Maintain the same structure and level of detail,
    but incorporate the changes requested by the user. Focus specifically on what the user wants to change.
    
    Return ONLY a JSON object with the following structure:
    {{
        "name": "Company Name",
        "description": "Adjusted company description...",
        "url": "Company URL",
        "target_geography": "Adjusted target geography"
    }}
    
    Do not include any explanations or notes outside the JSON object.
    """
    
    # Call OpenRouter API
    try:
        logger.info(f"Adjusting context for {company_name} based on user feedback")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet-20241022:beta",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that adjusts company context based on user feedback. You only respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from the response
        try:
            # First try to parse the entire response as JSON
            adjusted_context = json.loads(content)
            logger.debug("Successfully parsed response as JSON")
        except json.JSONDecodeError:
            # If that fails, look for JSON object in the response
            logger.debug("Failed to parse entire response as JSON, looking for JSON object")
            json_match = content.strip()
            
            # Remove markdown code block indicators if present
            json_match = json_match.replace("```json", "").replace("```", "").strip()
            
            try:
                adjusted_context = json.loads(json_match)
                logger.debug("Successfully parsed JSON from response")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from response")
                return None
        
        # Preserve fields that weren't in the adjusted context
        for key, value in current_context.items():
            if key not in adjusted_context:
                adjusted_context[key] = value
        
        logger.info(f"Successfully adjusted context for {company_name}")
        return adjusted_context
        
    except Exception as e:
        logger.error(f"Error adjusting context: {str(e)}")
        logger.exception("Detailed exception:")
        return None 
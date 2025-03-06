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
import re
from langchain_core.messages import SystemMessage, HumanMessage

from ai_utils.new_company_context_agent import ChatOpenRouter

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

IMPORTANT INSTRUCTIONS:
1. You MUST maintain the FULL LENGTH and DETAIL of the original description
2. DO NOT summarize or shorten the description in any way
3. Only modify the specific aspects mentioned in the user feedback
4. Keep all existing information that isn't explicitly changed by the feedback
5. Preserve the same level of detail, tone, and structure as the original
6. If the original description is long and detailed, your response must be equally long and detailed

Please adjust the company context to address the user's feedback. Your response must be AT LEAST as long and detailed as the original description.

IMPORTANT: You must ONLY respond with a properly formatted JSON object. Do not include any comments, explanations, 
code block formatting, or surrounding text. The response should parse as valid JSON directly.

The JSON object should have exactly this structure:
{{
    "name": "Company Name",
    "description": "Adjusted company description that MAINTAINS THE FULL LENGTH AND DETAIL of the original...",
    "url": "Company URL",
    "target_geography": "Adjusted target geography"
}}

For the JSON to be valid:
1. Use double quotes for strings, not single quotes
2. No trailing commas
3. No comments
4. No formatting blocks (like ```json)
5. All fields must be properly escaped strings
"""
    
    # Call OpenRouter API
    try:
        logger.info(f"Adjusting context for {company_name} based on user feedback")
        
        # Initialize ChatOpenRouter
        llm = ChatOpenRouter(api_key=api_key, model="anthropic/claude-3-sonnet")
        
        # Prepare messages
        messages = [
            SystemMessage(content="You are a JSON-formatted response generator. You MUST ONLY return valid JSON, with no explanation, no code blocks, no backticks. Your ENTIRE response must be parseable as JSON."),
            HumanMessage(content=prompt)
        ]
        
        # Call the LLM
        logger.debug("Sending request to OpenRouter API")
        response = llm.invoke(messages)
        content = response.content
        
        # Check for empty content
        if not content or not content.strip():
            logger.error("Received empty content from API")
            return None
        
        # Log the raw response for debugging
        logger.debug(f"Raw response content: {content[:500]}...")
        
        # Try to parse the content as JSON
        try:
            # Remove any markdown code block formatting if present
            if content.startswith("```json"):
                content = content.strip().removeprefix("```json").removesuffix("```").strip()
            elif content.startswith("```"):
                content = content.strip().removeprefix("```").removesuffix("```").strip()
                
            adjusted_context = json.loads(content)
            
            # Validate the updated context
            if not isinstance(adjusted_context, dict):
                logger.error(f"Adjusted context is not a dictionary: {type(adjusted_context)}")
                return None
                
            if "name" not in adjusted_context:
                logger.error("Adjusted context does not contain name field")
                return None
            
            # Preserve fields that weren't in the adjusted context
            for key, value in current_context.items():
                if key not in adjusted_context:
                    adjusted_context[key] = value
            
            # Log success
            logger.info(f"Successfully adjusted context for {company_name}")
            logger.debug(f"Adjusted context: {json.dumps(adjusted_context, indent=2)}")
            
            return adjusted_context
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {str(e)}")
            logger.error(f"Response content: {content}")
            
            # Fallback to original context with appended feedback
            fallback_context = current_context.copy()
            fallback_context["description"] = f"{current_context.get('description', '')}\n\nUPDATED BASED ON FEEDBACK: {user_feedback}"
            logger.info("Created fallback context with appended user feedback")
            return fallback_context
            
    except Exception as e:
        logger.error(f"Error adjusting context: {str(e)}")
        
        # Last resort fallback
        try:
            fallback_context = current_context.copy()
            fallback_context["description"] = f"{current_context.get('description', '')}\n\nNOTE: {user_feedback}"
            logger.info("Created emergency fallback context")
            return fallback_context
        except:
            # If all else fails
            return current_context 
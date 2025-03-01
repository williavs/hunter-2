"""
API Utility Functions

This module provides utility functions for API key validation and management 
used throughout the Email GTM Wizard application.
"""

import os
import logging
from ai_utils.personality_analyzer import PersonalityAnalyzer

# Use the centralized logger
logger = logging.getLogger(__name__)

# Define the fixed model name - duplicated here to avoid circular import
FIXED_MODEL = "anthropic/claude-3.5-haiku-20241022:beta"

async def test_api_keys():
    """Test if the API keys are valid"""
    try:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        
        if not openrouter_key:
            return False, "OpenRouter API key is missing"
        
        if not tavily_key:
            return False, "Tavily API key is missing"
        
        # Test the OpenRouter API key
        logger.debug("Creating PersonalityAnalyzer to test API keys")
        analyzer = PersonalityAnalyzer(
            openrouter_api_key=openrouter_key,
            tavily_api_key=tavily_key,
            model_name=FIXED_MODEL
        )
        
        # Test the API key
        key_valid = await analyzer.test_openrouter_api_key()
        
        if not key_valid:
            return False, "OpenRouter API key validation failed"
        
        return True, "API keys are valid"
    except Exception as e:
        logger.error(f"Error testing API keys: {str(e)}")
        return False, f"Error testing API keys: {str(e)}" 
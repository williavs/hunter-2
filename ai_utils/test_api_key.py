"""
Simple script to validate API keys are loaded correctly.
Run this script directly to check if your .env file is configured properly.
"""

import os
import logging
from dotenv import load_dotenv
from utils.logging_config import configure_logging

# Use the centralized logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
dotenv_path = os.path.join(root_dir, '.env')

logger.info(f"Looking for .env file at: {dotenv_path}")
logger.info(f".env file exists: {os.path.exists(dotenv_path)}")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f"No .env file found at {dotenv_path}")

# Check if API keys are set
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

logger.info(f"OPENROUTER_API_KEY loaded: {'Yes' if openrouter_key else 'No'}")
if openrouter_key:
    # Only show first 5 characters for security
    key_preview = openrouter_key[:5] + "..." if len(openrouter_key) > 5 else openrouter_key
    logger.info(f"OPENROUTER_API_KEY starts with: {key_preview}")
    logger.info(f"OPENROUTER_API_KEY length: {len(openrouter_key)}")
    
    # Check for placeholder values
    if "your_key" in openrouter_key or "placeholder" in openrouter_key.lower():
        logger.error("DETECTED PLACEHOLDER VALUE IN OPENROUTER_API_KEY. Please update your .env file with a real API key.")
else:
    logger.error("OPENROUTER_API_KEY is not set in environment variables")

logger.info(f"TAVILY_API_KEY loaded: {'Yes' if tavily_key else 'No'}")
if tavily_key:
    # Only show first 5 characters for security
    key_preview = tavily_key[:5] + "..." if len(tavily_key) > 5 else tavily_key
    logger.info(f"TAVILY_API_KEY starts with: {key_preview}")
    logger.info(f"TAVILY_API_KEY length: {len(tavily_key)}")
else:
    logger.error("TAVILY_API_KEY is not set in environment variables")

if __name__ == "__main__":
    # Configure logging with INFO level for this script when run directly
    configure_logging(level=logging.INFO)
    
    print("\nAPI Key Check Summary:")
    print(f"OpenRouter API Key: {'✅ Present' if openrouter_key else '❌ Missing'}")
    print(f"Tavily API Key: {'✅ Present' if tavily_key else '❌ Missing'}")
    print("\nRun this script anytime to verify your API keys are properly loaded.") 
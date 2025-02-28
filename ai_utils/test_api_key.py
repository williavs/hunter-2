"""
Test script to validate API keys are loaded correctly.
Run this script directly to check if your .env file is configured properly.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("api_key_test")

# First attempt to load keys with dotenv
try:
    from dotenv import load_dotenv
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to find the .env file in the project root
    root_dir = os.path.dirname(current_dir)
    dotenv_path = os.path.join(root_dir, '.env')
    
    logger.info(f"Looking for .env file at: {dotenv_path}")
    logger.info(f".env file exists: {os.path.exists(dotenv_path)}")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment variables from {dotenv_path}")
    else:
        logger.warning(f"No .env file found at {dotenv_path}")
except Exception as e:
    logger.error(f"Error loading dotenv: {str(e)}")

# Check if API keys are set
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

logger.info(f"ANTHROPIC_API_KEY loaded: {'Yes' if anthropic_key else 'No'}")
if anthropic_key:
    # Only show first 10 characters for security
    key_preview = anthropic_key[:10] + "..." if len(anthropic_key) > 10 else anthropic_key
    logger.info(f"ANTHROPIC_API_KEY starts with: {key_preview}")
    logger.info(f"ANTHROPIC_API_KEY length: {len(anthropic_key)}")
    logger.info(f"ANTHROPIC_API_KEY format appears to be: {'valid' if anthropic_key.startswith('sk-ant') else 'invalid'}")
    
    # Check for placeholder values
    if "your_anthr" in anthropic_key or "placeholder" in anthropic_key.lower():
        logger.error("DETECTED PLACEHOLDER VALUE IN ANTHROPIC_API_KEY. Please update your .env file with a real API key.")
else:
    logger.error("ANTHROPIC_API_KEY is not set in environment variables")

logger.info(f"TAVILY_API_KEY loaded: {'Yes' if tavily_key else 'No'}")

# Try to import and initialize the Anthropic client directly
try:
    logger.info("Attempting to initialize Anthropic client directly...")
    import anthropic
    client = anthropic.Anthropic(api_key=anthropic_key)
    logger.info("Successfully initialized Anthropic client")
    
    # Try a simple API call to verify the key works
    logger.info("Attempting a simple API call...")
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Say hello"}
            ]
        )
        logger.info(f"API call successful! Response: {message.content}")
        logger.info("✅ Your Anthropic API key is working correctly!")
    except Exception as api_error:
        logger.error(f"API call failed with error: {str(api_error)}")
        logger.error("❌ Your Anthropic API key is not working correctly.")
        
except Exception as e:
    logger.error(f"Error initializing Anthropic client: {str(e)}")

if __name__ == "__main__":
    # Nothing to do here, the script runs when imported
    pass 
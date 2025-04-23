import os
import sys
import logging
from importlib import reload
from pydantic import SecretStr

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Import the company context module
from old_files.new_company_context_agent import ChatOpenRouter, analyze_company_context_agentic

def test_openrouter():
    """Test the OpenRouter connection."""
    logger.info("=== TESTING OPENROUTER CONNECTION ===")
    
    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OpenRouter API key not found in environment")
        return False
    
    logger.info(f"OpenRouter API key found (length: {len(api_key)})")
    
    try:
        # Create the ChatOpenRouter instance
        logger.info("Creating ChatOpenRouter instance...")
        llm = ChatOpenRouter(api_key=api_key)
        
        # Send a test message
        logger.info("Sending test message...")
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="How are you today?")])
        
        # Log the response
        response_text = response.content[:100] + "..." if len(response.content) > 100 else response.content
        logger.info(f"Response received: {response_text}")
        logger.info("OpenRouter connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"OpenRouter connection test failed: {str(e)}")
        return False

def test_company_analysis():
    """Test the company analysis function."""
    logger.info("\n=== TESTING COMPANY ANALYSIS ===")
    
    # Target company for testing
    test_url = "https://gtmwizards.com"
    target_geography = "Europe and North America"
    
    try:
        # Run the analysis
        logger.info(f"Testing company context analysis for {test_url}...")
        result = analyze_company_context_agentic(
            company_url=test_url,
            target_geography=target_geography
        )
        
        # Log the results
        logger.info("Analysis completed successfully!")
        logger.info(f"Company name: {result.get('name', 'Not found')}")
        logger.info(f"Confidence: {result.get('confidence', 'Not specified')}")
        logger.info(f"Search quality: {result.get('search_quality', 'Not specified')}")
        logger.info(f"Description length: {len(result.get('description', ''))}")
        logger.info(f"Target geography: {result.get('target_geography', 'Not specified')}")
        
        if 'errors' in result and result['errors']:
            logger.warning(f"Errors encountered: {result['errors']}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all debug tests."""
    logger.info("Running OpenRouter debug script...")
    
    # Test OpenRouter connection
    openrouter_success = test_openrouter()
    
    # Test company analysis if OpenRouter connection is successful
    if openrouter_success:
        analysis_success = test_company_analysis()
        if not analysis_success:
            logger.error("\n❌ Analysis test failed!")
    else:
        logger.error("\n❌ OpenRouter connection test failed!")
    
    logger.info("Debug tests completed.")

if __name__ == "__main__":
    main() 
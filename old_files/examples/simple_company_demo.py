#!/usr/bin/env python3
"""
Simple demonstration of the company analyzer.

This script demonstrates how to use the simple_company_analyzer module to analyze
company information based on a URL.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_utils.simple_company_analyzer import analyze_company

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the company analysis demo."""
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found in environment")
        return 1
        
    if not os.environ.get("TAVILY_API_KEY"):
        logger.error("TAVILY_API_KEY not found in environment")
        return 1
    
    # Analyze a test company
    company_url = "https://gtmwizards.com"
    target_geography = "new york"
    
    print(f"\n{'=' * 50}")
    print(f"Analyzing company: {company_url}")
    print(f"Target geography: {target_geography}")
    print(f"{'=' * 50}\n")
    
    result = analyze_company(
        company_url=company_url,
        target_geography=target_geography
    )
    
    # Display the results
    print(f"\n{'=' * 50}")
    print(f"ANALYSIS RESULTS")
    print(f"{'=' * 50}")
    print(f"Company Name: {result['name']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Target Geography: {result['target_geography']}")
    print(f"Search Quality: {result['search_quality']:.2f}")
    
    print("\nDESCRIPTION:")
    print(f"{result['description']}")
    
    print("\nFULL ANALYSIS:")
    print(f"{result['analysis']}")
    
    if 'errors' in result and result['errors']:
        print("\nERRORS:")
        for error in result['errors']:
            print(f"- {error}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
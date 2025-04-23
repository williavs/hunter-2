"""
Simple example of using the company context analyzer.

This script demonstrates how to use the analyze_company_context_agentic function
to get information about a company based on its website URL.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the analyzer
from old_files.new_company_context_agent import analyze_company_context_agentic

def analyze_company(url, target_geography=None):
    """Analyze a company and print the results."""
    print(f"\n=== ANALYZING COMPANY: {url} ===")
    if target_geography:
        print(f"Target Geography: {target_geography}")
        
    # Run the analysis    
    result = analyze_company_context_agentic(
        company_url=url,
        target_geography=target_geography
    )
    
    # Print the results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Company Name: {result.get('name', 'Not found')}")
    print(f"Confidence: {result.get('confidence', 'Not specified')}")
    print(f"Target Geography: {result.get('target_geography', 'Global')}")
    print("\nDescription:")
    print(result.get('description', 'No description available'))
    
    # Print any errors
    if 'errors' in result and result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"- {error}")
    
    return result

if __name__ == "__main__":
    # Test with a few example URLs
    companies = [
        {"url": "https://gtmwizards.com", "geography": "North America and Europe"},
        {"url": "https://www.salesforce.com", "geography": None},
    ]
    
    for company in companies:
        analyze_company(company["url"], company["geography"])
        print("\n" + "="*50 + "\n") 
"""
Minimal example script demonstrating the personality analyzer

This script shows how to use the personality analyzer independently of the 
Streamlit application, which is useful for testing or batch processing.

Usage:
    python example.py
"""

import asyncio
import pandas as pd
import os
from dotenv import load_dotenv
from personality_analyzer import PersonalityAnalyzer

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run a minimal example of the personality analyzer."""
    
    # Check for required API keys
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if not openrouter_api_key or not tavily_api_key:
        print("Error: Please set OPENROUTER_API_KEY and TAVILY_API_KEY environment variables.")
        return
    
    # Create a simple sample dataframe with one contact
    data = {
        'name': ['John Doe'],
        'company': ['Example Company'],
        'title': ['CEO'],
        'website_content': ['Example Company is a leading provider of innovative solutions...']
    }
    
    df = pd.DataFrame(data)
    
    print(f"Analyzing {len(df)} contact...")
    
    # Create the analyzer
    analyzer = PersonalityAnalyzer(
        openrouter_api_key=openrouter_api_key,
        tavily_api_key=tavily_api_key,
        model_name="anthropic/claude-3.5-haiku-20241022:beta"
    )
    
    # Run the analysis
    result_df = await analyzer.analyze_personality(df)
    
    # Print the result
    if 'personality_analysis' in result_df.columns:
        print("\nAnalysis complete!")
        print(f"Conversation Style: {result_df.iloc[0]['conversation_style']}")
        print(f"Professional Interests: {result_df.iloc[0]['professional_interests']}")
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 
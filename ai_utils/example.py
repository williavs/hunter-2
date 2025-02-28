"""
Example script demonstrating the personality analyzer

This script shows how to use the personality analyzer independently of the 
Streamlit application, which is useful for testing or batch processing.

Usage:
    python example.py
"""

import asyncio
import pandas as pd
import os
from dotenv import load_dotenv
from personality_analyzer import PersonalityAnalyzer, ContactInfo

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run an example of the personality analyzer on sample data."""
    
    # Check for required API keys
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if not anthropic_api_key or not tavily_api_key:
        print("Error: Please set ANTHROPIC_API_KEY and TAVILY_API_KEY environment variables.")
        print("You can create a .env file with these values or export them in your shell.")
        return
    
    # Create sample contact data
    data = [
        {
            "name": "Elon Musk",
            "email": "elon@tesla.com",
            "company": "Tesla, SpaceX, X",
            "title": "CEO",
            "website": "https://twitter.com/elonmusk"
        },
        {
            "name": "Sundar Pichai",
            "email": "sundar@google.com",
            "company": "Google",
            "title": "CEO",
            "website": "https://www.google.com"
        },
        {
            "name": "Sam Altman",
            "email": "sam@openai.com",
            "company": "OpenAI",
            "title": "CEO",
            "website": "https://twitter.com/sama"
        }
    ]
    
    contacts_df = pd.DataFrame(data)
    
    print(f"Analyzing {len(contacts_df)} contacts...")
    
    # Create the analyzer
    analyzer = PersonalityAnalyzer(
        anthropic_api_key=anthropic_api_key,
        tavily_api_key=tavily_api_key,
        model_name="claude-3-7-sonnet-20250219",  # You might want to use a different model
        max_concurrent=3  # Limit concurrent requests
    )
    
    # Run the analysis
    results = await analyzer.analyze_personalities(contacts_df)
    
    # Print results
    for contact_id, result in results.items():
        print(f"\n{'='*40}")
        print(f"Analysis for {contact_id}")
        print(f"{'='*40}")
        
        print(f"\nConversation Style: {result.conversation_style}")
        
        print("\nProfessional Interests:")
        for interest in result.professional_interests:
            print(f"- {interest}")
        
        print("\nPersonality Analysis:")
        # Print a shorter version for demonstration
        paragraphs = result.personality_analysis.split('\n\n')
        for i, para in enumerate(paragraphs[:3]):  # Just show first 3 paragraphs
            print(f"{para}\n")
        
        if len(paragraphs) > 3:
            print("... [additional analysis truncated for brevity] ...")
        
        if result.error:
            print(f"\nErrors: {result.error}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 
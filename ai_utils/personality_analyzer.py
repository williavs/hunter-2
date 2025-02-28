"""
Personality Analyzer Agent

This module implements an agent that analyzes a person's personality based on 
their online presence and LinkedIn profile. It uses LangGraph for workflow
orchestration and Anthropic's Claude for analysis.

Usage:
    analyzer = PersonalityAnalyzer(anthropic_api_key, tavily_api_key)
    results = await analyzer.analyze_personalities(contacts_df)

Requirements:
    - langchain
    - langgraph
    - langchain_anthropic
    - tavily_python
    - pandas
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import json
import re

# Import dotenv and load environment variables
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Updated imports for the current LangGraph API
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== ADD DEBUG LEVEL LOGGING ====
import sys
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# ===============================

class ContactInfo(BaseModel):
    """Contact information for a person to analyze."""
    name: str = Field(description="Full name of the contact")
    email: Optional[str] = Field(default=None, description="Email address")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    twitter_url: Optional[str] = Field(default=None, description="Twitter/X profile URL")
    personal_website: Optional[str] = Field(default=None, description="Personal website URL")
    company: Optional[str] = Field(default=None, description="Company name")
    title: Optional[str] = Field(default=None, description="Job title")
    website_content: Optional[str] = Field(default=None, description="Scraped website content")
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ContactInfo":
        """
        Create a ContactInfo object from a row dictionary with proper fallbacks.
        
        This method ensures required fields are always present and handles missing data gracefully.
        """
        # Ensure required fields have fallbacks
        data = row.copy()
        
        # If name is missing, try to generate one from other fields or use a placeholder
        if "name" not in data or not data["name"]:
            if "email" in data and data["email"]:
                # Extract name from email (user part)
                email_parts = data["email"].split('@')
                if len(email_parts) > 0:
                    data["name"] = email_parts[0].replace(".", " ").title()
            elif "company" in data and data["company"]:
                # Use company name as fallback
                data["name"] = f"Contact at {data['company']}"
            else:
                # Use a generated unique identifier as absolute fallback
                import uuid
                data["name"] = f"Contact-{str(uuid.uuid4())[:8]}"
        
        # Only include fields that are in the model
        model_fields = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in model_fields}
        
        return cls(**filtered_data)

class AnalysisResult(BaseModel):
    """Results of personality analysis."""
    contact_id: str
    personality_analysis: str
    conversation_style: str
    professional_interests: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    search_queries_used: List[str] = Field(default_factory=list)
    search_results: List[Dict] = Field(default_factory=list)

class PersonalityState(TypedDict):
    """State maintained throughout the analysis process."""
    contact: Dict[str, Any]  # Changed from ContactInfo to Dict for simpler serialization
    contact_data: Dict[str, Any]
    messages: List[Dict]
    search_queries: List[str]
    search_results: List[Dict]
    analysis: Optional[Dict[str, Any]]  # Changed from AnalysisResult to Dict
    errors: List[str]
    complete: bool

class PersonalityAnalyzer:
    """Agent that analyzes personalities based on online information."""
    
    def __init__(self, 
                 anthropic_api_key: Optional[str] = None,
                 tavily_api_key: Optional[str] = None,
                 model_name: str = "claude-3-7-sonnet-20250219",
                 max_concurrent: int = 5):
        """
        Initialize the personality analyzer.
        
        Args:
            anthropic_api_key: API key for Anthropic (defaults to env var ANTHROPIC_API_KEY)
            tavily_api_key: API key for Tavily search (defaults to env var TAVILY_API_KEY)
            model_name: Claude model to use
            max_concurrent: Maximum number of concurrent analyses
        """
        # Print dotenv info for debugging
        import os.path
        logger.debug(f"Current working directory: {os.getcwd()}")
        dotenv_path = os.path.join(os.getcwd(), '.env')
        logger.debug(f"Looking for .env file at: {dotenv_path}")
        logger.debug(f".env file exists: {os.path.exists(dotenv_path)}")
        
        # Try to reload dotenv just to be sure
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)
            logger.debug("Reloaded environment variables from .env file")
        except Exception as e:
            logger.error(f"Error reloading .env file: {str(e)}")
        
        # Get API keys
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        
        # DEBUG: Log API key information (safely)
        if self.anthropic_api_key:
            key_preview = self.anthropic_api_key[:10] + "..." if len(self.anthropic_api_key) > 10 else "[empty]"
            logger.debug(f"Anthropic API key loaded (first chars): {key_preview}")
            logger.debug(f"Anthropic API key length: {len(self.anthropic_api_key)}")
            logger.debug(f"Anthropic API key format appears to be: {'valid' if self.anthropic_api_key.startswith('sk-ant') else 'invalid'}")
            
            # Check for placeholder values
            if "your_anthr" in self.anthropic_api_key or "placeholder" in self.anthropic_api_key.lower():
                logger.error("DETECTED PLACEHOLDER VALUE IN ANTHROPIC_API_KEY. Please update your .env file with a real API key.")
                raise ValueError("Placeholder detected in ANTHROPIC_API_KEY. Please update with a real API key.")
        else:
            logger.debug("No Anthropic API key found!")
        
        if not self.anthropic_api_key:
            logger.error("Anthropic API key is required but not found")
            raise ValueError("Anthropic API key is required")
        
        if not self.tavily_api_key:
            logger.error("Tavily API key is required but not found")
            raise ValueError("Tavily API key is required")
        
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize components
        try:
            # FIXED: Changed parameter name from 'anthropic_api_key' to 'api_key'
            # This is the likely cause of the 401 error
            logger.debug(f"Initializing ChatAnthropic with model: {self.model_name}")
            self.llm = ChatAnthropic(
                model=self.model_name,
                temperature=0.1,
                api_key=self.anthropic_api_key  # Changed from anthropic_api_key to api_key
            )
            logger.debug("ChatAnthropic initialization successful")
        except Exception as e:
            logger.error(f"Error initializing ChatAnthropic: {str(e)}")
            logger.exception("Detailed exception information:")
            raise
        
        # Initialize search tool
        try:
            self.search_tool = TavilySearchResults(
                api_key=self.tavily_api_key,
                max_results=5
            )
            logger.debug("TavilySearchResults initialization successful")
        except Exception as e:
            logger.error(f"Error initializing TavilySearchResults: {str(e)}")
            logger.exception("Detailed exception information:")
            raise
        
        # Build the workflow using StateGraph
        self.workflow = self._build_workflow()
    
    async def test_anthropic_api_key(self) -> bool:
        """
        Test if the Anthropic API key is valid by making a simple request.
        
        Returns:
            bool: True if the key is valid, False otherwise
        """
        try:
            logger.debug("Testing Anthropic API key with a simple request")
            
            # Log key format details (safely)
            if self.anthropic_api_key:
                key_preview = self.anthropic_api_key[:10] + "..." if len(self.anthropic_api_key) > 10 else "[empty]"
                logger.debug(f"API key being tested: {key_preview}")
                logger.debug(f"API key length: {len(self.anthropic_api_key)}")
                logger.debug(f"API key format check: starts with 'sk-ant'? {'Yes' if self.anthropic_api_key.startswith('sk-ant') else 'No'}")
                
                # Check for common issues
                if '"' in self.anthropic_api_key:
                    logger.warning("API key contains quote characters - this may cause authentication issues")
                if ' ' in self.anthropic_api_key:
                    logger.warning("API key contains spaces - this may cause authentication issues")
            else:
                logger.error("No API key available for testing")
                return False
            
            # Create a direct test with minimal dependencies
            try:
                import anthropic
                logger.debug("Testing with direct anthropic client")
                direct_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                # Just create a simple client to test authentication
                logger.debug("Direct anthropic client created successfully")
            except Exception as e:
                logger.error(f"Direct anthropic client test failed: {str(e)}")
            
            # Test with LangChain's ChatAnthropic
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Respond with 'API key is working' if you can see this message.")
            ]
            
            # Make a simple request
            logger.debug("Sending test request to ChatAnthropic")
            response = self.llm.invoke(messages)
            
            # Check if we got a valid response
            if response and hasattr(response, 'content') and response.content:
                logger.debug(f"Anthropic API test successful, received response: {response.content[:50]}...")
                return True
            else:
                logger.error("Anthropic API test failed: Received empty response")
                return False
                
        except Exception as e:
            logger.error(f"Anthropic API test failed with error: {str(e)}")
            logger.exception("Detailed exception:")
            return False
    
    def _planning_task(self, state: PersonalityState) -> PersonalityState:
        """Plan what information to search for based on contact info."""
        contact_data = state["contact_data"]
        
        # Create context for the LLM
        prompt = f"""
        I need to analyze the personality of the following person to help with sales outreach:
        
        Contact Data:
        {contact_data}
        
        Your task is to create 3-5 search queries that will help me understand their:
        1. Communication style
        2. Professional background and interests
        3. Personal values or motivations
        4. Potential conversation starters or topics of interest
        
        For each query, explain why this information would be valuable for personalizing outreach.
        Format your response as a JSON array of search queries only.
        """
        
        try:
            # Get search queries from LLM
            messages = [
                SystemMessage(content="You are a helpful assistant that creates search queries to gather information about professional contacts."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract search queries from LLM response (assuming valid JSON array format)
            
            # Look for JSON array in the response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                queries_json = json_match.group(0)
                search_queries = json.loads(queries_json)
            else:
                # Fallback to basic extraction if JSON parsing fails
                lines = response.content.strip().split('\n')
                search_queries = [line.strip() for line in lines if line.strip()]
            
            # Add the queries to the messages
            state["messages"].append({
                "role": "system",
                "content": f"Generated search queries: {search_queries}"
            })
            
            # Store search queries in state
            state["search_queries"] = search_queries[:5]  # Limit to 5 queries
            
        except Exception as e:
            error_msg = f"Error in planning: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            contact_name = state["contact"].get("name", "Unknown")
            state["search_queries"] = [f"{contact_name} professional background"]  # Fallback query
        
        return state
    
    async def _search_task(self, state: PersonalityState) -> PersonalityState:
        """Execute searches for each query in parallel."""
        search_queries = state.get("search_queries", [])
        
        if not search_queries:
            state["errors"].append("No search queries available")
            return state
        
        search_results = []
        
        # Execute searches in parallel
        async def execute_search(query: str):
            try:
                result = await asyncio.to_thread(self.search_tool.invoke, query)
                return {"query": query, "results": result}
            except Exception as e:
                logger.error(f"Search error for query '{query}': {str(e)}")
                return {"query": query, "results": [], "error": str(e)}
        
        # Run searches concurrently
        tasks = [execute_search(query) for query in search_queries]
        search_outputs = await asyncio.gather(*tasks)
        
        # Collect results
        for output in search_outputs:
            if "error" not in output:
                search_results.append(output)
        
        state["search_results"] = search_results
        
        # Add the search results to the messages
        state["messages"].append({
            "role": "system",
            "content": f"Search results: {len(search_results)} queries returned information."
        })
        
        return state
    
    def _analysis_task(self, state: PersonalityState) -> PersonalityState:
        """Analyze search results to generate personality insights."""
        contact = state["contact"]
        contact_data = state["contact_data"]
        search_results = state["search_results"]
        
        contact_name = contact.get("name", "Unknown")
        
        if not search_results:
            state["errors"].append("No search results available for analysis")
            state["analysis"] = {
                "contact_id": contact_name,
                "personality_analysis": "Insufficient data for analysis",
                "conversation_style": "Unknown",
                "error": "No search results available",
                "professional_interests": [],
                "search_queries_used": [],
                "search_results": []
            }
            state["complete"] = True
            return state
        
        # Prepare search results for prompt
        results_text = ""
        for result in search_results:
            results_text += f"\nQuery: {result['query']}\n"
            for i, item in enumerate(result['results']):
                results_text += f"Result {i+1}: {item.get('title', 'No title')}\n"
                results_text += f"URL: {item.get('url', 'No URL')}\n"
                results_text += f"Content: {item.get('content', 'No content')[:500]}...\n\n"
        
        # Create analysis prompt
        prompt = f"""
        Based on the following information about the contact, please create a personality analysis that would help me personalize my outreach.

        Contact Information:
        {contact_data}
        
        Search Results:
        {results_text}
        
        Please provide the following in your analysis:
        
        1. Personality Analysis: A 2-3 paragraph summary of their likely personality traits, communication preferences, and working style based on available information.
        
        2. Conversation Style: How they likely prefer to communicate (direct, analytical, relational, etc.)
        
        3. Professional Interests: A list of 3-5 professional topics they seem most interested in.
        
        4. Approach Recommendations: Specific suggestions for how to approach this person in a sales context, including:
           - Tone to use
           - Topics to highlight
           - Potential pain points to address
           - Questions that might engage them
        
        5. Conversation Starters: 2-3 specific conversation starters tailored to their interests and background.
        
        If there's insufficient information for any section, please note that rather than making assumptions.
        """
        
        try:
            # Get analysis from LLM
            messages = [
                SystemMessage(content="You are an expert personality analyst who helps sales professionals personalize their outreach."),
                HumanMessage(content=prompt)
            ]
            
            state["messages"].append({
                "role": "user",
                "content": "Please analyze the personality based on the search results."
            })
            
            response = self.llm.invoke(messages)
            
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            
            # Extract professional interests (assume they're in a list format)
            interests = []
            interests_section = re.search(r'Professional Interests:.*?(\n\n|\Z)', response.content, re.DOTALL)
            if interests_section:
                interest_text = interests_section.group(0)
                # Look for list items (- or • or 1. format)
                interest_items = re.findall(r'(?:[-•*]|\d+\.)\s+(.*?)(?=\n[-•*]|\n\d+\.|\n\n|\Z)', interest_text, re.DOTALL)
                interests = [item.strip() for item in interest_items if item.strip()]
            
            # Extract conversation style
            style = "Unknown"
            style_section = re.search(r'Conversation Style:(.*?)(?=\n\n|\n\d\.|\Z)', response.content, re.DOTALL)
            if style_section:
                style = style_section.group(1).strip()
            
            # Create analysis result
            analysis = {
                "contact_id": contact_name,
                "personality_analysis": response.content,
                "conversation_style": style,
                "professional_interests": interests,
                "search_queries_used": [r.get("query", "") for r in search_results],
                "search_results": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", "")
                    } for result in search_results for item in result.get("results", [])
                ]
            }
            
            state["analysis"] = analysis
            
        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["analysis"] = {
                "contact_id": contact_name,
                "personality_analysis": "Error during analysis",
                "conversation_style": "Unknown",
                "error": error_msg,
                "professional_interests": [],
                "search_queries_used": [],
                "search_results": []
            }
        
        state["complete"] = True
        return state
    
    def _build_workflow(self):
        """Build the LangGraph workflow for personality analysis using StateGraph."""
        # Create a state graph with the PersonalityState type
        workflow = StateGraph(PersonalityState)
        
        # Add nodes for each task
        workflow.add_node("planning", self._planning_task)
        workflow.add_node("search", self._search_task)
        workflow.add_node("analysis_task", self._analysis_task)
        
        # Define the edges to create a linear flow
        workflow.add_edge("planning", "search")
        workflow.add_edge("search", "analysis_task")
        workflow.add_edge("analysis_task", END)
        
        # Set the entry point
        workflow.set_entry_point("planning")
        
        # Compile the graph
        return workflow.compile()
    
    async def _analyze_contact(self, contact: ContactInfo, contact_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze a single contact."""
        try:
            async with self.semaphore:
                # Initialize state for the workflow
                state = {
                    "contact": contact.dict(),
                    "contact_data": contact_data,
                    "messages": [],
                    "search_results": [],
                    "analysis": None,
                    "errors": [],
                    "complete": False
                }
                
                # Execute the workflow using the updated API
                result = await self.workflow.ainvoke(state)
                
                # Extract analysis from result
                analysis_dict = result.get("analysis")
                if analysis_dict:
                    return AnalysisResult(**analysis_dict)
                
                # If no analysis was generated
                return AnalysisResult(
                    contact_id=contact.name,
                    personality_analysis="Analysis failed",
                    conversation_style="Unknown",
                    error="Workflow did not produce an analysis"
                )
                
        except Exception as e:
            logger.error(f"Error analyzing contact {contact.name}: {str(e)}")
            return AnalysisResult(
                contact_id=contact.name,
                personality_analysis="Analysis failed",
                conversation_style="Unknown",
                error=str(e)
            )
    
    async def analyze_personalities(self, contacts_df: pd.DataFrame) -> Dict[str, AnalysisResult]:
        """
        Analyze personalities for multiple contacts in parallel.
        
        Args:
            contacts_df: DataFrame with contact information
            
        Returns:
            Dictionary mapping contact ID to analysis results
        """
        logger.info(f"Starting personality analysis for {len(contacts_df)} contacts")
        
        # Validate that we have a DataFrame with at least some columns
        if contacts_df is None or contacts_df.empty:
            logger.error("Empty or None DataFrame provided to analyze_personalities")
            return {}
        
        # Process each row in the DataFrame
        tasks = []
        for idx, row in contacts_df.iterrows():
            try:
                # Convert row to dictionary
                row_data = row.to_dict()
                
                # Use the from_row class method to create ContactInfo with proper fallbacks
                contact = ContactInfo.from_row(row_data)
                
                # Log the contact being processed
                logger.info(f"Processing contact: {contact.name}")
                
                # Create task
                tasks.append(self._analyze_contact(contact, row_data))
                
            except Exception as e:
                # Handle individual row processing errors
                logger.error(f"Error preparing contact at row {idx}: {str(e)}")
                # Create a minimal failure result
                contact_id = row.get('name', f"Unknown-{idx}")
                tasks.append(asyncio.sleep(0, result=AnalysisResult(
                    contact_id=contact_id,
                    personality_analysis="Analysis preparation failed",
                    conversation_style="Unknown",
                    error=f"Error preparing contact data: {str(e)}"
                )))
        
        # Execute tasks with graceful error handling
        results = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Critical error during personality analysis: {str(e)}")
        
        # Process results and handle exceptions
        result_dict = {}
        for i, result in enumerate(results):
            try:
                # Check if the result is an exception
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {str(result)}")
                    # Create dummy result using index as fallback ID
                    result = AnalysisResult(
                        contact_id=f"Failed-{i}",
                        personality_analysis="Analysis failed",
                        conversation_style="Unknown",
                        error=f"Exception during analysis: {str(result)}"
                    )
                
                # Add to result dictionary
                result_dict[result.contact_id] = result
                
            except Exception as e:
                logger.error(f"Error processing result {i}: {str(e)}")
        
        logger.info(f"Completed personality analysis with {len(result_dict)} results")
        return result_dict

# Function for integration with the main application
async def analyze_personality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze personalities for contacts in the provided DataFrame
    
    Args:
        df: DataFrame with contact data
        
    Returns:
        DataFrame with added personality analysis information
    """
    try:
        # Create the analyzer
        analyzer = PersonalityAnalyzer()
        
        # Test the API key first
        logger.info("Testing Anthropic API key before proceeding with analysis")
        key_valid = await analyzer.test_anthropic_api_key()
        
        if not key_valid:
            logger.error("Anthropic API key validation failed. Analysis cannot proceed.")
            # Add error columns to DataFrame
            df['personality_analysis'] = "API key validation failed"
            df['conversation_style'] = "Error"
            df['professional_interests'] = "Error"
            return df
            
        logger.info("Anthropic API key validation successful, proceeding with analysis")
        
        # Analyze personalities
        results = await analyzer.analyze_personalities(df)
        
        # Add results to the DataFrame
        df['personality_analysis'] = df.apply(
            lambda row: results.get(row['name'], AnalysisResult(
                contact_id=row['name'],
                personality_analysis="Analysis not available",
                conversation_style="Unknown"
            )).personality_analysis,
            axis=1
        )
        
        df['conversation_style'] = df.apply(
            lambda row: results.get(row['name'], AnalysisResult(
                contact_id=row['name'],
                personality_analysis="Analysis not available",
                conversation_style="Unknown"
            )).conversation_style,
            axis=1
        )
        
        df['professional_interests'] = df.apply(
            lambda row: ', '.join(results.get(row['name'], AnalysisResult(
                contact_id=row['name'],
                personality_analysis="Analysis not available",
                conversation_style="Unknown",
                professional_interests=[]
            )).professional_interests),
            axis=1
        )
        
        return df
    
    except Exception as e:
        logging.error(f"Error in personality analysis: {str(e)}")
        # Return original DataFrame if analysis fails
        return df


# Example usage
async def main():
    """Example usage of the PersonalityAnalyzer."""
    # Check for required API keys
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    if not tavily_api_key:
        print("Error: TAVILY_API_KEY environment variable not set")
        return
    
    # Create sample contact data
    sample_data = {
        'name': ['Samantha Tech', 'Michael Business'],
        'email': ['samantha@techcorp.com', 'michael@bizgroup.com'],
        'linkedin_url': ['https://linkedin.com/in/samantha-tech', 'https://linkedin.com/in/michael-business'],
        'company': ['TechCorp', 'BizGroup'],
        'title': ['CTO', 'Director of Sales']
    }
    
    contacts_df = pd.DataFrame(sample_data)
    
    print("Creating personality analyzer...")
    
    # Create the analyzer
    analyzer = PersonalityAnalyzer(
        anthropic_api_key=anthropic_api_key,
        tavily_api_key=tavily_api_key
    )
    
    print(f"Analyzing {len(contacts_df)} contacts...")
    
    # Analyze personalities
    results = await analyzer.analyze_personalities(contacts_df)
    
    # Print results (truncated for readability)
    for contact_id, result in results.items():
        print(f"\n\n=== Analysis for {contact_id} ===")
        print(f"Conversation Style: {result.conversation_style}")
        
        # Print truncated personality analysis
        analysis_text = result.personality_analysis
        if len(analysis_text) > 300:
            analysis_text = analysis_text[:300] + "..."
        print(f"\nPersonality Analysis: {analysis_text}")
        
        print("\nProfessional Interests:")
        for interest in result.professional_interests:
            print(f"- {interest}")
        
        if result.error:
            print(f"\nErrors: {result.error}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 
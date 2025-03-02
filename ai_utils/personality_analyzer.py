"""
Personality Analyzer Agent

This module implements an agent that analyzes a person's personality based on 
their online presence and LinkedIn profile. It uses LangGraph for workflow
orchestration and OpenRouter for analysis.

Usage:
    analyzer = PersonalityAnalyzer(openrouter_api_key, tavily_api_key)
    results = await analyzer.analyze_personalities(contacts_df)

Requirements:
    - langchain
    - langgraph
    - langchain_openai
    - tavily_python
    - pandas
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence
import pandas as pd
from pydantic import BaseModel, Field, SecretStr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import json
import re

# Import dotenv and load environment variables
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Custom OpenRouter integration with LangChain
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )

# Updated imports for the current LangGraph API
from langgraph.graph import StateGraph, END

# Use the centralized logger
logger = logging.getLogger(__name__)

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
        
        # Check for full_name first (for combined first/last name fields)
        if "full_name" in data and data["full_name"]:
            data["name"] = data["full_name"]
        # If name is missing, try to generate one from other fields or use a placeholder
        elif "name" not in data or not data["name"]:
            # Check for first_name and last_name fields to combine them
            if "first_name" in data and "last_name" in data and data["first_name"] and data["last_name"]:
                data["name"] = f"{data['first_name']} {data['last_name']}".strip()
            elif "email" in data and data["email"]:
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
    purchasing_behavior: Optional[str] = Field(default="")
    error: Optional[str] = None
    search_queries_used: List[str] = Field(default_factory=list)
    search_results: List[Dict] = Field(default_factory=list)

class PersonalityState(TypedDict):
    """State maintained throughout the analysis process."""
    contact: Dict[str, Any]  # Changed from ContactInfo to Dict for simpler serialization
    contact_data: Dict[str, Any]
    company_context: Dict[str, Any]  # Added company context
    messages: List[Dict]
    search_queries: List[str]
    search_results: List[Dict]
    analysis: Optional[Dict[str, Any]]  # Changed from AnalysisResult to Dict
    errors: List[str]
    complete: bool

class PersonalityAnalyzer:
    """Analyzes personality based on online presence."""
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 tavily_api_key: Optional[str] = None,
                 model_name: str = "anthropic/claude-3.5-haiku-20241022:beta",
                 max_concurrent: int = 10):
        """
        Initialize the personality analyzer.
        
        Args:
            openrouter_api_key: API key for OpenRouter
            tavily_api_key: API key for Tavily search
            model_name: Model to use for analysis (defaults to claude-3.7-sonnet)
            max_concurrent: Maximum number of concurrent analyses (used for search operations)
        """
        self.openrouter_api_key = openrouter_api_key
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        
        # Initialize search cache - shared across all instances
        if not hasattr(PersonalityAnalyzer, '_global_search_cache'):
            PersonalityAnalyzer._global_search_cache = {}
        self._search_cache = PersonalityAnalyzer._global_search_cache
        
        # Set up the search tool
        if tavily_api_key:
            self.search_tool = TavilySearchResults(api_key=tavily_api_key)
        else:
            raise ValueError("Tavily API key is required")
        
        # Set up the LLM
        if openrouter_api_key:
            self.llm = ChatOpenRouter(
                openai_api_key=openrouter_api_key,
                model=model_name,
                temperature=0.1,
                streaming=False
            )
        else:
            raise ValueError("OpenRouter API key is required")
        
        # Build the workflow
        self.workflow = self._build_workflow()
    
    def enable_tracing(self, project_name="email-gtmwiz-personality-analysis"):
        """
        Enable LangSmith tracing for this analyzer instance.
        
        Args:
            project_name: The project name to use in LangSmith
            
        Returns:
            bool: True if tracing was enabled, False otherwise
        """
        # Check if LANGCHAIN_API_KEY is set
        langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")
        if not langchain_api_key:
            logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing will not be enabled.")
            return False
        
        # Enable LangSmith tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        
        # Rebuild the workflow to ensure it uses the new tracing settings
        self.workflow = self._build_workflow()
        
        logger.info(f"LangSmith tracing enabled for project: {project_name}")
        return True
    
    async def test_openrouter_api_key(self) -> bool:
        """
        Test if the OpenRouter API key is valid by making a simple request.
        
        Returns:
            bool: True if the key is valid, False otherwise
        """
        try:
            logger.debug("Testing OpenRouter API key with a simple request")
            
            # Log key format details (safely)
            if self.openrouter_api_key:
                key_preview = self.openrouter_api_key[:10] + "..." if len(self.openrouter_api_key) > 10 else "[empty]"
                logger.debug(f"API key being tested: {key_preview}")
                logger.debug(f"API key length: {len(self.openrouter_api_key)}")
                
                # Check for common issues
                if '"' in self.openrouter_api_key:
                    logger.warning("API key contains quote characters - this may cause authentication issues")
                if ' ' in self.openrouter_api_key:
                    logger.warning("API key contains spaces - this may cause authentication issues")
            else:
                logger.error("No API key available for testing")
                return False
            
            # Create a direct test with minimal dependencies
            try:
                # Note: There's no direct OpenRouter client library like anthropic
                # So we'll skip this test and rely on the LangChain test
                logger.debug("Skipping direct client test as there's no official OpenRouter client library")
            except Exception as e:
                logger.error(f"Direct client test failed: {str(e)}")
            
            # Test with LangChain's ChatOpenRouter
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Respond with 'API key is working' if you can see this message.")
            ]
            
            # Make a simple request
            logger.debug("Sending test request to ChatOpenRouter")
            response = self.llm.invoke(messages)
            
            # Check if we got a valid response
            if response and hasattr(response, 'content') and response.content:
                logger.debug(f"OpenRouter API test successful, received response: {response.content[:50]}...")
                return True
            else:
                logger.error("OpenRouter API test failed: Received empty response")
                return False
                
        except Exception as e:
            logger.error(f"OpenRouter API test failed with error: {str(e)}")
            logger.exception("Detailed exception:")
            return False
    
    def _planning_task(self, state: PersonalityState) -> PersonalityState:
        """Plan what information to search for based on contact info and company context."""
        contact_data = state["contact_data"]
        company_context = state.get("company_context", {})
        
        # Check if website content is missing
        has_website_content = contact_data.get("website_content", "").strip() != ""
        
        # Prepare company context text separately
        company_context_text = ""
        if company_context:
            company_context_text = """
My Company Context:
Below is information about my company, our target market, and the specific problems we solve.
When creating the personality analysis, you MUST directly connect the contact's pain points 
to the specific problems our company solves. The Route-Ruin-Multiply analysis should explicitly 
show how our solutions address their challenges:

"""
            company_context_text += json.dumps(company_context, indent=2)
        
        # Create context for the LLM with strict JSON formatting instructions
        prompt = f"""
        <search_query_generation>
            <context>
                I need to analyze the personality of the following person to help with sales outreach from my company:
                
                Contact Data:
                {contact_data}
                
                {"NOTE: This contact does not have any website content available. Please create search queries based on their name, company, and role only." if not has_website_content else ""}
                
                {company_context_text}
            </context>
            
            <objective>
                Your task is to create EXACTLY 2 highly targeted search queries that will help me understand:
                1. Their communication style, professional background, and the specific PAIN POINTS they are likely experiencing in their role
                2. Their professional interests, values, and how they typically respond to sales approaches
            </objective>
            
            <search_strategy>
                - Create ONLY 2 queries - quality over quantity
                - Make each query specific and information-rich
                - Include their name, company, and role in each query
                - IMPORTANT: Include terms that will find EVIDENCE of their actual behavior (LinkedIn activity, writing style, speaking engagements)
                - Focus on identifying challenges, struggles, or "pain points" they might have that our solution could address
                - Include industry-specific problems or objections they might raise based on their role
                - Consider how our company's context relates to the specific problems they're trying to solve
                - Include terms to uncover industry benchmarks and metrics relevant to their role
            </search_strategy>
            
            <response_format>
                You must respond with ONLY a valid JSON array of strings. Each string should be a search query.
                
                Example of the EXACT format required:
                ```json
                [
                  "John Smith Microsoft CEO leadership style communication preferences LinkedIn activity speaking engagements writing samples",
                  "John Smith Microsoft decision-making process pain points enterprise transformation challenges vendor evaluation criteria"
                ]
                ```
                
                Do not include any explanations, notes, or other text outside the JSON array. The response must be a valid JSON array that can be parsed directly.
            </response_format>
        </search_query_generation>
        """
        
        try:
            # Get search queries from LLM with strict system message
            messages = [
                SystemMessage(content="You are a helpful assistant that creates search queries to gather information about professional contacts. You MUST respond with ONLY a valid JSON array of strings, with no additional text."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract search queries from LLM response
            search_queries = []
            
            # First try to parse the entire response as JSON
            try:
                # Strip any markdown code block indicators
                clean_response = re.sub(r'```json|```', '', response.content).strip()
                search_queries = json.loads(clean_response)
                logger.debug(f"Successfully parsed entire response as JSON: {search_queries}")
            except json.JSONDecodeError:
                # If that fails, look for JSON array in the response
                logger.debug("Failed to parse entire response as JSON, looking for JSON array")
                json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                if json_match:
                    queries_json = json_match.group(0)
                    try:
                        search_queries = json.loads(queries_json)
                        logger.debug(f"Successfully parsed JSON array from response: {search_queries}")
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON array from response")
                        # Fallback to basic extraction
                        lines = response.content.strip().split('\n')
                        search_queries = [line.strip() for line in lines if line.strip() and not line.startswith('```')]
                else:
                    # Fallback to basic extraction if no JSON array found
                    logger.error("No JSON array found in response, falling back to basic extraction")
                    lines = response.content.strip().split('\n')
                    search_queries = [line.strip() for line in lines if line.strip() and not line.startswith('```')]
            
            # Ensure all queries are strings
            search_queries = [str(query) for query in search_queries]
            
            # Add the queries to the messages
            state["messages"].append({
                "role": "system",
                "content": f"Generated search queries: {search_queries}"
            })
            
            # Store search queries in state - limit to 2 queries
            state["search_queries"] = search_queries[:2]  # Limit to 2 queries
            
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
        
        # Deduplicate search queries to avoid redundant searches
        unique_queries = list(set(search_queries))
        logger.debug(f"Reduced {len(search_queries)} queries to {len(unique_queries)} unique queries")
        
        search_results = []
        
        # Execute searches in parallel with improved caching
        async def execute_search(query: str):
            try:
                # Normalize query to improve cache hits
                query_str = str(query).strip().lower()
                
                # Check if we have a cached result for this query
                if query_str in self._search_cache:
                    logger.debug(f"Using cached result for query: {query_str}")
                    return self._search_cache[query_str]
                
                logger.debug(f"Executing new search for query: {query_str}")
                # Execute the search
                result = await asyncio.to_thread(self.search_tool.invoke, query_str)
                
                # Cache the result
                self._search_cache[query_str] = {"query": query, "results": result}
                
                return {"query": query, "results": result}
            except Exception as e:
                logger.error(f"Search error for query '{query}': {str(e)}")
                return {"query": str(query), "results": [], "error": str(e)}
        
        # Run searches concurrently with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Reduce to 5 concurrent searches for better stability
        
        async def search_with_semaphore(query):
            async with semaphore:
                return await execute_search(query)
        
        # Create tasks for each unique query
        tasks = [search_with_semaphore(query) for query in unique_queries]
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
        """Analyze search results to generate personality insights relevant to company context."""
        contact = state["contact"]
        contact_data = state["contact_data"]
        search_results = state["search_results"]
        company_context = state.get("company_context", {})
        
        contact_name = contact.get("name", "Unknown")
        has_website_content = contact_data.get("website_content", "").strip() != ""
        
        if not search_results:
            # Create a more informative message based on whether website content was available
            if not has_website_content:
                analysis_message = "No website content was provided for this contact. Analysis is based solely on search results."
            else:
                analysis_message = "Insufficient data for analysis. No search results were found."
                
            state["errors"].append("No search results available for analysis")
            state["analysis"] = {
                "contact_id": contact_name,
                "personality_analysis": analysis_message,
                "conversation_style": "Unknown",
                "error": "No search results available",
                "professional_interests": [],
                "search_queries_used": [],
                "search_results": []
            }
            state["complete"] = True
            return state
        
        # Prepare search results for prompt - more efficiently
        results_text = ""
        for result in search_results:
            query_str = result.get('query', '')
            results_text += f"\nQuery: {query_str}\n"
            
            # Limit to top 3 results per query for efficiency
            for i, item in enumerate(result['results'][:3]):
                results_text += f"Result {i+1}: {item.get('title', 'No title')}\n"
                
                # Extract only the most relevant part of the content (first 300 chars)
                content = item.get('content', 'No content')
                results_text += f"Content: {content[:300]}...\n\n"
        
        # Prepare company context text separately
        company_context_text = ""
        if company_context:
            company_context_text = """
My Company Context:
Below is information about my company, our target market, and the specific problems we solve.
When creating the personality analysis, you MUST directly connect the contact's pain points 
to the specific problems our company solves. The Route-Ruin-Multiply analysis should explicitly 
show how our solutions address their challenges:

"""
            company_context_text += json.dumps(company_context, indent=2)
        
        # Create analysis prompt without JSON output instructions
        prompt = f"""
        <personality_analysis_request>
            <context>
                Based on the following information about the contact, please create a concise personality analysis that would help me personalize my outreach from my company.

                Contact Information:
                {contact_data}
                
                {company_context_text}
                
                {"NOTE: This contact does not have any website content available. The analysis is based solely on search results." if not has_website_content else ""}
                
                Search Results:
                {results_text}
            </context>
            
            <analysis_framework>
                <personality_analysis>
                    - A detailed analysis of their specific personality traits, behavioral patterns, and psychological drivers
                    - For each insight, clearly indicate CONFIDENCE LEVEL (High/Medium/Low) and the source of evidence
                    - Example: "Shows a preference for direct communication [HIGH CONFIDENCE: based on LinkedIn writing style]"
                    - Explore both analytical AND emotional aspects of their personality - what motivates them beyond just efficiency?
                    - What career aspirations and professional growth goals might they have in their current role?
                    - Note any unique characteristics that differentiate them from others in similar roles
                    - For each inferred trait, provide ONE specific validation question a salesperson could ask to confirm
                </personality_analysis>

                <conversation_style>
                    - How they specifically prefer to communicate (direct, analytical, relational, etc.)
                    - Provide 2-3 exact conversational phrases or questions that would resonate with this person
                    - Suggest a tailored email subject line that would catch their attention
                    - Specify the ideal communication channel, timing, and frequency based on their personality
                    - Provide ALTERNATIVE approaches for different possible communication preferences (since these are often inferred)
                </conversation_style>

                <professional_interests>
                    - A list of 3-5 professional topics they seem most interested in
                    - For each interest, note how it connects to their specific role and company situation
                    - Identify any thought leaders or resources they might follow related to these interests
                    - Consider both functional AND aspirational interests - what would help them grow professionally?
                    - Include specific industry metrics and benchmarks relevant to each interest area
                </professional_interests>

                <company_context>
                    - Analyze the company's likely growth stage (startup, scaling, mature) based on available evidence
                    - Estimate their team size and organizational structure
                    - Identify industry-specific challenges unique to their business sector
                    - Note any recent news, funding, or market conditions that might affect their priorities
                    - Consider their competitive landscape and how it impacts their decision-making
                    - Identify potential internal champions/stakeholders who might influence decisions
                </company_context>

                <pain_points>
                    - Identify 2-3 specific problems or challenges this person is likely facing in their role
                    - For each pain point, include:
                      * Specific evidence or inference source [with confidence level]
                      * Industry-specific metrics/benchmarks related to this challenge
                      * Concrete manifestations of this pain point in their specific role and context
                      * Observable indicators a salesperson could look for in conversation
                      * How our company's offerings specifically address this pain point
                      * 1-2 validation questions to confirm if this pain point exists
                </pain_points>

                <purchasing_behavior>
                    - Analyze how this specific individual likely makes purchasing decisions
                    - Consider both rational factors AND emotional/career motivations that influence their buying process
                    - What metrics or outcomes would most influence their buying process?
                    - Note their likely research process before making decisions (demos, case studies, testimonials, etc.)
                    - Specify potential budget constraints or approval processes based on their role and company
                    - How does their career trajectory impact their purchasing priorities?
                </purchasing_behavior>

                <objections>
                    - Based on the JM "Ack-Peel" methodology, predict 1-2 likely objections specific to this person
                    - Include BOTH practical/technical objections AND personal/career-related hesitations
                    - Suggest response approaches that acknowledge their concern before asking an open question
                    - Include specific industry data or metrics to address each objection
                    - Tailor the objection handling to their specific personality type and company situation
                </objections>

                <personalized_approach>
                    - Route: Identify the specific decision-maker position/role this person occupies and their sphere of influence. Include their relationships with other stakeholders based on company size and structure.
                    - Ruin: Analyze specific gaps in their current approach and show how these directly relate to problems OUR COMPANY solves. Reference their company's specific situation, growth stage, and industry challenges.
                    - Multiply: Determine a natural progression path that would help them evaluate our solution with specific reference to our company's capabilities and their company's particular needs. Include how our solution helps their personal career trajectory.
                    - Quick Wins: Suggest 1-2 specific items that could establish value in the first 1-2 weeks of engagement
                </personalized_approach>
            </analysis_framework>
            
            <guidance>
                When insufficient information exists about the prospect's company, make reasonable inferences based on:
                - The typical size/structure of companies with their job title
                - Common challenges in their industry sector
                - Standard growth patterns for their type of business
                - Typical career paths for professionals in their role
                
                However, clearly indicate when you are making such inferences rather than stating known facts.
                
                REMEMBER: Focus equally on both PRACTICAL/TECHNICAL aspects AND EMOTIONAL/CAREER motivations throughout your analysis.
            </guidance>
        </personality_analysis_request>
        """
        
        try:
            # Get analysis from LLM
            messages = [
                SystemMessage(content="You are an expert sales coach and personality analyst who specializes in the JM methodology for sales outreach. You help sales professionals personalize their approach based on prospect personalities and pain points. Your analysis should be practical, focused on actionable insights, and designed to help salespeople connect with prospects by addressing their specific problems with the company's solutions. Be specific and concrete - avoid generic advice. Always link insights to the prospect's particular role, company stage, industry, and personality traits. Balance technical/practical analysis with emotional and career motivations - people buy for both rational AND emotional reasons."),
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
            
            # Extract professional interests using regex
            interests = []
            interests_section = re.search(r'Professional Interests:.*?(\n\n|\Z)', response.content, re.DOTALL)
            if interests_section:
                interest_text = interests_section.group(0)
                # Look for list items (- or • or 1. format)
                interest_items = re.findall(r'(?:[-•*]|\d+\.)\s+(.*?)(?=\n[-•*]|\n\d+\.|\n\n|\Z)', interest_text, re.DOTALL)
                interests = [item.strip() for item in interest_items if item.strip()]
            
            # Extract conversation style using regex
            style = "Unknown"
            style_section = re.search(r'Conversation Style:(.*?)(?=\n\n|\n\d\.|\Z)', response.content, re.DOTALL)
            if style_section:
                style = style_section.group(1).strip()
            
            # Extract purchasing behavior using regex
            purchasing_behavior = ""
            purchasing_section = re.search(r'Purchasing Behavior Analysis:(.*?)(?=\n\n|\n\d\.|\Z)', response.content, re.DOTALL)
            if purchasing_section:
                purchasing_behavior = purchasing_section.group(1).strip()
            
            # Create analysis result - ensure search_queries_used contains strings only
            analysis = {
                "contact_id": contact_name,
                "personality_analysis": response.content,
                "conversation_style": style,
                "purchasing_behavior": purchasing_behavior,
                "professional_interests": interests,
                "search_queries_used": [str(r.get("query", "")) for r in search_results],  # Convert to string to ensure compatibility
                "search_results": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", "")
                    } for result in search_results for item in result.get("results", [])[:3]  # Limit to top 3 results per query
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
        
        # Enable tracing for the workflow if LangSmith is configured
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
            logger.info("LangSmith tracing enabled for personality analysis workflow")
            # The compiled workflow will automatically use LangSmith tracing
        
        # Compile the graph
        return workflow.compile()
    
    async def _analyze_contact(self, contact: ContactInfo, contact_data: Dict[str, Any], company_context: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analyze a single contact."""
        try:
            # Initialize state for the workflow
            state = {
                "contact": contact.dict(),
                "contact_data": contact_data,
                "company_context": company_context or {},
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
    
    async def analyze_personalities_with_contacts(self, contacts: List[ContactInfo], company_context: Optional[Dict[str, Any]] = None) -> Dict[str, AnalysisResult]:
        """
        Analyze personalities for multiple contacts in parallel using pre-generated ContactInfo objects.
        
        Args:
            contacts: List of ContactInfo objects
            company_context: Dictionary containing information about the user's company
            
        Returns:
            Dictionary mapping contact ID to analysis results
        """
        logger.info(f"Starting personality analysis for {len(contacts)} contacts")
        
        # Validate that we have a list of ContactInfo objects
        if not contacts:
            logger.warning("No contacts provided for analysis")
            return {}
        
        # Count contacts with website content for logging
        contacts_with_content = sum(1 for contact in contacts if contact.website_content)
        logger.info(f"Of {len(contacts)} contacts, {contacts_with_content} have website content")
        
        # Set up a semaphore to limit concurrent API calls to 15
        semaphore = asyncio.Semaphore(15)
        result_dict = {}
        
        # Create a function to process a single contact with the semaphore
        async def process_contact_with_semaphore(contact):
            async with semaphore:
                # Extract relevant data for the analysis
                contact_data = {
                    "name": contact.name,
                    "company": contact.company or "",
                    "title": contact.title or "",
                    "linkedin_url": contact.linkedin_url or "",
                    "twitter_url": contact.twitter_url or "",
                    "personal_website": contact.personal_website or "",
                    "website_content": contact.website_content[:1000] if contact.website_content else ""  # Limit website content
                }
                
                # Analyze the contact with company context
                return await self._analyze_contact(contact, contact_data, company_context)
        
        # Create tasks for all contacts
        tasks = [process_contact_with_semaphore(contact) for contact in contacts]
        
        # Process all contacts and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            try:
                # Handle exceptions
                if isinstance(result, Exception):
                    logger.error(f"Error in analysis task: {str(result)}")
                    contact = contacts[i]
                    result = AnalysisResult(
                        contact_id=contact.name,
                        personality_analysis="Analysis failed",
                        conversation_style="Unknown",
                        error=str(result)
                    )
                
                # Add to result dictionary
                result_dict[result.contact_id] = result
                
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
        
        logger.info(f"Completed personality analysis with {len(result_dict)} results")
        return result_dict
        
    async def analyze_personalities(self, contacts_df: pd.DataFrame) -> Dict[str, AnalysisResult]:
        """
        Analyze personalities for multiple contacts in parallel.
        
        Args:
            contacts_df: DataFrame with contact information
            
        Returns:
            Dictionary mapping contact ID to analysis results
        """
        # Convert DataFrame rows to ContactInfo objects
        contacts = []
        for _, row in contacts_df.iterrows():
            contact = ContactInfo.from_row(row.to_dict())
            contacts.append(contact)
            
        # Use the new method that works with pre-generated ContactInfo objects
        return await self.analyze_personalities_with_contacts(contacts)

# Function for integration with the main application
async def analyze_personality(df: pd.DataFrame, model_name: str = "anthropic/claude-3.5-haiku-20241022:beta", company_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Analyze personalities for contacts in a DataFrame.
    
    Args:
        df: DataFrame with contact information
        model_name: Name of the model to use for analysis (default: "anthropic/claude-3.5-haiku-20241022:beta")
        company_context: Dictionary containing information about the user's company
        
    Returns:
        DataFrame with personality analysis results
    """
    # First, validate the OpenRouter API key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    
    if not openrouter_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        raise ValueError("OpenRouter API key is required for personality analysis")
    
    if not tavily_key:
        logger.error("TAVILY_API_KEY not found in environment")
        raise ValueError("Tavily API key is required for personality analysis")
    
    # Create analyzer with the OpenRouter API key and specified model
    analyzer = PersonalityAnalyzer(
        openrouter_api_key=openrouter_key,
        tavily_api_key=tavily_key,
        model_name=model_name
    )
    
    # Enable LangSmith tracing if available
    analyzer.enable_tracing()
    
    # Run analysis
    try:
        logger.info(f"Starting personality analysis for {len(df)} contacts")
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Add personality analysis columns if they don't exist
        if 'personality_analysis' not in result_df.columns:
            result_df['personality_analysis'] = ""
        if 'conversation_style' not in result_df.columns:
            result_df['conversation_style'] = ""
        if 'professional_interests' not in result_df.columns:
            result_df['professional_interests'] = ""
        
        # Add company_context as a column in the dataframe
        if company_context:
            # Create a simplified version of company context for the dataframe
            simplified_context = {
                'name': company_context.get('name', ''),
                'description': company_context.get('description', ''),
                'target_geography': company_context.get('target_geography', 'Global')
            }
            # Convert to JSON string for storage in dataframe
            context_json = json.dumps(simplified_context)
            result_df['company_context'] = context_json
            logger.info(f"Added company context to dataframe: {simplified_context['name']}")
        else:
            # Add empty context if none provided
            result_df['company_context'] = ""
            logger.info("No company context provided, added empty context column")
        
        # Pre-generate all contact info objects and store their IDs
        # This ensures we use the exact same contact IDs during analysis and when updating results
        contact_infos = []
        row_indices = []
        contacts_with_content = 0
        
        # Process all rows, but keep track of which ones have website content
        for idx, row in df.iterrows():
            contact_info = ContactInfo.from_row(row.to_dict())
            contact_infos.append(contact_info)
            row_indices.append(idx)
            
            # Count how many contacts have website content (for logging purposes)
            if 'website_content' in row and row['website_content']:
                contacts_with_content += 1
        
        if not contact_infos:
            logger.warning("No contacts found for analysis")
            return result_df
            
        logger.info(f"Analyzing {len(contact_infos)} contacts (of which {contacts_with_content} have website content)")
        
        # Pass the pre-generated contact info objects to the analyzer with company context
        results = await analyzer.analyze_personalities_with_contacts(contact_infos, company_context=company_context)
        
        # Create a mapping of contact IDs to row indices for more reliable matching
        contact_id_to_index = {}
        for i, contact_info in enumerate(contact_infos):
            contact_id_to_index[contact_info.name] = row_indices[i]
        
        # Update the DataFrame with analysis results
        for contact_id, analysis_result in results.items():
            try:
                # Check if we have this contact ID in our mapping
                if contact_id in contact_id_to_index:
                    idx = contact_id_to_index[contact_id]
                    
                    # Update the row with analysis results
                    result_df.at[idx, 'personality_analysis'] = analysis_result.personality_analysis
                    result_df.at[idx, 'conversation_style'] = analysis_result.conversation_style
                    
                    # Ensure professional_interests is a list of strings before joining
                    interests = analysis_result.professional_interests
                    if isinstance(interests, list):
                        # Make sure all items are strings
                        interests = [str(item) for item in interests]
                        result_df.at[idx, 'professional_interests'] = ', '.join(interests)
                    else:
                        # Handle case where it's not a list
                        result_df.at[idx, 'professional_interests'] = str(interests)
                else:
                    # Log that we couldn't find a matching row
                    logger.warning(f"No matching row found for contact ID: {contact_id}")
            except Exception as e:
                logger.error(f"Error updating results for {contact_id}: {str(e)}")
        
        # Apply a lambda function to handle any rows that weren't analyzed
        result_df['professional_interests'] = result_df.apply(
            lambda row: row['professional_interests'] if row['professional_interests'] else "",
            axis=1
        )
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in personality analysis: {str(e)}")
        logger.exception("Detailed exception:")
        # Return the original DataFrame if analysis fails
        return df 
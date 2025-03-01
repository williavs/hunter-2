"""
Company Context Workflow

This module implements a simple workflow that analyzes a company's website
to generate a concise context summary for use in personality analysis.

Usage:
    analyzer = CompanyContextAnalyzer(openrouter_api_key, tavily_api_key)
    company_context = await analyzer.analyze_company("https://example.com")

Requirements:
    - langchain
    - langgraph
    - langchain_openai
    - tavily_python
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import re

# Import the CompanyScraper for direct website scraping
from utils.company_scraper import CompanyScraper

# Import dotenv and load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import LangGraph components
from langgraph.graph import StateGraph, END

# Import pydantic components
from pydantic import BaseModel, Field, SecretStr

# Use the centralized logger
logger = logging.getLogger(__name__)

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

class CompanyState(TypedDict):
    """State maintained throughout the company analysis process."""
    company_url: str
    search_queries: List[str]
    search_results: List[Dict]
    scraped_content: Optional[str]  # Added to store scraped website content
    company_context: Optional[Dict[str, Any]]
    errors: List[str]
    complete: bool

class CompanyContextAnalyzer:
    """Analyzes a company's website to generate context for personality analysis."""
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 tavily_api_key: Optional[str] = None,
                 model_name: str = "anthropic/claude-3.5-haiku-20241022:beta"):
        """
        Initialize the company context analyzer.
        
        Args:
            openrouter_api_key: API key for OpenRouter
            tavily_api_key: API key for Tavily search
            model_name: Model to use for analysis
        """
        self.openrouter_api_key = openrouter_api_key
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
        
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
    
    def _planning_task(self, state: CompanyState) -> CompanyState:
        """Generate search queries to gather information about the company."""
        company_url = state["company_url"]
        
        # Create context for the LLM
        prompt = f"""
        I need to deep-dive research on a company with URL: {company_url} - and I need to UNLEARN everything I think I know about generic search queries.
        
        Your task is to create EXACTLY 3 hyper-specific, needle-moving search queries that will help me understand:
        1. The PAIN POINTS this company claims to solve - not just what they do, but what PROBLEMS they actually FIX for their customers
        2. Their exact GTM approach, sales methodology, and how they position against competitors - SPECIFICALLY focusing on their claims of differentiation and unique selling points
        3. Their TARGET MARKETS and GEOGRAPHIC FOCUS - specifically where they operate, which regions/countries they target, and how they adapt their solutions to different geographic markets
        
        THIS IS CRITICAL: 
        - Drop the corporate fluff - I need to know what keeps their BUYERS up at night
        - Extract the company name from the URL and include it with industry-specific jargon 
        - Use words like "challenges," "pain points," "struggles," "barriers," and "obstacles" in your queries
        - Include terms about their claimed competitive advantages, unique approach, or methodology
        - Focus ENTIRELY on what makes them different, what problems they solve, and where they operate - nothing else matters
        
        You must respond with ONLY a valid JSON array of strings. Each string should be a search query.
        
        Example of the EXACT format required:
        ```json
        [
          "GTM Wizards B2B lead generation pain points challenges solved ICP development pipeline management go-to-market strategy",
          "GTM Wizards competitive advantage unique approach methodology differentiation vs traditional agencies market positioning",
          "GTM Wizards target markets geographic focus regional expansion international local adaptation strategy"
        ]
        ```
        
        Do not include any explanations, notes, or other text outside the JSON array. ONLY THE JSON ARRAY.
        """
        
        try:
            # Get search queries from LLM
            messages = [
                SystemMessage(content="You are a hyper-aggressive, no-BS sales intelligence researcher who cuts through the noise to find exactly what matters for sales conversations. You ONLY output valid JSON arrays - nothing else."),
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
            
            # Store search queries in state - limit to 3 queries
            state["search_queries"] = search_queries[:3]  # Updated to 3 queries
            
        except Exception as e:
            error_msg = f"Error in planning: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
            # Extract domain name for fallback query
            domain = company_url.replace("http://", "").replace("https://", "").split("/")[0]
            state["search_queries"] = [
                f"{domain} customer pain points problems solved", 
                f"{domain} competitive advantage unique approach",
                f"{domain} target markets geographic focus"
            ]
        
        return state
    
    async def _scrape_website_task(self, state: CompanyState) -> CompanyState:
        """Scrape content directly from the company website."""
        company_url = state["company_url"]
        
        try:
            # Initialize the company scraper
            scraper = CompanyScraper(company_url, max_pages=10)
            
            # Scrape the website
            logger.info(f"Scraping company website: {company_url}")
            await scraper.scrape_website()
            
            # Get combined content from important pages
            scraped_content = scraper.get_important_pages_content()
            
            # If no important pages were found, try to get all content
            if not scraped_content:
                scraped_content = scraper.get_combined_content()
            
            # Store the scraped content in the state
            state["scraped_content"] = scraped_content
            
            # Log how many pages were scraped
            logger.info(f"Successfully scraped {len(scraper.pages_content)} pages from {company_url}")
            
        except Exception as e:
            error_msg = f"Error scraping website {company_url}: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["scraped_content"] = f"Unable to scrape website content: {str(e)}"
        
        return state
    
    async def _search_task(self, state: CompanyState) -> CompanyState:
        """Execute searches for each query in parallel."""
        search_queries = state.get("search_queries", [])
        
        if not search_queries:
            state["errors"].append("No search queries available")
            return state
        
        search_results = []
        
        # Execute searches in parallel
        async def execute_search(query: str):
            try:
                logger.debug(f"Executing search for query: {query}")
                # Execute the search
                result = await asyncio.to_thread(self.search_tool.invoke, query)
                return {"query": query, "results": result}
            except Exception as e:
                logger.error(f"Search error for query '{query}': {str(e)}")
                return {"query": str(query), "results": [], "error": str(e)}
        
        # Create tasks for each query
        tasks = [execute_search(query) for query in search_queries]
        search_outputs = await asyncio.gather(*tasks)
        
        # Collect results
        for output in search_outputs:
            if "error" not in output:
                search_results.append(output)
        
        state["search_results"] = search_results
        
        return state
    
    def _analysis_task(self, state: CompanyState) -> CompanyState:
        """Analyze search results and scraped content to generate company context."""
        company_url = state["company_url"]
        search_results = state["search_results"]
        scraped_content = state.get("scraped_content", "")
        
        # Check if we have search results or scraped content
        has_search_results = len(search_results) > 0
        has_scraped_content = scraped_content and len(scraped_content) > 100  # At least 100 chars
        
        if not has_search_results and not has_scraped_content:
            state["errors"].append("No search results or website content available for analysis")
            state["company_context"] = {
                "name": "Unknown",
                "description": f"Unable to analyze company at {company_url}. No data found.",
                "url": company_url,
                "target_geography": "Unknown"
            }
            state["complete"] = True
            return state
        
        # Prepare search results for prompt
        results_text = ""
        if has_search_results:
            for result in search_results:
                query_str = result.get('query', '')
                results_text += f"\nQuery: {query_str}\n"
                
                # Limit to top 3 results per query for efficiency
                for i, item in enumerate(result['results'][:3]):
                    results_text += f"Result {i+1}: {item.get('title', 'No title')}\n"
                    
                    # Extract only the most relevant part of the content
                    content = item.get('content', 'No content')
                    results_text += f"Content: {content[:500]}...\n\n"
        
        # Prepare scraped content for prompt (if available)
        scraped_content_text = ""
        if has_scraped_content:
            # Limit to a reasonable size to avoid context window issues
            scraped_content_preview = scraped_content[:10000] if len(scraped_content) > 10000 else scraped_content
            scraped_content_text = f"\nWebsite Content:\n{scraped_content_preview}\n\n"
        
        # Create analysis prompt - completely avoid f-strings with conditionals containing newlines
        search_results_section = ""
        if has_search_results:
            search_results_section = "\nSearch Results:\n" + results_text
            
        website_content_section = ""
        if has_scraped_content:
            website_content_section = "\nWebsite Content:\n" + scraped_content_text
        
        # Check if we already have a user-specified target geography
        user_geography = None
        if "company_context" in state and state["company_context"] and "target_geography" in state["company_context"]:
            user_geography = state["company_context"]["target_geography"]
            logger.info(f"Found user-specified target geography in state: '{user_geography}'")
        
        # Add a note about the user-specified target geography if available
        user_geography_note = ""
        if user_geography:
            logger.info(f"Adding user-specified target geography to prompt: '{user_geography}'")
            user_geography_note = f"\n\nIMPORTANT: The user has specified their target geography as: {user_geography}. Your analysis MUST focus specifically on this market and discuss how their solutions address problems in this specific region."
        
        prompt = f"""
        SALES DEEP DIVE: Company at {company_url}
        
        SKIP THE FLUFF. As a killer sales professional, you need to understand this company so you can:
        1. Know EXACTLY what problems they solve
        2. Understand how they make money
        3. See through their marketing BS to find their TRUE differentiators
        4. Identify who ACTUALLY buys from them and why
        5. Determine WHERE they operate and how problems/solutions differ by region{user_geography_note}
{search_results_section}
{website_content_section}
        
        You're going to use the Route-Ruin-Multiply framework to create a KILLER sales-oriented company profile. Take time to THINK STEP BY STEP through:
        
        1. PAINS & SOLUTIONS: Identify 3-4 SPECIFIC pain points this company claims to solve. Not vague marketing speak - the actual business problems their customers have that keep decision-makers up at night. Pair each pain with how they specifically claim to solve it.
        
        2. DIFFERENTIATION: Analyze what truly makes them unique in their space - find the 2-3 things they do differently from competitors that customers actually care about. Cut through the BS.
        
        3. BUYER PROFILE: Who are their ACTUAL buyers? What titles, industries, and company sizes? What priorities and pressures do these buyers have?
        
        4. TARGET GEOGRAPHY: Identify their primary geographic target markets - where are they selling? Include both current regions and expansion targets. Explain how problems and solutions might differ by region (regulatory differences, market maturity, cultural factors).{user_geography_note}
        
        5. OBJECTION INSIGHTS: Based on their positioning, what are the likely 1-2 objections prospects raise when considering them? Think about cost, implementation time, competing priorities, and regional challenges.
        
        6. SALES APPROACH: How do they likely sell? Direct? Channel? Product-led? What's their conversion strategy and pricing model? Do they adapt their approach by region?
        
        Finally, synthesize all of this into ONE KILLER PARAGRAPH that captures the essence of this company as if you were explaining it to a CEO in an elevator. This is your only chance to explain what they do, who they serve, what problems they solve, how they're different, where they operate, and why anyone should care. Make every word count, kill the fluff, and add something truly insightful.
        
        ALSO PROVIDE: A concise one-line statement of their primary geographic focus (e.g., "North American enterprise market" or "Global with emphasis on EMEA financial sector").
        """
        
        try:
            # Get analysis from LLM
            messages = [
                SystemMessage(content=f"You are a battle-tested sales executive who cuts through corporate BS, finding exactly what matters for deal creation. You have the rare ability to see through marketing fluff to identify the actual business problems and unique value a company provides. Your analysis must be brutally direct, insightful, and focused on what actually drives buying decisions across different geographic markets.{user_geography_note}"),
                HumanMessage(content=prompt)
            ]
            
            # Add an additional system message with stronger emphasis on target geography if specified
            if user_geography:
                messages = [
                    SystemMessage(content=f"CRITICAL INSTRUCTION: The user has specified their target geography as: {user_geography}. You MUST focus your analysis on this specific market. Your responses MUST be tailored to {user_geography}-specific challenges, regulations, market conditions, and competitive dynamics. This is NOT optional - if you don't address {user_geography} specifically throughout your analysis, your response will be considered completely WRONG."),
                    SystemMessage(content=f"You are a battle-tested sales executive who cuts through corporate BS, finding exactly what matters for deal creation. You have the rare ability to see through marketing fluff to identify the actual business problems and unique value a company provides. Your analysis must be brutally direct, insightful, and focused on what actually drives buying decisions in {user_geography} specifically."),
                    HumanMessage(content=prompt)
                ]
            
            response = self.llm.invoke(messages)
            
            # Extract company name from URL or search results
            company_name = self._extract_company_name(company_url, search_results, response.content)
            
            # Extract target geography from response
            target_geography = user_geography if user_geography else "Global" # Default value
            if not user_geography:
                geography_match = re.search(r'(?:Primary geographic focus:|TARGET GEOGRAPHY:|Geographic focus:|Target market:)\s*([^\.]+)', response.content)
                if geography_match:
                    target_geography = geography_match.group(1).strip()
            
            # Create company context
            state["company_context"] = {
                "name": company_name,
                "description": response.content.strip(),
                "url": company_url,
                "target_geography": target_geography,
                "has_scraped_content": has_scraped_content
            }
            
            # If we have scraped content, add a preview to the context
            if has_scraped_content:
                state["company_context"]["website_content"] = scraped_content
            
        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["company_context"] = {
                "name": self._extract_domain(company_url),
                "description": f"Error analyzing company: {error_msg}",
                "url": company_url,
                "target_geography": "Unknown"
            }
        
        state["complete"] = True
        return state
    
    def _extract_company_name(self, url: str, search_results: List[Dict], analysis_text: str) -> str:
        """Extract company name from URL, search results, or analysis text."""
        # First try to extract from URL
        domain = self._extract_domain(url)
        
        # Remove common TLDs and split by dots or dashes
        name_parts = re.sub(r'\.com$|\.org$|\.net$|\.io$|\.co$', '', domain).split('.')[-1].split('-')
        
        # Capitalize each part
        company_name = ' '.join(part.capitalize() for part in name_parts)
        
        # Look for company name in analysis text with common patterns
        name_match = re.search(r'(^|\s)([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5})\s+is\s+a', analysis_text)
        if name_match:
            return name_match.group(2)
        
        name_match = re.search(r'(^|\s)([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5})\s+provides', analysis_text)
        if name_match:
            return name_match.group(2)
        
        name_match = re.search(r'(^|\s)([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5})\s+offers', analysis_text)
        if name_match:
            return name_match.group(2)
        
        return company_name
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        # Remove protocol
        domain = url.replace("http://", "").replace("https://", "")
        # Get the domain part (before any path)
        domain = domain.split("/")[0]
        # Remove www. if present
        domain = domain.replace("www.", "")
        return domain
    
    def _build_workflow(self):
        """Build the LangGraph workflow for company analysis."""
        # Create a state graph with the CompanyState type
        workflow = StateGraph(CompanyState)
        
        # Add nodes for each task
        workflow.add_node("planning", self._planning_task)
        workflow.add_node("scrape_website", self._scrape_website_task)
        workflow.add_node("search", self._search_task)
        workflow.add_node("analysis", self._analysis_task)
        
        # Define the edges to create a more comprehensive flow
        workflow.add_edge("planning", "scrape_website")
        workflow.add_edge("scrape_website", "search")
        workflow.add_edge("search", "analysis")
        workflow.add_edge("analysis", END)
        
        # Set the entry point
        workflow.set_entry_point("planning")
        
        # Enable tracing for the workflow if LangSmith is configured
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
            logger.info("LangSmith tracing enabled for company context workflow")
        
        # Compile the graph
        return workflow.compile()
    
    async def analyze_company(self, company_url: str, target_geography: str = None) -> Dict[str, Any]:
        """
        Analyze a company based on its website URL.
        
        Args:
            company_url: URL of the company website
            target_geography: User-specified target geography
            
        Returns:
            Dictionary containing company context information
        """
        try:
            # Initialize state for the workflow
            state = {
                "company_url": company_url,
                "search_queries": [],
                "search_results": [],
                "scraped_content": None,
                "company_context": None,
                "errors": [],
                "complete": False
            }
            
            # If target geography is provided, initialize company_context with it
            if target_geography:
                logger.info(f"Setting target geography in initial state: '{target_geography}'")
                state["company_context"] = {"target_geography": target_geography}
            
            # Execute the workflow
            logger.info(f"Starting company workflow for {company_url}")
            try:
                result = await self.workflow.ainvoke(state)
                logger.info(f"Workflow completed successfully for {company_url}")
                
                # Check for errors in the workflow
                if result.get("errors") and len(result.get("errors", [])) > 0:
                    logger.warning(f"Workflow completed with errors: {result.get('errors')}")
            except Exception as workflow_error:
                logger.error(f"Error in workflow execution: {str(workflow_error)}")
                logger.exception("Workflow exception details:")
                # Create a minimal result with the error information
                result = {
                    "company_url": company_url,
                    "errors": [f"Workflow error: {str(workflow_error)}"],
                    "company_context": {"error": str(workflow_error)},
                    "complete": False
                }
            
            # Extract company context from result
            company_context = result.get("company_context")
            
            # If no context was generated but we have target geography, create a minimal context
            if not company_context and target_geography:
                logger.warning(f"No company context generated, creating minimal context with target geography: {target_geography}")
                company_context = {
                    "name": self._extract_domain(company_url),
                    "description": f"Unable to analyze company at {company_url}.",
                    "url": company_url,
                    "target_geography": target_geography,
                    "errors": result.get("errors", [])
                }
            elif not company_context:
                logger.warning(f"No company context generated, creating minimal context")
                company_context = {
                    "name": self._extract_domain(company_url),
                    "description": f"Unable to analyze company at {company_url}.",
                    "url": company_url,
                    "target_geography": "Unknown",
                    "errors": result.get("errors", [])
                }
            
            # Ensure target geography is preserved
            if target_geography and "target_geography" not in company_context:
                logger.info(f"Adding target geography to result: '{target_geography}'")
                company_context["target_geography"] = target_geography
                
            # Log any errors that were encountered
            errors = result.get("errors", [])
            if errors:
                logger.warning(f"Company analysis completed with {len(errors)} errors: {errors}")
                company_context["errors"] = errors
                
            return company_context
            
        except Exception as e:
            logger.error(f"Error analyzing company {company_url}: {str(e)}")
            logger.exception("Detailed exception:")
            # Return a minimal context if analysis fails
            error_context = {
                "name": self._extract_domain(company_url),
                "description": f"Error analyzing company: {str(e)}",
                "url": company_url,
                "target_geography": target_geography if target_geography else "Unknown",
                "error": str(e)  # Add explicit error field
            }
            logger.info(f"Created fallback context due to error: {error_context}")
            return error_context

# Function for integration with the main application
async def analyze_company_context(company_url: str, model_name: str = "anthropic/claude-3.5-haiku-20241022:beta", target_geography: str = None) -> Dict[str, Any]:
    """
    Analyze a company based on its website URL.
    
    Args:
        company_url: URL of the company website
        model_name: Name of the model to use for analysis
        target_geography: User-specified target geography or market
        
    Returns:
        Dictionary containing company context information
    """
    # Validate the API keys
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    
    if not openrouter_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        raise ValueError("OpenRouter API key is required for company analysis")
    
    if not tavily_key:
        logger.error("TAVILY_API_KEY not found in environment")
        raise ValueError("Tavily API key is required for company analysis")
    
    # Create analyzer with the API keys and specified model
    analyzer = CompanyContextAnalyzer(
        openrouter_api_key=openrouter_key,
        tavily_api_key=tavily_key,
        model_name=model_name
    )
    
    # Run analysis
    try:
        logger.info(f"Starting company analysis for {company_url}")
        if target_geography:
            logger.info(f"Analysis will include user-specified target geography: '{target_geography}'")
        
        # Analyze the company
        company_context = await analyzer.analyze_company(company_url, target_geography)
        
        # If user provided a target geography, override the detected one
        if target_geography:
            logger.info(f"Overriding detected geography with user-specified: '{target_geography}'")
            company_context["target_geography"] = target_geography
            
        logger.info(f"Completed company analysis for {company_url} with target geography: '{company_context.get('target_geography', 'Unknown')}'")
        return company_context
        
    except Exception as e:
        logger.error(f"Error in company analysis: {str(e)}")
        logger.exception("Detailed exception:")
        # Return a minimal context if analysis fails
        error_context = {
            "name": analyzer._extract_domain(company_url),
            "description": f"Error analyzing company: {str(e)}",
            "url": company_url,
            "target_geography": target_geography if target_geography else "Unknown",
            "error": str(e)  # Add explicit error field
        }
        logger.info(f"Created fallback context due to error: {error_context}")
        return error_context 
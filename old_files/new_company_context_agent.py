"""
Agentic Company Analysis Workflow

This module implements an improved two-agent workflow for company analysis:
1. SearchAgent: Generates, executes, and refines search queries for company information
2. AnalysisAgent: Analyzes validated search results to produce quality company insights

The workflow is orchestrated using LangGraph for proper state management and agent coordination.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence, Union
from pydantic import Field, SecretStr
import urllib.parse

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# Import dotenv for environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Custom OpenRouter integration with LangChain
class ChatOpenRouter(ChatOpenAI):
    """Custom implementation of ChatOpenAI that uses OpenRouter."""
    
    def __init__(
        self,
        api_key: Optional[Union[str, SecretStr]] = None,
        model: str = "anthropic/claude-3.5-sonnet",
        **kwargs
    ):
        """
        Initialize ChatOpenRouter with OpenRouter API key.
        
        Args:
            api_key: OpenRouter API key
            model: Model name (default: "anthropic/claude-3.5-sonnet")
            **kwargs: Additional arguments passed to ChatOpenAI
        """
        # Ensure we have an API key
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key is required")
        
        # Process the API key to handle string or SecretStr
        try:
            if isinstance(api_key, str):
                processed_api_key = SecretStr(api_key)
            else:
                processed_api_key = api_key
                
            # Initialize with OpenRouter base URL
            super().__init__(
                model=model,
                base_url="https://openrouter.ai/api/v1",
                api_key=processed_api_key,
                max_retries=2,
                **kwargs
            )
            
        except Exception as e:
            error_msg = f"Error initializing ChatOpenRouter: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

class CompanyAnalysisState(TypedDict, total=False):
    """State maintained throughout the company analysis process."""
    url: str  # Company URL
    company_name: str  # Extracted company name
    search_results: List[Dict]  # Search results
    search_quality: float  # Simple quality score
    target_geography: Optional[str]  # Optional geographic focus
    name: str  # Final company name
    description: str  # Company description/analysis
    analysis: str  # Full analysis
    confidence: str  # Confidence level
    errors: List[str]  # Error messages

class CompanyContextWorkflow:
    """Workflow for analyzing company context using an agent-based approach."""
    
    def __init__(self, openrouter_api_key: Optional[Union[str, SecretStr]] = None):
        """
        Initialize the company context workflow.
        
        Args:
            openrouter_api_key: Optional API key for OpenRouter, if not provided will try to get from env
        """
        # Validate or get API keys
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY")
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required")
            
        # Initialize search tool
        self.search_tool = TavilySearchResults(api_key=self.tavily_api_key)
        
        # Initialize LLM
        self.llm = ChatOpenRouter(
            api_key=self.openrouter_api_key,
            model="anthropic/claude-3.5-sonnet",
        )
        
        # Initialize system prompts
        self.search_system_prompt = """You are an expert research agent tasked with finding comprehensive information about companies.
            
            Your main responsibilities:
            1. EVIDENCE GATHERING: Find specific, factual information from credible sources about the company
            2. CRITICAL ASSESSMENT: Evaluate the quality, recency, and reliability of each piece of information
            3. SOURCE TRACKING: Document where each key fact comes from to establish credibility
            4. VERIFICATION: Cross-check claims across multiple sources when possible
            5. GAP IDENTIFICATION: Explicitly note when important information can't be found
            
            ESSENTIAL GUIDELINES:
            - Be extremely specific in your searches - prefer "CompanyX specific sales process SaaS 2023" over generic "CompanyX sales"
            - NEVER make up or infer information not supported by evidence
            - Prioritize facts over marketing claims
            - Explicitly label the confidence level of each finding (High/Medium/Low)
            - When information conflicts, note the discrepancy and explain which source seems more reliable
            
            SEARCH STRATEGY:
            1. Start with company name, domain and basic details
            2. Investigate specific products, services and pricing models
            3. Research leadership team, founding story, and company history
            4. Examine customer testimonials, case studies, and reviews
            5. Look for competitive comparisons, industry analyses, and market positioning
            6. Identify specific technologies, methodologies, and unique approaches
            
            OUTPUT FORMAT:
            - Clearly separate FACTS (verified information) from POSSIBILITIES (reasonable inferences)
            - For each key finding, note source and recency
            - Explicitly state confidence levels for each section
            - Document search queries attempted and their effectiveness
            """
        
        self.analysis_system_prompt = """You are an elite business analyst creating precise, actionable company profiles for B2B sales teams.
            
            Your analysis MUST be:
            1. SPECIFIC: Avoid generic statements ("AI-powered solutions", "customer-centric approach") unless you can detail HOW they implement these concepts
            2. EVIDENCE-BASED: Only include claims you can support with specific sources
            3. BALANCED: Present both strengths and limitations of the company
            4. COMPARATIVE: Position them against specific named competitors
            5. ACTIONABLE: Focus on intelligence that would help in sales conversations
            
            REQUIRED SECTIONS:
            
            1. COMPANY BASICS
            - Full legal name, founding date, HQ location, employee count
            - Ownership structure (public/private, parent company)
            - Recent funding or significant financial events
            - Leadership team with backgrounds
            
            2. OFFERINGS (Be specific with names, features, pricing when available)
            - Main product/service lines with SPECIFIC capabilities
            - Pricing models and contract structures
            - Implementation process and timeline
            - Technical requirements and integrations
            
            3. DIFFERENTIATION (Must be specific and verifiable)
            - Proprietary technology or methodologies (name them specifically)
            - Unique capabilities competitors lack (with evidence)
            - Published success metrics or performance benchmarks
            - Patents, certifications, or unique partnerships
            
            4. TARGET MARKET
            - Specific industry verticals served (name the top 3-5)
            - Company size sweet spot (employee count/revenue range)
            - Buyer personas with job titles and responsibilities
            - Geographic focus with any regional specialization
            
            5. SALES INTELLIGENCE
            - Sales process and typical sales cycle length
            - Decision-makers and influencers in target accounts
            - Common objections and how to counter them
            - Competitive displacement strategies
            - Trigger events that create sales opportunities
            
            CRITICAL GUIDELINES:
            - Indicate confidence level (High/Medium/Low) for each section
            - When information is limited, explicitly acknowledge gaps rather than filling with generalities
            - Include direct quotes from customers or executives when available
            - For topics with limited evidence, clearly label as "Limited information available" rather than speculating
            - Focus on details that would actually influence a buying decision
            """
        
        # Setup state graph
        graph = StateGraph(CompanyAnalysisState)
        
        # Add nodes for search and analysis
        graph.add_node("search", self._search_agent_node)
        graph.add_node("analysis", self._analysis_agent_node)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "search",
            self._should_proceed_to_analysis,
            {
                "proceed": "analysis",
                "insufficient_data": END
            }
        )
        
        # Set analysis as the end node
        graph.add_edge("analysis", END)
        
        # Compile the graph
        self.workflow = graph.compile()
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the company context workflow.
        
        Args:
            state: Initial state dictionary with 'url' and optional 'target_geography'
            
        Returns:
            Dict with analysis results
        """
        try:
            result = self.workflow.invoke(state)
            return result
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            # Return state with error
            if isinstance(state, dict):
                state["errors"] = state.get("errors", []) + [f"Workflow error: {str(e)}"]
                return state
            return {"errors": [f"Workflow error: {str(e)}"]}
    
    def _should_proceed_to_analysis(self, state: CompanyAnalysisState) -> str:
        """Determine if we should proceed to analysis based on search results."""
        # Simplified logic - just check if we have any results
        if len(state.get("search_results", [])) > 0:
            return "proceed"
        else:
            logger.warning("No search results found")
            return "insufficient_data"
            
    def _search_agent_node(self, state: CompanyAnalysisState) -> CompanyAnalysisState:
        """Generate search queries and execute searches to gather information about a company."""
        if "url" not in state:
            state["errors"] = ["No company URL provided"]
            return state
            
        company_url = state["url"]
        logger.info(f"Executing search agent for URL: {company_url}")
        
        # Initialize errors list if it doesn't exist
        if "errors" not in state:
            state["errors"] = []
        
        try:
            # Extract domain name for basic company name
            domain = self._extract_domain(company_url)
            company_name = domain.replace('-', ' ').replace('.', ' ')
            state["company_name"] = company_name
            
            # Simple search queries that cover important business aspects
            search_queries = [
                f"{company_name} company overview what they do",
                f"{company_name} products services offerings",
                f"{company_name} target market customers industry",
                f"{company_name} competitive advantage unique selling proposition",
                f"{company_name} leadership team founders history"
            ]
            
            # Add geography-targeted search if specified
            target_geography = state.get("target_geography")
            if target_geography:
                search_queries.append(f"{company_name} business {target_geography} market presence")
            
            # Execute searches and collect results
            all_search_results = []
            for query in search_queries:
                try:
                    logger.info(f"Searching for: {query}")
                    results = self.search_tool.invoke({"query": query})
                    
                    # Add metadata
                    for result in results:
                        result["query"] = query
                        
                    all_search_results.extend(results)
                    
                except Exception as e:
                    error_msg = f"Error executing search query '{query}': {str(e)}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)
            
            # Update state with search results
            state["search_results"] = all_search_results
            
            # Set a default search quality based on number of results
            if len(all_search_results) > 10:
                state["search_quality"] = 0.8
            elif len(all_search_results) > 5:
                state["search_quality"] = 0.6
            elif len(all_search_results) > 0:
                state["search_quality"] = 0.4
            else:
                state["search_quality"] = 0.0
                state["errors"].append("No search results found")
            
            return state
            
        except Exception as e:
            error_msg = f"Error in search agent: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["search_quality"] = 0.0
            return state
            
    def _analysis_agent_node(self, state: CompanyAnalysisState):
        """Analyze search results to create a company profile."""
        # Initialize errors list if not present
        if "errors" not in state:
            state["errors"] = []
        
        try:
            # Format search results for the prompt
            formatted_results = []
            for i, result in enumerate(state.get("search_results", [])[:15]):  # Limit to top 15 results
                query = result.get("query", "Unknown query")
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                source = result.get("source", "Unknown source")
                
                # Truncate long content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                    
                formatted_results.append(f"RESULT {i+1}\nQuery: {query}\nSource: {source}\nTitle: {title}\nContent: {content}\n")
            
            results_text = "\n\n".join(formatted_results)
            
            # Include geography instruction if specified
            target_geography = state.get("target_geography", "")
            geography_instruction = f"Pay special attention to information relevant to {target_geography}." if target_geography else ""
            
            # Construct the analysis prompt
            human_prompt = f"""Analyze the following search results about the company at URL: {state.get('url', 'No URL provided')}
            
            {geography_instruction}
            
            SEARCH RESULTS:
            {results_text}
            
            Create a comprehensive company profile that would be useful for B2B sales professionals.
            Be specific, factual, and focus on concrete details about the company from the search results.
            If information is missing on certain topics, clearly state what isn't known rather than making assumptions.
            """
            
            # Use messaging format for analysis
            messages = [
                SystemMessage(content=self.analysis_system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Send to LLM
            logger.info("Executing analysis with search results")
            result = self.llm.invoke(messages)
            analysis_output = result.content
            logger.info(f"Analysis completed, output length: {len(analysis_output)}")
            
            # Extract company name
            company_name = self._extract_company_name(analysis_output) or state.get("company_name", "Unknown")
            
            # Update state with analysis results
            state["analysis"] = analysis_output
            state["name"] = company_name
            state["description"] = self._extract_company_description(analysis_output)
            state["target_geography"] = target_geography or "Global"
            
            # Set confidence based on number of search results
            result_count = len(state.get("search_results", []))
            if result_count > 10:
                state["confidence"] = "High"
            elif result_count > 5:
                state["confidence"] = "Medium"
            else:
                state["confidence"] = "Low"
                
            return state
        
        except Exception as e:
            logger.error(f"Error in analysis agent: {str(e)}")
            state["errors"].append(f"Analysis error: {str(e)}")
            state["confidence"] = "Low"
            return state
            
    def _extract_company_description(self, text):
        """Extract a concise company description from analysis text."""
        # Look for company description in common formats
        description_patterns = [
            r"(?:COMPANY BASICS|OVERVIEW|SUMMARY)[\s\S]*?([\w\s.,;:()\-\"\'&]+?)(?:\n\n|\n#|\Z)",
            r"(?:is|are) ((?:a|an) [\w\s.,;:()\-\"\'&]+?company[\w\s.,;:()\-\"\'&]+?)(?:\.|$|\n)",
            r"(?:DESCRIPTION|ABOUT)[\s\S]*?([\w\s.,;:()\-\"\'&]+?)(?:\n\n|\n#|\Z)"
        ]
        
        for pattern in description_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                description = matches.group(1).strip()
                # Limit to ~100 words for conciseness
                words = description.split()
                if len(words) > 100:
                    description = " ".join(words[:100]) + "..."
                return description
        
        # Fallback: Use first paragraph if no clear description found
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            words = paragraphs[0].split()
            if len(words) > 100:
                return " ".join(words[:100]) + "..."
            return paragraphs[0].strip()
        
        return "No description available."
        
    def _extract_company_name(self, analysis_text: str) -> str:
        """Extract company name from analysis text using simple patterns."""
        # Look for common company name patterns
        simple_patterns = [
            r"Company Name:[\s]*([^.\n]+)",
            r"Name:[\s]*([^.\n]+)",
            r"^#[\s]*([^.\n]+)",
            r"([A-Za-z0-9][A-Za-z0-9\s.,&-]{2,25}) is a "
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to domain name
        return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            # Parse the URL and extract domain
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Remove TLD for a cleaner name
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                return domain_parts[0].capitalize()
                
            return domain.capitalize()
        except:
            # If URL parsing fails, return the URL as is
            return url

class AgenticCompanyAnalyzer:
    """
    Implements an agentic approach to company analysis using a two-agent workflow.
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 tavily_api_key: Optional[str] = None,
                 model_name: str = "anthropic/claude-3.5-sonnet-20241022:beta"):
        """
        Initialize the agentic company analyzer.
        
        Args:
            openrouter_api_key: API key for OpenRouter
            tavily_api_key: API key for Tavily search
            model_name: Name of the model to use
        """
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self.model_name = model_name
        
        # Validate keys
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required")
        
        # Set up the model
        self.llm = ChatOpenRouter(
            openai_api_key=self.openrouter_api_key,
            model=model_name,
            temperature=0.2
        )
        
        # Set up search tool
        self.search_tool = TavilySearchResults(api_key=self.tavily_api_key)
        
        # Initialize agents and workflow
        self._initialize_agents()
        self.workflow = self._build_workflow()
    
    def _initialize_agents(self):
        """Initialize the search and analysis agents."""
        # Define search tools
        @tool
        def search(query: str) -> str:
            """
            Execute a search query to find information about a company.
            
            Args:
                query: The search query to execute
                
            Returns:
                Search results as formatted text
            """
            try:
                logger.debug(f"Executing search query: {query}")
                results = self.search_tool.invoke(query)
                
                # Format results for readability
                if not results:
                    return "No search results found for this query."
                
                formatted_results = ""
                for i, result in enumerate(results[:5]):  # Limit to top 5 results
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    formatted_results += f"Result {i+1}: {title}\n"
                    formatted_results += f"Content: {content[:500]}...\n\n"
                
                return formatted_results
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return f"Error executing search: {str(e)}"
        
        @tool
        def extract_domain(url: str) -> str:
            """
            Extract the domain name from a URL.
            
            Args:
                url: The URL to extract the domain from
                
            Returns:
                The extracted domain name
            """
            try:
                # Parse the URL and extract domain
                parsed = urllib.parse.urlparse(url)
                domain = parsed.netloc
                
                # Remove www. prefix if present
                if domain.startswith('www.'):
                    domain = domain[4:]
                    
                # Remove TLD for a cleaner name
                domain_parts = domain.split('.')
                if len(domain_parts) > 1:
                    return domain_parts[0].capitalize()
                    
                return domain.capitalize()
            except:
                # If URL parsing fails, return the URL as is
                return url
        
        # Store the search tools for use in execution
        self.search_tools = [search, extract_domain]
        
        # We don't need to create React agents anymore, we'll use direct LLM calls
        self.search_system_prompt = """You are an expert research agent tasked with finding comprehensive information about companies.
            
            Your main responsibilities:
            1. EVIDENCE GATHERING: Find specific, factual information from credible sources about the company
            2. CRITICAL ASSESSMENT: Evaluate the quality, recency, and reliability of each piece of information
            3. SOURCE TRACKING: Document where each key fact comes from to establish credibility
            4. VERIFICATION: Cross-check claims across multiple sources when possible
            5. GAP IDENTIFICATION: Explicitly note when important information can't be found
            
            ESSENTIAL GUIDELINES:
            - Be extremely specific in your searches - prefer "CompanyX specific sales process SaaS 2023" over generic "CompanyX sales"
            - NEVER make up or infer information not supported by evidence
            - Prioritize facts over marketing claims
            - Explicitly label the confidence level of each finding (High/Medium/Low)
            - When information conflicts, note the discrepancy and explain which source seems more reliable
            
            SEARCH STRATEGY:
            1. Start with company name, domain and basic details
            2. Investigate specific products, services and pricing models
            3. Research leadership team, founding story, and company history
            4. Examine customer testimonials, case studies, and reviews
            5. Look for competitive comparisons, industry analyses, and market positioning
            6. Identify specific technologies, methodologies, and unique approaches
            
            OUTPUT FORMAT:
            - Clearly separate FACTS (verified information) from POSSIBILITIES (reasonable inferences)
            - For each key finding, note source and recency
            - Explicitly state confidence levels for each section
            - Document search queries attempted and their effectiveness
            """
        
        self.analysis_system_prompt = """You are an elite business analyst creating precise, actionable company profiles for B2B sales teams.
            
            Your analysis MUST be:
            1. SPECIFIC: Avoid generic statements ("AI-powered solutions", "customer-centric approach") unless you can detail HOW they implement these concepts
            2. EVIDENCE-BASED: Only include claims you can support with specific sources
            3. BALANCED: Present both strengths and limitations of the company
            4. COMPARATIVE: Position them against specific named competitors
            5. ACTIONABLE: Focus on intelligence that would help in sales conversations
            
            REQUIRED SECTIONS:
            
            1. COMPANY BASICS
            - Full legal name, founding date, HQ location, employee count
            - Ownership structure (public/private, parent company)
            - Recent funding or significant financial events
            - Leadership team with backgrounds
            
            2. OFFERINGS (Be specific with names, features, pricing when available)
            - Main product/service lines with SPECIFIC capabilities
            - Pricing models and contract structures
            - Implementation process and timeline
            - Technical requirements and integrations
            
            3. DIFFERENTIATION (Must be specific and verifiable)
            - Proprietary technology or methodologies (name them specifically)
            - Unique capabilities competitors lack (with evidence)
            - Published success metrics or performance benchmarks
            - Patents, certifications, or unique partnerships
            
            4. TARGET MARKET
            - Specific industry verticals served (name the top 3-5)
            - Company size sweet spot (employee count/revenue range)
            - Buyer personas with job titles and responsibilities
            - Geographic focus with any regional specialization
            
            5. SALES INTELLIGENCE
            - Sales process and typical sales cycle length
            - Decision-makers and influencers in target accounts
            - Common objections and how to counter them
            - Competitive displacement strategies
            - Trigger events that create sales opportunities
            
            CRITICAL GUIDELINES:
            - Indicate confidence level (High/Medium/Low) for each section
            - When information is limited, explicitly acknowledge gaps rather than filling with generalities
            - Include direct quotes from customers or executives when available
            - For topics with limited evidence, clearly label as "Limited information available" rather than speculating
            - Focus on details that would actually influence a buying decision
            """
    
    def _search_agent_node(self, state: CompanyAnalysisState) -> CompanyAnalysisState:
        """Generate search queries and execute searches to gather information about a company."""
        if "url" not in state:
            state["errors"] = ["No company URL provided"]
            return state
            
        company_url = state["url"]
        logger.info(f"Executing search agent for URL: {company_url}")
        
        # Initialize errors list if it doesn't exist
        if "errors" not in state:
            state["errors"] = []
        
        try:
            # Extract domain name for basic company name
            domain = self._extract_domain(company_url)
            company_name = domain.replace('-', ' ').replace('.', ' ')
            state["company_name"] = company_name
            
            # Simple search queries that cover important business aspects
            search_queries = [
                f"{company_name} company overview what they do",
                f"{company_name} products services offerings",
                f"{company_name} target market customers industry",
                f"{company_name} competitive advantage unique selling proposition",
                f"{company_name} leadership team founders history"
            ]
            
            # Add geography-targeted search if specified
            target_geography = state.get("target_geography")
            if target_geography:
                search_queries.append(f"{company_name} business {target_geography} market presence")
            
            # Execute searches and collect results
            all_search_results = []
            for query in search_queries:
                try:
                    logger.info(f"Searching for: {query}")
                    results = self.search_tool.invoke({"query": query})
                    
                    # Add metadata
                    for result in results:
                        result["query"] = query
                        
                    all_search_results.extend(results)
                    
                except Exception as e:
                    error_msg = f"Error executing search query '{query}': {str(e)}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)
            
            # Update state with search results
            state["search_results"] = all_search_results
            
            # Set a default search quality based on number of results
            if len(all_search_results) > 10:
                state["search_quality"] = 0.8
            elif len(all_search_results) > 5:
                state["search_quality"] = 0.6
            elif len(all_search_results) > 0:
                state["search_quality"] = 0.4
            else:
                state["search_quality"] = 0.0
                state["errors"].append("No search results found")
            
            return state
            
        except Exception as e:
            error_msg = f"Error in search agent: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["search_quality"] = 0.0
            return state
            
    def _should_proceed_to_analysis(self, state: CompanyAnalysisState) -> str:
        """Determine if we should proceed to analysis based on search results."""
        # Simplified logic - just check if we have any results
        if len(state.get("search_results", [])) > 0:
            return "proceed"
        else:
            logger.warning("No search results found")
            return "insufficient_data"
    
    def _analysis_agent_node(self, state: CompanyAnalysisState):
        """Analyze search results to create a company profile."""
        # Initialize errors list if not present
        if "errors" not in state:
            state["errors"] = []
        
        try:
            # Format search results for the prompt
            formatted_results = []
            for i, result in enumerate(state.get("search_results", [])[:15]):  # Limit to top 15 results
                query = result.get("query", "Unknown query")
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                source = result.get("source", "Unknown source")
                
                # Truncate long content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                    
                formatted_results.append(f"RESULT {i+1}\nQuery: {query}\nSource: {source}\nTitle: {title}\nContent: {content}\n")
            
            results_text = "\n\n".join(formatted_results)
            
            # Include geography instruction if specified
            target_geography = state.get("target_geography", "")
            geography_instruction = f"Pay special attention to information relevant to {target_geography}." if target_geography else ""
            
            # Construct the analysis prompt
            human_prompt = f"""Analyze the following search results about the company at URL: {state.get('url', 'No URL provided')}
            
            {geography_instruction}
            
            SEARCH RESULTS:
            {results_text}
            
            Create a comprehensive company profile that would be useful for B2B sales professionals.
            Be specific, factual, and focus on concrete details about the company from the search results.
            If information is missing on certain topics, clearly state what isn't known rather than making assumptions.
            """
            
            # Use messaging format for analysis
            messages = [
                SystemMessage(content=self.analysis_system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Send to LLM
            logger.info("Executing analysis with search results")
            result = self.llm.invoke(messages)
            analysis_output = result.content
            logger.info(f"Analysis completed, output length: {len(analysis_output)}")
            
            # Extract company name
            company_name = self._extract_company_name(analysis_output) or state.get("company_name", "Unknown")
            
            # Update state with analysis results
            state["analysis"] = analysis_output
            state["name"] = company_name
            state["description"] = self._extract_company_description(analysis_output)
            state["target_geography"] = target_geography or "Global"
            
            # Set confidence based on number of search results
            result_count = len(state.get("search_results", []))
            if result_count > 10:
                state["confidence"] = "High"
            elif result_count > 5:
                state["confidence"] = "Medium"
            else:
                state["confidence"] = "Low"
                
            return state
        
        except Exception as e:
            logger.error(f"Error in analysis agent: {str(e)}")
            state["errors"].append(f"Analysis error: {str(e)}")
            state["confidence"] = "Low"
            return state
    
    def _extract_company_description(self, text):
        """Extract a concise company description from analysis text."""
        # Look for company description in common formats
        description_patterns = [
            r"(?:COMPANY BASICS|OVERVIEW|SUMMARY)[\s\S]*?([\w\s.,;:()\-\"\'&]+?)(?:\n\n|\n#|\Z)",
            r"(?:is|are) ((?:a|an) [\w\s.,;:()\-\"\'&]+?company[\w\s.,;:()\-\"\'&]+?)(?:\.|$|\n)",
            r"(?:DESCRIPTION|ABOUT)[\s\S]*?([\w\s.,;:()\-\"\'&]+?)(?:\n\n|\n#|\Z)"
        ]
        
        for pattern in description_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                description = matches.group(1).strip()
                # Limit to ~100 words for conciseness
                words = description.split()
                if len(words) > 100:
                    description = " ".join(words[:100]) + "..."
                return description
        
        # Fallback: Use first paragraph if no clear description found
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            words = paragraphs[0].split()
            if len(words) > 100:
                return " ".join(words[:100]) + "..."
            return paragraphs[0].strip()
        
        return "No description available."
    
    def _extract_company_name(self, analysis_text: str) -> str:
        """Extract company name from analysis text using simple patterns."""
        # Look for common company name patterns
        simple_patterns = [
            r"Company Name:[\s]*([^.\n]+)",
            r"Name:[\s]*([^.\n]+)",
            r"^#[\s]*([^.\n]+)",
            r"([A-Za-z0-9][A-Za-z0-9\s.,&-]{2,25}) is a "
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to domain name
        return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            # Parse the URL and extract domain
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Remove TLD for a cleaner name
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                return domain_parts[0].capitalize()
                
            return domain.capitalize()
        except:
            # If URL parsing fails, return the URL as is
            return url
    
    def _build_workflow(self):
        """Build the LangGraph workflow for company analysis."""
        # Create a state graph with the CompanyAnalysisState type
        workflow = StateGraph(CompanyAnalysisState)
        
        # Add nodes for each task
        workflow.add_node("search", self._search_agent_node)
        workflow.add_node("analysis", self._analysis_agent_node)
        
        # Define the edges
        workflow.add_edge("search", "analysis")
        workflow.add_conditional_edges(
            "search",
            self._should_proceed_to_analysis,
            {
                "proceed": "analysis",
                "insufficient_data": END
            }
        )
        workflow.add_edge("analysis", END)
        
        # Set the entry point
        workflow.set_entry_point("search")
        
        # Enable tracing for the workflow if LangSmith is configured
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
            logger.info("LangSmith tracing enabled for company context workflow")
        
        # Compile the graph
        return workflow.compile()
    
    async def analyze_company(self, company_url: str, target_geography: str = None) -> Dict[str, Any]:
        """
        Analyze a company based on its website URL using the agentic workflow.
        
        Args:
            company_url: URL of the company website
            target_geography: User-specified target geography
            
        Returns:
            Dictionary containing company context information
        """
        try:
            # Initialize state for the workflow
            state = {
                "url": company_url,
                "search_queries": [],
                "search_results": [],
                "search_quality": None,
                "target_geography": target_geography,
                "name": "",
                "description": "",
                "analysis": "",
                "confidence": "",
                "errors": []
            }
            
            # Execute the workflow
            logger.info(f"Starting agentic company workflow for {company_url}")
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
                    "url": company_url,
                    "errors": [f"Workflow error: {str(workflow_error)}"],
                    "company_context": {"error": str(workflow_error)},
                    "name": "",
                    "description": "",
                    "target_geography": target_geography if target_geography else "Unknown",
                    "search_quality": None,
                    "confidence": "Low"
                }
            
            # Extract company context from result
            company_context = result.get("company_context")
            
            # If no context was generated, create a minimal context
            if not company_context:
                logger.warning(f"No company context generated, creating minimal context")
                company_context = {
                    "name": self._extract_domain(company_url),
                    "description": f"Unable to analyze company at {company_url}.",
                    "url": company_url,
                    "target_geography": target_geography if target_geography else "Unknown",
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

def analyze_company_context_agentic(
    company_url: str, 
    target_geography: Optional[str] = None,
    openrouter_api_key: Optional[Union[str, SecretStr]] = None
) -> Dict[str, Any]:
    """
    Analyze a company using search tools to create a detailed context profile.
    
    Args:
        company_url: URL of the company website to analyze
        target_geography: Optional target geography to focus analysis on
        openrouter_api_key: Optional OpenRouter API key for LLM usage
        
    Returns:
        Dict containing the analysis results with key information
    """
    try:
        logger.info(f"Starting company context analysis for URL: {company_url}")
        
        # Initialize the workflow
        workflow = CompanyContextWorkflow(openrouter_api_key=openrouter_api_key)
        
        # Prepare initial state
        initial_state = {
            "url": company_url,
            "target_geography": target_geography,
            "errors": []
        }
        
        # Execute the workflow
        logger.info("Executing company context workflow")
        result = workflow.run(initial_state)
        
        # Clean up and return only relevant data
        return {
            "name": result.get("name", _extract_domain_from_url(company_url)),
            "description": result.get("description", "No description available"),
            "url": company_url,
            "target_geography": result.get("target_geography", "Global"),
            "search_quality": result.get("search_quality", 0),
            "confidence": result.get("confidence", "Low"),
            "errors": result.get("errors", [])
        }
        
    except Exception as e:
        error_msg = f"Error analyzing company context: {str(e)}"
        logger.error(error_msg)
        return {
            "name": _extract_domain_from_url(company_url),
            "description": "Error analyzing company. See errors for details.",
            "url": company_url,
            "target_geography": target_geography if target_geography else "Global",
            "search_quality": 0,
            "confidence": "None",
            "errors": [error_msg]
        }

def _extract_domain_from_url(url: str) -> str:
    """Helper function to extract domain name from URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # Remove TLD for a cleaner name
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            return domain_parts[0].capitalize()
            
        return domain.capitalize()
    except:
        # If URL parsing fails, return the URL as is
        return url

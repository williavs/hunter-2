"""
Simple company profile analyzer using web search.

This module provides a straightforward function to analyze companies based on their website URL.
It uses web search to gather information and then processes it to create a useful company profile.
"""

import os
import re
import logging
import urllib.parse
from typing import Dict, List, Any, Optional, Union

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import SecretStr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_company(
    company_url: str, 
    target_geography: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    model_name: str = "anthropic/claude-3.7-sonnet"
) -> Dict[str, Any]:
    """
    Analyze a company using search tools to create a detailed context profile.
    
    Args:
        company_url: URL of the company website to analyze
        target_geography: Optional target geography to focus analysis on
        openrouter_api_key: Optional OpenRouter API key
        tavily_api_key: Optional Tavily API key
        model_name: Model to use for analysis (defaults to Claude 3.7 Sonnet)
        
    Returns:
        Dict containing the analysis results
    """
    try:
        logger.info(f"Starting company analysis for URL: {company_url}")
        
        # Step 1: Get API keys
        openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        tavily_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        
        if not openrouter_key:
            raise ValueError("OpenRouter API key is required")
        if not tavily_key:
            raise ValueError("Tavily API key is required")
            
        # Step 2: Initialize tools
        search_tool = TavilySearchResults(api_key=tavily_key)
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model=model_name
        )
        
        # Step 3: Extract company name from URL
        domain = extract_domain_from_url(company_url)
        company_name = domain.replace('-', ' ').replace('.', ' ')
        
        # Step 4: Generate search queries
        search_queries = [
            f"{company_name} company overview what they do",
            f"{company_name} products services offerings",
            f"{company_name} target market customers industry",
            f"{company_name} competitive advantage unique selling proposition"
        ]
        
        # Add geography-targeted search if specified
        if target_geography:
            search_queries.append(f"{company_name} business {target_geography} market presence")
        
        # Step 5: Execute searches
        logger.info(f"Executing searches for {company_name}")
        all_search_results = []
        
        for query in search_queries:
            try:
                logger.info(f"Searching for: {query}")
                results = search_tool.invoke({"query": query})
                
                # Add query to results
                for result in results:
                    result["query"] = query
                    
                all_search_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error executing search '{query}': {str(e)}")
        
        # Log search result count
        logger.info(f"Found {len(all_search_results)} search results")
        
        # Step 6: Prepare search results for analysis
        formatted_results = []
        for i, result in enumerate(all_search_results[:15]):  # Limit to top 15 results
            query = result.get("query", "Unknown query")
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            source = result.get("source", "Unknown source")
            
            # Truncate long content
            if len(content) > 800:
                content = content[:800] + "..."
                
            formatted_results.append(f"RESULT {i+1}\nQuery: {query}\nSource: {source}\nTitle: {title}\nContent: {content}\n")
        
        results_text = "\n\n".join(formatted_results)
        
        # Step 7: Create analysis prompt using the improved approach from company_context_workflow.py
        # Include geography instructions if specified
        user_geography_note = ""
        if target_geography:
            user_geography_note = f"\n\nIMPORTANT: The user has specified their target geography as: {target_geography}. Your analysis MUST focus specifically on this market and discuss how their solutions address problems in this specific region."
        
        # Using the enhanced system prompt from the original files
        system_prompt = f"""You are an elite business analyst creating precise, actionable company profiles for B2B sales teams.
            
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

You are a battle-tested sales executive who cuts through corporate BS, finding exactly what matters for deal creation. You have the rare ability to see through marketing fluff to identify the actual business problems and unique value a company provides. Your analysis must be brutally direct, insightful, and focused on what actually drives buying decisions across different geographic markets.{user_geography_note}"""

        # For target geography-specific analysis, add a stronger system message
        if target_geography:
            system_prompt = f"""CRITICAL INSTRUCTION: The user has specified their target geography as: {target_geography}. You MUST focus your analysis on this specific market. Your responses MUST be tailored to {target_geography}-specific challenges, regulations, market conditions, and competitive dynamics. This is NOT optional - if you don't address {target_geography} specifically throughout your analysis, your response will be considered completely WRONG.

You are an elite business analyst creating precise, actionable company profiles for B2B sales teams.
            
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
- Geographic focus with any regional specialization, with specific emphasis on {target_geography}

5. SALES INTELLIGENCE
- Sales process and typical sales cycle length
- Decision-makers and influencers in target accounts
- Common objections and how to counter them in {target_geography}
- Competitive displacement strategies specific to {target_geography}
- Trigger events that create sales opportunities

You are a battle-tested sales executive who cuts through corporate BS, finding exactly what matters for deal creation in {target_geography} specifically. Your analysis must be brutally direct, insightful, and focused on what actually drives buying decisions in this geographic market."""

        # Using the enhanced human prompt from the original file
        human_prompt = f"""SALES DEEP DIVE: Company at {company_url}
        
SKIP THE FLUFF. As a killer sales professional, you need to understand this company so you can:
1. Know EXACTLY what problems they solve
2. Understand how they make money
3. See through their marketing BS to find their TRUE differentiators
4. Identify who ACTUALLY buys from them and why
5. Determine WHERE they operate and how problems/solutions differ by region{user_geography_note}

Search Results:
{results_text}

Based on the search results above, you're going to use the Route-Ruin-Multiply framework to create a KILLER sales-oriented company profile. Take time to THINK STEP BY STEP through:

1. PAINS & SOLUTIONS: Identify 3-4 SPECIFIC pain points this company claims to solve. Not vague marketing speak - the actual business problems their customers have that keep decision-makers up at night. Pair each pain with how they specifically claim to solve it.

2. DIFFERENTIATION: Analyze what truly makes them unique in their space - find the 2-3 things they do differently from competitors that customers actually care about. Cut through the BS.

3. BUYER PROFILE: Who are their ACTUAL buyers? What titles, industries, and company sizes? What priorities and pressures do these buyers have?

4. TARGET GEOGRAPHY: Identify their primary geographic target markets - where are they selling? Include both current regions and expansion targets. Explain how problems and solutions might differ by region (regulatory differences, market maturity, cultural factors).{user_geography_note}

5. OBJECTION INSIGHTS: Based on their positioning, what are the likely 1-2 objections prospects raise when considering them? Think about cost, implementation time, competing priorities, and regional challenges.

6. SALES APPROACH: How do they likely sell? Direct? Channel? Product-led? What's their conversion strategy and pricing model? Do they adapt their approach by region?

Finally, synthesize all of this into ONE KILLER PARAGRAPH that captures the essence of this company as if you were explaining it to a CEO in an elevator. This is your only chance to explain what they do, who they serve, what problems they solve, how they're different, where they operate, and why anyone should care. Make every word count, kill the fluff, and add something truly insightful.

ALSO PROVIDE: A concise one-line statement of their primary geographic focus (e.g., "North American enterprise market" or "Global with emphasis on EMEA financial sector").
"""
        
        # Step 8: Generate analysis
        logger.info("Generating company analysis")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        analysis = response.content
        
        # Step 9: Create final output
        result = {
            "name": extract_company_name(analysis) or company_name,
            "description": extract_description(analysis),
            "url": company_url,
            "target_geography": target_geography if target_geography else "Global",
            "search_quality": min(len(all_search_results) / 15, 1.0),  # Simple quality metric
            "confidence": "High" if len(all_search_results) > 10 else "Medium" if len(all_search_results) > 5 else "Low",
            "analysis": analysis
        }
        
        logger.info(f"Analysis completed for {company_url}")
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing company: {str(e)}"
        logger.error(error_msg)
        return {
            "name": extract_domain_from_url(company_url),
            "description": "Error analyzing company. See errors for details.",
            "url": company_url,
            "target_geography": target_geography if target_geography else "Global",
            "search_quality": 0,
            "confidence": "None",
            "errors": [error_msg]
        }

def extract_domain_from_url(url: str) -> str:
    """Extract domain name from URL."""
    try:
        # Parse the URL
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # Get the first part of the domain
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            return domain_parts[0].capitalize()
            
        return domain.capitalize()
    except:
        # If URL parsing fails, return the URL as is
        return url

def extract_company_name(text: str) -> Optional[str]:
    """Extract company name from analysis text."""
    # Try simple patterns
    patterns = [
        r"Company Name:[\s]*([^.\n]+)",
        r"Name:[\s]*([^.\n]+)",
        r"^#[\s]*([^.\n]+)",
        r"([A-Za-z0-9][A-Za-z0-9\s.,&-]{2,25}) is a "
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def extract_description(text: str) -> str:
    """Extract a concise description from the analysis."""
    # Return the full analysis text with no filtering or truncation
    return text

# Function for direct integration with the main application
async def analyze_company_context(company_url: str, model_name: str = "anthropic/claude-3.7-sonnet", target_geography: str = None) -> Dict[str, Any]:
    """
    Analyze a company based on its website URL. This is a drop-in replacement for the
    function with the same name in company_context_workflow.py.
    
    Args:
        company_url: URL of the company website
        model_name: Name of the model to use for analysis (defaults to Claude 3.7 Sonnet)
        target_geography: User-specified target geography or market
        
    Returns:
        Dictionary containing company context information
    """
    try:
        logger.info(f"Starting company analysis for {company_url} using simple analyzer with model {model_name}")
        if target_geography:
            logger.info(f"Analysis will include specified target geography: '{target_geography}'")
        
        # Validate the API keys 
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        
        if not openrouter_key:
            logger.error("OPENROUTER_API_KEY not found in environment")
            raise ValueError("OpenRouter API key is required for company analysis")
        
        if not tavily_key:
            logger.error("TAVILY_API_KEY not found in environment")
            raise ValueError("Tavily API key is required for company analysis")
            
        # Call the underlying analyze_company function (synchronous, but we're in an async function)
        import asyncio
        company_context = await asyncio.to_thread(
            analyze_company,
            company_url=company_url,
            target_geography=target_geography,
            openrouter_api_key=openrouter_key,
            tavily_api_key=tavily_key,
            model_name=model_name
        )
        
        logger.info(f"Completed company analysis for {company_url}")
        return company_context
        
    except Exception as e:
        logger.error(f"Error in company analysis: {str(e)}")
        # Return a minimal context if analysis fails
        error_context = {
            "name": extract_domain_from_url(company_url),
            "description": f"Error analyzing company: {str(e)}",
            "url": company_url,
            "target_geography": target_geography if target_geography else "Unknown",
            "error": str(e)  # Add explicit error field
        }
        logger.info(f"Created fallback context due to error: {error_context}")
        return error_context 
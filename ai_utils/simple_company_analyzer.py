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
from pydantic import SecretStr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_company(
    company_url: str, 
    company_name: str,
    target_geography: Optional[str] = None,
    model_name: str = "gpt-4.1",
) -> Dict[str, Any]:
    """
    Analyze a company using OpenAI web search to create a detailed context profile.
    
    Args:
        company_url: URL of the company website to analyze
        company_name: Name of the company (explicitly provided)
        target_geography: Optional target geography to focus analysis on
        model_name: Model to use for analysis (defaults to GPT-4.1)
        
    Returns:
        Dict containing the analysis results
    """
    try:
        logger.info(f"Starting company analysis for {company_name} (URL: {company_url})")
        
        # Step 1: Initialize tools - Always use OpenAI web search
        logger.info("Using OpenAI web search")
        search_llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4.1" # Use a cost-effective model for search
        )
        # Bind the web search tool
        search_tool = search_llm.bind_tools([{"type": "web_search_preview"}])
            
        # Initialize the main LLM for analysis
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=0.1,
            streaming=False,
            timeout=90  # Adding a longer timeout for reliability
        )
        
        # Step 2: Generate search queries using the provided company name
        logger.info(f"Using company name: {company_name}")
        
        # Step 3: Generate search queries
        search_queries = [
            f"{company_name} company overview what they do",
            f"{company_name} products services offerings",
            f"{company_name} target market customers industry",
            f"{company_name} competitive advantage unique selling proposition"
        ]
        
        # Add geography-targeted search if specified
        if target_geography:
            search_queries.append(f"{company_name} business {target_geography} market presence")
        
        # Step 4: Execute searches
        logger.info(f"Executing searches for {company_name}")
        all_search_results = []
        
        for query in search_queries:
            try:
                logger.info(f"Searching for: {query}")
                # Format for OpenAI web search
                search_query = f"Search for information about: {query}"
                logger.info(f"Search query sent to OpenAI: {search_query}")
                response = search_tool.invoke(search_query)
                
                # Extract results from OpenAI response
                processed_results = process_openai_search_results(response, query)
                all_search_results.extend(processed_results)
                
            except Exception as e:
                logger.error(f"Error executing search '{query}': {str(e)}")
        
        # Log search result count
        logger.info(f"Found {len(all_search_results)} search results")
        
        # Step 5: Prepare search results for analysis
        formatted_results = []
        for i, result in enumerate(all_search_results[:15]):  # Limit to top 15 results
            query = result.get("query", "Unknown query")
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            source = result.get("source", "Unknown source")
            
          
                
            formatted_results.append(f"RESULT {i+1}\nQuery: {query}\nSource: {source}\nTitle: {title}\nContent: {content}\n")
        
        results_text = "\n\n".join(formatted_results)
        
        # Step 6: Create preliminary analysis to identify problems the company solves
        logger.info("Generating preliminary analysis to identify key problems solved")
        
        # Improved prompt to extract concise service descriptions for search queries
        problem_extraction_prompt = f"""Analyze the following search results about {company_name} and identify EXACTLY 3 core services or products this company provides. 

Search Results:
{results_text}

IMPORTANT INSTRUCTIONS:
1. Each service description must be 5-7 words MAXIMUM
2. Describe ACTUAL services the company provides, not generic capabilities
3. Focus on their CORE business offerings, not peripheral services
4. Ensure the services are SPECIFIC and CONCRETE, not abstract concepts
5. The services should align with the company's actual business model

FORMAT YOUR RESPONSE AS FOLLOWS (do not include any other text):
1. [5-7 word service description]
2. [5-7 word service description]
3. [5-7 word service description]

EXAMPLES OF GOOD RESPONSES FOR AN HR COMPANY:
1. Payroll processing and tax filing
2. Benefits administration and compliance
3. Employee onboarding HR software

EXAMPLES OF BAD RESPONSES:
1. They provide innovative solutions to improve business efficiency (too long/vague)
2. HR (too short/vague)
3. Customer-centric business transformation services (too vague)

YOUR RESPONSE MUST BE A SIMPLE NUMBERED LIST WITH SHORT CONCRETE PHRASES ONLY:"""
        
        # Extract core services using a constrained query to the LLM
        try:
            problems_response = llm.invoke(problem_extraction_prompt)
            problems_list = problems_response.content
            
            logger.info(f"Identified key services: {problems_list}")
            
            # Ultra-simplified extraction - just get the phrases without complex validation
            valid_problems = []
            
            # Basic parsing - get lines, remove numbering
            for line in problems_list.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Remove numbering pattern
                phrase = re.sub(r'^\d+\.?\s*', '', line).strip()
                
                # Only add non-empty phrases (simple validation)
                if phrase and len(phrase) > 2:
                    valid_problems.append(phrase)
            
            # Simple fallback - only if we got nothing at all
            if not valid_problems:
                valid_problems = [
                    f"{company_name} core services", 
                    f"{company_name} primary offerings", 
                    f"{company_name} solutions"
                ]
                
            # Limit to 3 items
            valid_problems = valid_problems[:3]
            
            logger.info(f"Service phrases for search: {valid_problems}")
                
        except Exception as e:
            logger.error(f"Error extracting services: {str(e)}")
            # Simple fallback
            valid_problems = [
                f"{company_name} core services", 
                f"{company_name} primary offerings", 
                f"{company_name} solutions"
            ]
            
        # Step 7: Conduct second-stage search for geographic complexity
        geographic_context_results = []
        
        # Only proceed with second search if target geography is specified
        if target_geography:
            logger.info(f"Executing second-stage search for geographic complexities in {target_geography}")
            
            # Generate simple, direct search queries combining the core service and geography
            geo_search_queries = []
            
            for phrase in valid_problems:
                # Create a clean search query without quotes or complex formatting
                geo_search_queries.append(f"{phrase} regulations {target_geography}")
                
            # Add a focused industry-specific search query
            geo_search_queries.append(f"business compliance requirements {target_geography}")
            
            # Execute geography-specific searches
            for query in geo_search_queries:
                try:
                    logger.info(f"Searching for geographic context: {query}")
                    search_query = f"Search for information about: {query}"
                    logger.info(f"Geographic search query sent to OpenAI: {search_query}")
                    response = search_tool.invoke(search_query)
                    
                    # Process and add to results
                    geo_results = process_openai_search_results(response, query)
                    geographic_context_results.extend(geo_results)
                    
                except Exception as e:
                    logger.error(f"Error executing geographic context search '{query}': {str(e)}")
            
            logger.info(f"Found {len(geographic_context_results)} geographic context search results")
            
            # Format the geographic context results
            geo_formatted_results = []
            for i, result in enumerate(geographic_context_results):
                query = result.get("query", "Unknown query")
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                source = result.get("source", "Unknown source")
                
                
                    
                geo_formatted_results.append(f"GEO CONTEXT {i+1}\nQuery: {query}\nSource: {source}\nTitle: {title}\nContent: {content}\n")
            
            # Add geographic context to results text if any was found
            if geo_formatted_results:
                geo_context_text = "\n\n".join(geo_formatted_results)
                results_text += "\n\n--- GEOGRAPHIC CONTEXT ---\n\n" + geo_context_text
                logger.info("Added geographic context to analysis input")
        
        # Prepare enhanced context section with identified services and other metadata
        enhanced_context = ""
        
        # Add identified services/products with the full LLM response
        if valid_problems:
            enhanced_context += f"\n\n--- IDENTIFIED CORE SERVICES ---\n"
            # Include the raw LLM response first
            enhanced_context += f"Raw LLM Analysis:\n{problems_list}\n\n"
            # Then include the extracted services
            service_list = "\n".join([f"- {service}" for service in valid_problems])
            enhanced_context += f"Extracted Services:\n{service_list}\n"
        
        # Add search process metadata
        enhanced_context += "\n\n--- SEARCH METADATA ---\n"
        enhanced_context += f"- Total initial search results: {len(all_search_results)}\n"
        enhanced_context += f"- Company name: {company_name}\n"
        enhanced_context += f"- Original search queries: {', '.join(search_queries)}\n"
        if target_geography:
            enhanced_context += f"- Target geography: {target_geography}\n"
            enhanced_context += f"- Geographic search queries: {', '.join(geo_search_queries)}\n"
            enhanced_context += f"- Geographic context results: {len(geographic_context_results)}\n"
        
        # Add search confidence information
        search_quality = min(len(all_search_results) / 15, 1.0)
        confidence_level = "High" if len(all_search_results) > 10 else "Medium" if len(all_search_results) > 5 else "Low"
        enhanced_context += f"- Search quality metric: {search_quality:.2f}\n"
        enhanced_context += f"- Confidence level: {confidence_level}\n"
        
        # Add the enhanced context to results_text
        results_text += enhanced_context
        
        # Step 8: Create analysis prompt using a simplified approach
        
        # Single streamlined system prompt that works for all cases
        system_prompt = f"""You are an elite business analyst creating precise, actionable company profiles for B2B sales teams.
            
IMPORTANT: Provide a comprehensive understanding of the company's overall operations. Then, identify how the problems this company solves manifest specifically in {target_geography} - focus on unique regulatory, business, or cultural challenges in {target_geography} that make the company's solutions particularly valuable there.
            
Your analysis MUST be:
1. SPECIFIC: Avoid generic statements unless you can detail HOW they implement these concepts
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
- Trigger events that create sales opportunities"""

        # Simplified human prompt with streamlined instructions
        human_prompt = f"""SALES DEEP DIVE: Company at {company_url}
        
SKIP THE FLUFF. As a killer sales professional analyzing {company_name}, you need to understand:
1. What problems they ACTUALLY solve (not what they claim)
2. How they make money
3. Their TRUE differentiators (cut through the marketing BS)
4. Who ACTUALLY buys from them and why
5. How their solutions address SPECIFIC challenges in {target_geography}

Search Results:
{results_text}

Based on these results, use the Route-Ruin-Multiply framework to create a KILLER sales-oriented company profile that addresses:

1. PAINS & SOLUTIONS: 3-4 SPECIFIC pain points this company solves - the actual business problems that keep decision-makers up at night. Pair each pain with how they specifically solve it.

2. DIFFERENTIATION: What truly makes them unique - find 2-3 genuine differentiators competitors lack that customers actually care about. Cut through the BS.

3. BUYER PROFILE: Who are their ACTUAL buyers? What titles, industries, company sizes? What motivates these buyers?

4. MARKET FOCUS: Their primary industry and customer segments across all geographies.

5. OBJECTION INSIGHTS: Based on their positioning, what are 1-2 likely objections prospects raise? Think about cost, implementation time, competing priorities.

6. SALES APPROACH: How do they sell? Direct? Channel? Product-led? What's their conversion strategy and pricing model?

7. GEOGRAPHY-SPECIFIC CHALLENGES: Identify specific problems in {target_geography} that this company's solutions address. Focus on unique regulatory requirements, business challenges, or cultural factors in {target_geography} that make this company's solutions particularly valuable there.

Finally, synthesize all of this into ONE KILLER PARAGRAPH capturing the company's essence as if explaining to a CEO in an elevator. Make every word count, kill the fluff, and add something truly insightful.

Also provide a one-line statement of their primary industry focus.
"""
        
        # Generate analysis
        logger.info("Generating company analysis")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        analysis = response.content
        
        # Log the raw analysis received from the LLM
        logger.info("--- Raw Analysis Received from LLM ---")
        logger.info(analysis)
        logger.info("--------------------------------------")

        # Step 10: Create final output
        result = {
            "name": extract_company_name(analysis) or company_name,
            "description": extract_description(analysis),
            "url": company_url,
            "target_geography": target_geography if target_geography else "Global",
            "search_quality": min(len(all_search_results) / 15, 1.0),  # Simple quality metric
            "confidence": "High" if len(all_search_results) > 10 else "Medium" if len(all_search_results) > 5 else "Low",
            "analysis": analysis,
            # Add enhanced metadata to the result
            "extracted_services": valid_problems,
            "search_queries": search_queries,
            "geographic_queries": geo_search_queries if target_geography else []
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
    except Exception as e:
        logger.error(f"Error extracting domain: {str(e)}")
        # If URL parsing fails, make one more attempt
        try:
            # Try to extract anything before the first period
            domain_part = url.split('//')[-1].split('.')[0]
            if domain_part and domain_part != "www":
                return domain_part.capitalize()
        except:
            pass
        
        # Return a placeholder if all extraction attempts fail
        return "Company"

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

def process_openai_search_results(response, original_query):
    """
    Process OpenAI web search results, capturing the full synthesized text.
    
    Args:
        response: The response from OpenAI web search
        original_query: The original search query
        
    Returns:
        List of search results containing the full text and cited sources.
    """
    results = []
    
    try:
        # Check if the response object and content attribute exist
        if not hasattr(response, 'content') or not response.content:
            logger.warning(f"Received empty or invalid response content for query: {original_query}")
            return results
            
        # Process each content block (usually just one text block for web search)
        for block in response.content:
            if block.get('type') == 'text':
                text_content = block.get('text', '')
                if not text_content:
                    continue # Skip empty text blocks
                    
                # Extract all citation URLs from annotations
                annotations = block.get('annotations', [])
                citation_urls = []
                primary_title = "Web Search Synthesis" # Default title
                
                for annotation in annotations:
                    if annotation.get('type') == 'url_citation':
                        url = annotation.get('url')
                        if url:
                            citation_urls.append(url)
                        # Try to get a primary title from the first citation
                        if annotation.get('title') and primary_title == "Web Search Synthesis":
                            primary_title = annotation.get('title')
                            
                # Create one result dictionary containing the full text
                results.append({
                    "query": original_query,
                    "title": primary_title, 
                    "content": text_content, # Use the full synthesized text
                    "source": ", ".join(citation_urls) if citation_urls else "Synthesized (No specific sources cited)" # List all sources
                })
                
    except Exception as e:
        logger.error(f"Error processing OpenAI search results for query '{original_query}': {str(e)}")
        # Add a fallback result indicating the processing error
        results.append({
            "query": original_query,
            "title": "Error Processing Search Results",
            "content": f"Failed to process search response: {str(e)}",
            "source": "Error"
        })
    
    # Log the number of processed result blocks
    logger.info(f"Processed {len(results)} result block(s) for query: {original_query}")
    return results

# Function for direct integration with the main application
async def analyze_company_context(
    company_url: str, 
    company_name: str,
    model_name: str = "gpt-4.1", 
    target_geography: str = None
) -> Dict[str, Any]:
    """
    Analyze a company based on its website URL using OpenAI web search.
    
    Args:
        company_url: URL of the company website
        company_name: Name of the company (explicitly provided)
        model_name: Name of the model to use for analysis (defaults to GPT-4.1)
        target_geography: User-specified target geography or market
        
    Returns:
        Dictionary containing company context information
    """
    try:
        logger.info(f"Starting company analysis for {company_name} using simple analyzer with model {model_name}")
        if target_geography:
            logger.info(f"Analysis will include specified target geography: '{target_geography}'")
            
        # Call the underlying analyze_company function (synchronous, but we're in an async function)
        import asyncio
        company_context = await asyncio.to_thread(
            analyze_company,
            company_url=company_url,
            company_name=company_name,
            target_geography=target_geography,
            model_name=model_name,
        )
        
        logger.info(f"Completed company analysis for {company_name}")
        return company_context
        
    except Exception as e:
        logger.error(f"Error in company analysis: {str(e)}")
        # Return a minimal context if analysis fails
        error_context = {
            "name": company_name,
            "description": f"Error analyzing company: {str(e)}",
            "url": company_url,
            "target_geography": target_geography if target_geography else "Unknown",
            "error": str(e)  # Add explicit error field
        }
        logger.info(f"Created fallback context due to error: {error_context}")
        return error_context 
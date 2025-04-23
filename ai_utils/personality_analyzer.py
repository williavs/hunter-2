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
import openai # Add direct OpenAI client import
import streamlit as st

# Import fuzzy matching library
from rapidfuzz import fuzz

# Updated imports for the current LangGraph API
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

class ContactInfo(BaseModel):
    """Contact information for a person to analyze."""
    id: str = Field(description="Unique identifier for the contact")
    name: str = Field(description="Full name of the contact")
    email: Optional[str] = Field(default=None, description="Email address")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    twitter_url: Optional[str] = Field(default=None, description="Twitter/X profile URL")
    personal_website: Optional[str] = Field(default=None, description="Personal website URL")
    company: Optional[str] = Field(default=None, description="Company name")
    title: Optional[str] = Field(default=None, description="Job title")
    website_content: Optional[str] = Field(default=None, description="Scraped website content")
    company_linkedin_url: Optional[str] = Field(default=None, description="Company LinkedIn URL")
    facebook_url: Optional[str] = Field(default=None, description="Facebook profile URL")
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ContactInfo":
        """
        Create a ContactInfo object from a row dictionary with proper fallbacks.
        
        This method ensures required fields are always present and handles missing data gracefully.
        """
        data = row.copy()
        # Assign or generate a unique id
        if "id" in data and data["id"]:
            data["id"] = str(data["id"])
        else:
            import uuid
            data["id"] = str(uuid.uuid4())
        # Name logic as before
        if "full_name" in data and data["full_name"]:
            data["name"] = data["full_name"]
        elif "name" not in data or not data["name"]:
            if "first_name" in data and "last_name" in data and data["first_name"] and data["last_name"]:
                data["name"] = f"{data['first_name']} {data['last_name']}".strip()
            elif "email" in data and data["email"]:
                email_parts = data["email"].split('@')
                if len(email_parts) > 0:
                    data["name"] = email_parts[0].replace(".", " ").title()
            elif "company" in data and data["company"]:
                data["name"] = f"Contact at {data['company']}"
            else:
                data["name"] = f"Contact-{data['id'][:8]}"
        title = data.get("title", None)
        if title is not None:
            if pd.isna(title):
                data["title"] = None
            elif not isinstance(title, str):
                data["title"] = str(title)
        model_fields = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in model_fields}
        return cls(**filtered_data)

class AnalysisResult(BaseModel):
    """Results of personality analysis."""
    contact_id: str  # This will now be the unique id
    personality_analysis: str
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
    
    # Class variable for shared search cache across all instances
    _global_search_cache = {}
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 ai_mode: str = "Proxy",
                 openai_api_key: Optional[str] = None,
                 proxy_api_key: Optional[str] = None,
                 proxy_base_url: str = "https://llm.data-qa.justworks.com"
              ):
        """
        Initialize the personality analyzer.
        Args:
            max_concurrent: Maximum number of concurrent analyses
            ai_mode: Mode of operation (Proxy or OpenAI)
            openai_api_key: API key for OpenAI mode
            proxy_api_key: API key for Proxy mode
            proxy_base_url: Base URL for Proxy mode
        """
        self.max_concurrent = max_concurrent
        self.ai_mode = ai_mode
        self.openai_api_key = openai_api_key
        self.proxy_api_key = proxy_api_key
        self.proxy_base_url = proxy_base_url
        # Initialize search cache using class variable
        self._search_cache = PersonalityAnalyzer._global_search_cache
        # Set up tools based on configuration
        self._initialize_tools()
        # Build the workflow
        self.workflow = self._build_workflow()
    
    def _initialize_tools(self):
        """Initialize the search tool and LLM based on the current configuration."""
        mode = self.ai_mode
        if mode == "OpenAI":
            api_key = self.openai_api_key
            model = "gpt-4.1"
            base_url = None
        else:
            api_key = self.proxy_api_key
            model = "openai/gpt-4.1"
            base_url = self.proxy_base_url
        if not api_key:
            raise ValueError(f"No API key found for mode: {mode}")
        llm_kwargs = dict(
            api_key=api_key,
            model=model,
            temperature=0.1,
            streaming=False,
            timeout=90
        )
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        logger.debug(f"Initializing LLM with model: {model} (mode: {mode})")
        # Initialize direct OpenAI client here as well
        try:
            if mode == "OpenAI":
                self.direct_openai_client = openai.OpenAI(
                    api_key=api_key,
                    # No base_url for OpenAI direct
                )
            else:
                self.direct_openai_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.proxy_base_url + "/v1" # Direct client needs /v1
                )
        except Exception as client_error:
            logger.error(f"Failed to initialize direct OpenAI client in PersonalityAnalyzer: {client_error}")
            self.direct_openai_client = None 
            
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
    
    async def _search_task(self, state: PersonalityState) -> PersonalityState:
        """Execute comprehensive web search for the contact using direct OpenAI client."""
        contact_data = state["contact_data"]
        company_context = state.get("company_context", {})
        
        # Check if direct client was initialized successfully
        if not self.direct_openai_client:
            error_msg = "Direct OpenAI client not initialized. Cannot perform web search."
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["search_results"] = [] # Ensure search_results is initialized
            state["search_queries"] = [] # Ensure search_queries is initialized
            return state # Return early or handle as appropriate

        # Check if website content is missing
        has_website_content = contact_data.get("website_content", "").strip() != ""
        
        # Extract essential contact information
        name = contact_data.get("name", "Unknown")
        company = contact_data.get("company", "")
        title = contact_data.get("title", "")
        
        # Generate comprehensive search prompts
        search_prompts = []
        
        # Format company context as a simple block if available
        company_context_block = ""
        company_name = ""
        if company_context:
            company_name = company_context.get('company_name', '')
            company_url = company_context.get('url', '')
            company_description = company_context.get('description', '')
            
            company_context_block = f"""
Company Information:
Name: {company_name}
URL: {company_url}
Description: {company_description}
"""
            # Add any additional relevant fields
            for key, value in company_context.items():
                if key not in ['company_name', 'name', 'url', 'description'] and value:
                    company_context_block += f"{key.replace('_', ' ').title()}: {value}\n"
        
        # General professional background and personality prompt
        background_prompt = f"""Search for comprehensive information about {name}, who is {title} at {company}.
Focus on:
1. Their professional background, experience, and career trajectory
2. Their communication style, writing samples, and speaking engagements
3. Their specific expertise and notable accomplishments
4. Their decision-making process and leadership approach
5. Their sphere of influence and relationships within their organization
6. Their likely career aspirations and professional motivations
7. Any signs of personal values or emotional drivers that influence decisions
"""
        search_prompts.append(background_prompt)
        
        # Pain points and challenges prompt
        challenges_prompt = f"""Search for information about the specific challenges and pain points that {name} 
as {title} at {company} likely faces in their role. Focus on:
1. Industry-specific problems related to their position
2. Common challenges for {title} roles in companies like {company}
3. Recent business changes or transformations at {company}
4. Regulatory or compliance issues they might be dealing with
5. Technology or process challenges in their field
6. Infer their daily frustrations and emotional pain points based on their role
7. Assess whether they have decision-making authority for solutions to these challenges

Additionally, determine:
- Do they appear to be an influencer or decision-maker for operational/administrative solutions?
- Are they likely the right contact for HR, payroll, or administrative technology decisions?
- If not, who in their organization would likely be the better contact?
"""
        
        # If we have company context, add it as a block
        if company_context_block:
            challenges_prompt += f"\nAdditional context about their company: {company_context_block}"
            
        search_prompts.append(challenges_prompt)
        
        # Generate company-specific prompt (always, not conditionally)
        company_prompt = f"""Search for information about how {name} ({title} at {company}) 
might relate to the following company context:

{company_context_block}

Focus on:
1. Based on their role, would this person be involved in decisions about solutions described above?
2. Are there any signs they would or would NOT be interested in these solutions?
3. What specific problems from the company description would they personally care most about?
4. Who else at their company might be a better target if they're not the right person?
5. What would motivate this person to consider or champion such solutions?

Important: Make reasonable inferences even with limited information. Look for clues about their:
- Decision-making authority regarding operational or administrative solutions
- Likely interest in efficiency, compliance, or team enablement
- Pain points that the described services might address
- Emotional and career motivations that might influence their interest
"""           
        search_prompts.append(company_prompt)
        
        # --- Start of Search Execution Update ---
        search_results_list = [] # Store results in a list of dicts like simple_analyzer
        web_search_tool_def = [{"type": "web_search_preview"}] # Tool definition
        # Fix: Use correct model name for OpenAI vs Proxy mode
        if self.ai_mode == "OpenAI":
            model_for_search = "gpt-4.1"
        else:
            model_for_search = "openai/gpt-4.1"

        # Execute searches in parallel using direct client
        async def execute_search_direct(prompt: str):
            try:
                query_str = str(prompt).strip().lower()
                if query_str in self._search_cache:
                    logger.debug(f"Using cached result for search: {query_str[:50]}...")
                    # Return cached result in the expected new format if needed
                    # For now, just return the cached dict directly if format matches
                    return self._search_cache[query_str] 

                logger.debug(f"Executing new direct search: {query_str[:50]}...")
                response = await asyncio.to_thread(
                    self.direct_openai_client.responses.create,
                    model=model_for_search,
                    tools=web_search_tool_def,
                    input=prompt
                )
                
                # Process direct OpenAI response
                text_content = ""
                search_sources = []
                if response.output:
                    for item in response.output:
                        if item.type == "message" and item.content:
                            for content_block in item.content:
                                if content_block.type == "output_text":
                                    text_content += content_block.text + "\n"
                                    if hasattr(content_block, 'annotations'):
                                        for annotation in content_block.annotations:
                                            if annotation.type == 'url_citation':
                                                search_sources.append(annotation.url)

                result_item = {
                    "query": prompt,
                    "title": "Web Search Result", # Simplified title
                    "content": text_content.strip(),
                    "source": ", ".join(search_sources) or "Web Search"
                }
                
                # Cache the processed result
                if text_content: # Only cache if we got content
                    self._search_cache[query_str] = result_item
                
                return result_item
            except Exception as e:
                logger.error(f"Direct search error for prompt: '{prompt[:50]}...': {str(e)}")
                return {"query": prompt, "error": str(e), "content": f"Search Error: {str(e)}", "source": "Error"}
        
        semaphore = asyncio.Semaphore(3)  # Limit concurrency
        async def search_with_semaphore_direct(prompt):
            async with semaphore:
                return await execute_search_direct(prompt)

        tasks = [search_with_semaphore_direct(prompt) for prompt in search_prompts]
        search_outputs = await asyncio.gather(*tasks)
        
        # Collect results (append directly to list)
        for output in search_outputs:
            if "error" not in output:
                 search_results_list.append(output)
            else:
                logger.warning(f"Search error encountered: {output.get('error')}")
                state["errors"].append(f"Search error: {output.get('error')}")
                # Optionally add error result to list for context?
                search_results_list.append(output)
        
        # --- End of Search Execution Update ---
        
        # Store results in state (adjusting format if needed for downstream)
        # The analysis task now needs to handle a list of dicts directly
        state["search_results"] = search_results_list 
        
        # Add a system message indicating search completion
        state["messages"].append({
            "role": "system",
            "content": f"Search completed. Found {len(search_results_list)} results."
        })
        
        # Keep original search prompts for reference
        state["search_queries"] = [prompt for prompt in search_prompts]
        
        return state
    
    def _process_openai_search_results(self, response):
        """
        Process OpenAI web search results into a consistent format for analysis.
        
        Args:
            response: The response from OpenAI web search
            
        Returns:
            List of search results in a standardized format
        """
        results = []
        
        try:
            # Extract content blocks from response
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if block.get('type') == 'text':
                        text_content = block.get('text', '')
                        
                        # Extract annotations (citations)
                        annotations = block.get('annotations', [])
                        
                        # If there are no annotations, create a single result with the full text
                        if not annotations:
                            results.append({
                                "title": "Web Search Synthesis",
                                "content": text_content,
                                "source": "Synthesized content without specific sources"
                            })
                        else:
                            # Process annotations to create results with proper citations
                            for annotation in annotations:
                                if annotation.get('type') == 'url_citation':
                                    url = annotation.get('url', '')
                                    title = annotation.get('title', 'Untitled Source')
                                    
                                    # Get the cited text
                                    start_idx = annotation.get('start_index', 0)
                                    end_idx = annotation.get('end_index', len(text_content))
                                    
                                    # Ensure indices are within bounds
                                    start_idx = max(0, min(start_idx, len(text_content)))
                                    end_idx = max(0, min(end_idx, len(text_content)))
                                    
                                    # Extract the cited text with additional context
                                    # Include more context around the citation to ensure important information is preserved
                                    context_start = max(0, start_idx - 250)
                                    context_end = min(len(text_content), end_idx + 250)
                                    
                                    # Extract with expanded context
                                    full_content = text_content[context_start:context_end]
                                    
                                    results.append({
                                        "title": title,
                                        "content": full_content,
                                        "source": url
                                    })
                            
                            # Also add the full synthesized text as a single result for completeness
                            # This ensures we don't lose any important context that might be between citations
                            results.append({
                                "title": "Complete Synthesized Response",
                                "content": text_content,
                                "source": "Full search synthesis"
                            })
            
            # If we didn't get any results from annotations, use the full response text
            if not results and hasattr(response, 'text'):
                results.append({
                    "title": "Web Search Response",
                    "content": response.text(),
                    "source": "OpenAI web search synthesis"
                })
                
        except Exception as e:
            logger.error(f"Error processing OpenAI search results: {str(e)}")
            # Add a fallback result with error information
            results.append({
                "title": "Error processing results",
                "content": f"An error occurred while processing search results: {str(e)}",
                "source": "Error"
            })
        
        return results
    
    def _process_website_content(self, content: str, company_name: Optional[str] = None) -> str:
        """
        Process website content to ensure important information is preserved.
        
        Args:
            content: The original website content
            company_name: Optional company name to ensure mentions are preserved
            
        Returns:
            Processed content with important sections preserved
        """
        if not content:
            return ""
            
        # If content is short enough, return it as is
        if len(content) <= 2000:
            return content
            
        # Extract important sections
        important_sections = []
        
        # Always include the first 800 characters (likely contains important intro information)
        important_sections.append(content[:800])
        
        # Check for company name mentions if provided
        if company_name and company_name.strip():
            company_lower = company_name.lower()
            remaining_content = content[800:-800]
            
            # Find all company name mentions
            start_idx = 0
            while True:
                idx = remaining_content.lower().find(company_lower, start_idx)
                if idx == -1:
                    break
                    
                # Extract a section around the company mention (300 chars before and after)
                section_start = max(0, idx - 300)
                section_end = min(len(remaining_content), idx + len(company_name) + 300)
                section = remaining_content[section_start:section_end]
                
                important_sections.append(f"... {section} ...")
                start_idx = idx + len(company_name)
                
                # Limit to 3 company mentions to avoid excessive content
                if len(important_sections) >= 4:  # 1 intro + 3 mentions
                    break
        
        # Always include the last 800 characters (likely contains important concluding information)
        important_sections.append(content[-800:])
        
        # Join sections with separators
        return "\n...[Content truncated]...\n".join(important_sections)
    
    def _analysis_task(self, state: PersonalityState) -> PersonalityState:
        """Analyze search results to generate personality insights relevant to company context."""
        contact = state["contact"]
        contact_data = state["contact_data"]
        search_results = state["search_results"]
        company_context = state.get("company_context", {})
        contact_id = contact.get("id", "")
        contact_name = contact.get("name", "Unknown")
        # Check if website content was present (assuming it might be added to search_results)
        has_website_content = any(r.get("source") == "website" for r in search_results if isinstance(r, dict))

        if not search_results:
            # Create a more informative message based on whether website content was available
            if not has_website_content:
                analysis_message = "No website content was provided for this contact. Analysis is based solely on search results, but no search results were found."
            else:
                analysis_message = "Insufficient data for analysis. No search results were found despite website content being available."
                
            state["errors"].append("No search results available for analysis")
            state["analysis"] = {
                "contact_id": contact_id,
                "personality_analysis": analysis_message,
                "search_queries_used": [],
                "search_results": []
            }
            state["complete"] = True
            return state
        
        # Prepare search results for prompt - ADAPTED FOR LIST OF DICTS
        results_text = ""
        for i, result_item in enumerate(search_results):
             source = result_item.get('source', 'Unknown Source')
             title = result_item.get('title', 'No Title')
             content = result_item.get('content', 'No content')
             query = result_item.get('query', 'Original query unknown') # Get query info

             results_text += f"RESULT {i+1} (From Query: {query[:100]}...)\nSource: {source}\nTitle: {title}\n"

             # Truncation logic (can keep or adjust)
             company_name = company_context.get('company_name')
             if len(content) > 10000:
                 first_part = content[:6000]
                 last_part = content[-6000:]
                 middle_part = ""
                 if company_name and company_name.lower() in content.lower()[6000:-6000]:
                     middle_idx = content.lower()[6000:-6000].find(company_name.lower()) + 6000
                     start_idx = max(0, middle_idx - 150)
                     end_idx = min(len(content), middle_idx + len(company_name) + 150)
                     middle_part = f"... [Important company mention: {content[start_idx:end_idx]}] ..."
                 results_text += f"Content: {first_part}... {middle_part} ...{last_part}\n\n"
             else:
                 results_text += f"Content: {content}\n\n"
        
        # Prepare company context text separately
        company_context_text = ""
        if company_context:
            company_context_text = """
My Company Context:
Below is information about my company, our target market, and the specific problems we solve.
When creating the personality analysis, you MUST directly connect the contact's pain points 
to the specific problems our company solves. The Route-Ruin-Multiply analysis should explicitly 
show how our solutions address their challenges.
IMPORTANT: ALWAYS include our company's full legal name without abbreviation or truncation:

"""
            # Extract key fields for easier processing
            company_name = company_context.get('company_name', '')
            company_description = company_context.get('description', '')
            
            # Create a structured format to ensure clarity
            context_details = [
                f"Company Name: {company_name}",
                f"Description: {company_description}"
            ]
            
            # Add any additional fields present in the context
            for key, value in company_context.items():
                if key not in ['company_name', 'description'] and value:
                    context_details.append(f"{key.replace('_', ' ').title()}: {value}")
            
            # Join all context details with newlines for clarity
            company_context_text += "\n".join(context_details)
            
            # Also include the full JSON as a backup to ensure no information is lost
            company_context_text += "\n\nFull Company Context:\n" + json.dumps(company_context, indent=2)
        
        # Create analysis prompt using results_text
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
                    - IMPORTANT: Assess whether they likely have decision-making authority for solutions like ours
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
                **IMPORTANT FUNCTION REQUIREMENT: NO CONVERSATIONAL TEXT OR INTROS- GET RIGHT TO THE POINT - THIS IS NOT A CONVERSATION - FUNCTION AS A TOOL

                **CRITICAL DECISION-MAKER ASSESSMENT: 
                At the VERY BEGINNING of your analysis, you MUST clearly categorize the contact using ONE of these five designations:

                1. "PRIMARY DECISION-MAKER: [Name] appears to be a primary decision-maker for our solutions because [brief evidence]."

                2. "STRONG INFLUENCER: [Name] is likely a strong influencer but not the final decision-maker because [brief reason]. They can champion solutions to the actual decision-makers, typically [role types]."

                3. "POTENTIAL INFLUENCER: [Name] could be an influencer in the evaluation process because [brief reason], though their exact level of authority is unclear. Decisions typically require [role types]."

                4. "INFORMATION GATHERER: [Name] appears to be in an information-gathering role because [brief reason]. They may research solutions but decisions are typically made by [role types]."

                5. "DEFINITELY THE WRONG PERSON:  [You fill in the blanks]."

                For any designation other than PRIMARY DECISION-MAKER, focus more on company-wide benefits rather than personal benefits. DO NOT speculate about specific individuals who might be better contacts or fabricate names, titles, or people not mentioned in the data. Keep your assessment honest, evidence-based, and direct.
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
            
            # Create analysis result 
            analysis = {
                "contact_id": contact_id,
                "personality_analysis": response.content,
                "search_queries_used": state.get("search_queries", []),
                "search_results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("source", ""),
                        "snippet": r.get("content", "")[:300]
                    } for r in search_results if isinstance(r, dict) and "error" not in r
                ]
            }
            
            state["analysis"] = analysis
            
        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["analysis"] = {
                "contact_id": contact_id,
                "personality_analysis": "Error during analysis",
                "search_queries_used": [],
                "search_results": []
            }
        
        state["complete"] = True
        return state
    
    def _build_workflow(self):
        """Build the LangGraph workflow for personality analysis using StateGraph."""
        # Create a state graph with the PersonalityState type
        workflow = StateGraph(PersonalityState)
        
        # Add nodes for each task - removing the planning task
        workflow.add_node("search", self._search_task)
        workflow.add_node("analysis_task", self._analysis_task)
        
        # Define the edges to create a linear flow
        workflow.add_edge("search", "analysis_task")
        workflow.add_edge("analysis_task", END)
        
        # Set the entry point directly to search
        workflow.set_entry_point("search")
        
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
                contact_id=contact.id,
                personality_analysis="Analysis failed",
                error="Workflow did not produce an analysis"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing contact {contact.id}: {str(e)}")
            return AnalysisResult(
                contact_id=contact.id,
                personality_analysis="Analysis failed",
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
                    "id": contact.id,
                    "name": contact.name,
                    "company": contact.company or "",
                    "title": contact.title or "",
                    "linkedin_url": contact.linkedin_url or "",
                    "twitter_url": contact.twitter_url or "",
                    "personal_website": contact.personal_website or "",
                    "website_content": self._process_website_content(contact.website_content, contact.company) if contact.website_content else "",
                    "company_linkedin_url": contact.company_linkedin_url or "",
                    "facebook_url": contact.facebook_url or "",
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
                if isinstance(result, Exception):
                    logger.error(f"Error in analysis task: {str(result)}")
                    contact = contacts[i]
                    result = AnalysisResult(
                        contact_id=contact.id,
                        personality_analysis="Analysis failed",
                        error=str(result)
                    )
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
async def analyze_personality(df: pd.DataFrame, company_context: Optional[Dict[str, Any]] = None,
    ai_mode: str = "Proxy",
    openai_api_key: Optional[str] = None,
    proxy_api_key: Optional[str] = None,
    proxy_base_url: str = "https://llm.data-qa.justworks.com"
) -> pd.DataFrame:
    """
    Analyze the personalities of contacts in a DataFrame.
    Args:
        df: DataFrame containing contact information and website content
        company_context: Optional company context to provide for analysis
        ai_mode: Mode of operation (Proxy or OpenAI)
        openai_api_key: API key for OpenAI mode
        proxy_api_key: API key for Proxy mode
        proxy_base_url: Base URL for Proxy mode
    Returns:
        DataFrame containing the original data plus personality analysis results
    """
    try:
        analyzer = PersonalityAnalyzer(
            max_concurrent=10,
            ai_mode=ai_mode,
            openai_api_key=openai_api_key,
            proxy_api_key=proxy_api_key,
            proxy_base_url=proxy_base_url
        )
        contacts = []
        df = df.copy()
        if "id" not in df.columns:
            import uuid
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        for _, row in df.iterrows():
            contact_dict = row.to_dict()
            contact = ContactInfo.from_row(contact_dict)
            contacts.append(contact)
        results = await analyzer.analyze_personalities_with_contacts(contacts, company_context)
        result_df = df.copy()
        for contact_id, result in results.items():
            if contact_id in result_df["id"].values:
                idx = result_df.index[result_df["id"] == contact_id][0]
                result_df.at[idx, "personality_analysis"] = result.personality_analysis
                if result.error:
                    result_df.at[idx, "personality_analysis"] = f"Error: {result.error}"
        return result_df
    except Exception as e:
        logger.error(f"Error in analyze_personality: {str(e)}")
        result_df = df.copy()
        result_df["personality_analysis"] = f"Error in analysis: {str(e)}"
        return result_df 
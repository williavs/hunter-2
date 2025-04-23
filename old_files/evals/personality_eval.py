"""
Personality Analysis Evaluation Framework

This module provides a comprehensive evaluation framework for assessing the quality
and effectiveness of personality analyses generated for sales and outreach purposes.
The evaluation focuses on practical application, conversation guidance, and actionable
insights rather than theoretical accuracy.

Usage:
    python evals/personality_eval.py

Requirements:
    - langsmith
    - pydantic
    - langchain
    - openrouter API key
"""

import os
import sys
import json
import logging
import asyncio
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langsmith import Client
import langsmith.utils

# Maximum number of examples to evaluate (set to None for all)
MAX_EXAMPLES = 10

# Add the parent directory to sys.path so we can import from ai_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from ai_utils.personality_analyzer import PersonalityAnalyzer, ContactInfo
from old_files.company_context_workflow import ChatOpenRouter, analyze_company_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith client
client = Client()

# Get API keys from environment
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

# Initialize the OpenRouter client with the haiku model
llm = ChatOpenRouter(
    openai_api_key=openrouter_api_key,
    model="anthropic/claude-3.5-haiku-20241022:beta",
    temperature=0.1,
    streaming=False
)

# Define evaluation schema for personality analysis
class PersonalityAnalysisEval(BaseModel):
    """
    Evaluation schema for personality analysis with practical sales application focus.
    
    Each dimension is scored on a 1-5 scale where:
    1 = Poor/Ineffective
    2 = Basic/Limited
    3 = Adequate/Functional
    4 = Strong/Effective
    5 = Exceptional/Outstanding
    """
    practical_relevance: int = Field(
        description="How relevant and applicable the insights are for real sales conversations",
        ge=1, le=5
    )
    conversation_guidance: int = Field(
        description="Quality of specific conversation tactics, openers, and approaches provided",
        ge=1, le=5
    )
    pain_point_identification: int = Field(
        description="Accuracy and depth of identified professional pain points and challenges",
        ge=1, le=5
    )
    personality_insight_depth: int = Field(
        description="Depth and nuance of personality insights beyond generic observations",
        ge=1, le=5
    )
    company_context_integration: int = Field(
        description="How well the analysis integrates available company context",
        ge=1, le=5
    )
    actionability: int = Field(
        description="How immediately actionable the recommendations are for sales outreach",
        ge=1, le=5
    )
    overall_value: int = Field(
        description="Overall value of the analysis for improving sales conversations",
        ge=1, le=5
    )
    strengths: str = Field(
        description="Specific strengths of the analysis"
    )
    improvement_areas: str = Field(
        description="Specific areas where the analysis could be improved"
    )
    suggested_enhancements: str = Field(
        description="Concrete suggestions for enhancing the analysis quality"
    )

# Create a focused test dataset with gtmWizards (replacing Justworks)
personality_analysis_examples = [
    {
        "contact": {
            "name": "Benjamin Matlin",
            "email": "benjamin@adentro.com",
            "title": "Chief Executive Officer",
            "company": "Adentro",
            "linkedin_url": "https://www.linkedin.com/in/benjamin-matlin-a9b9732/"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://www.adentro.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Entrepreneurial CEO focused on growth and commercial partnerships. Values solutions that help build pipeline, enhance go-to-market strategy, and scale revenue generation while maintaining focus on core business activities."
    },
    {
        "contact": {
            "name": "Matt Lilya",
            "email": "mlilya@ventusgns.com",
            "title": "Business Professional",
            "company": "Ventus",
            "linkedin_url": "https://www.linkedin.com/in/matt-lilya-7b305142"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.ventusgns.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Business professional at Ventus likely focused on company growth. Values effective lead generation, sales pipeline development, and go-to-market strategies that drive revenue growth."
    },
    {
        "contact": {
            "name": "Lex Sivakumar",
            "email": "lex@regal.ai",
            "title": "Business Professional",
            "company": "Regal.ai",
            "linkedin_url": "https://www.linkedin.com/in/lexsivakumar"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://www.regal.ai",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at an AI company likely concerned with scaling operations efficiently. Values specialized lead generation and go-to-market strategies tailored to tech-forward growing companies."
    },
    {
        "contact": {
            "name": "Brian Powers",
            "email": "brian.powers@stensul.com",
            "title": "Business Professional",
            "company": "Stensul",
            "linkedin_url": "https://www.linkedin.com/in/brianjpowers"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://stensul.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Business professional at Stensul interested in optimizing operations. Values effective pipeline generation strategies and targeted lead generation that scales with company growth."
    },
    {
        "contact": {
            "name": "Ro Facundo",
            "email": "ro.facundo@eigentech.com",
            "title": "Business Professional",
            "company": "Eigen Technologies",
            "linkedin_url": "https://www.linkedin.com/in/ro-facundo"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://eigentech.com/",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a technology company focused on operational efficiency. Values strategic go-to-market planning, ICP development, and multi-channel outreach that drives qualified leads."
    },
    {
        "contact": {
            "name": "Cameron Evans",
            "email": "cevans@aspiretransforms.com",
            "title": "Business Professional",
            "company": "Aspire Technology Partners",
            "linkedin_url": "https://www.linkedin.com/in/cameronwevans"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.aspiretransforms.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Technology professional interested in operational efficiency. Values comprehensive go-to-market strategies and pipeline development that support company growth and transformation."
    },
    {
        "contact": {
            "name": "Mahaboob Basha",
            "email": "basha@excelgens.com",
            "title": "Business Professional",
            "company": "ExcelGens, Inc.",
            "linkedin_url": "https://www.linkedin.com/in/mahaboobbasha"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://www.excelgens.com/",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a tech services company focused on operational excellence. Values lead generation, ICP development, and sales strategy solutions that support business growth."
    },
    {
        "contact": {
            "name": "Victoria Zona",
            "email": "victoria.zona@opengear.com",
            "title": "Business Professional",
            "company": "Opengear",
            "linkedin_url": "https://www.linkedin.com/in/victoriazona"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.opengear.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a network solutions company interested in optimizing operations. Values efficient pipeline building and multi-channel outreach strategies that generate qualified leads."
    },
    {
        "contact": {
            "name": "Bobby Mohr",
            "email": "bobby@pinecone.io",
            "title": "Business Professional",
            "company": "Pinecone",
            "linkedin_url": "https://www.linkedin.com/in/robertdmohr"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://www.pinecone.io/",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a vector database company focused on scaling efficiently. Values specialized lead generation and go-to-market strategies tailored for fast-growing tech companies."
    },
    {
        "contact": {
            "name": "Ashley Jordan",
            "email": "ashley.jordan@sundaysky.com",
            "title": "Business Professional",
            "company": "SundaySky",
            "linkedin_url": "https://www.linkedin.com/in/ashley-jordan-42380531"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.sundaysky.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a video technology company interested in operational efficiency. Values effective lead generation and comprehensive go-to-market strategies that support company growth."
    },
    {
        "contact": {
            "name": "John Wehren",
            "email": "john.wehren@cubesoftware.com",
            "title": "Business Professional",
            "company": "Cube",
            "linkedin_url": "https://www.linkedin.com/in/john-wehren"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://www.cubesoftware.com/",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a financial planning software company focused on operational excellence. Values pipeline building and ICP development that targets specific verticals and drives qualified leads."
    },
    {
        "contact": {
            "name": "Uri Carmel",
            "email": "uri.carmel@soliduslabs.com",
            "title": "Business Professional",
            "company": "Solidus Labs",
            "linkedin_url": "https://www.linkedin.com/in/uri-carmel-73a39a"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.soliduslabs.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a crypto intelligence company interested in scaling operations. Values specialized go-to-market strategies and lead generation that work for companies in regulated industries."
    },
    {
        "contact": {
            "name": "Lance Berks",
            "email": "lance.berks@kasisto.com",
            "title": "Business Professional",
            "company": "Kasisto, Inc.",
            "linkedin_url": "https://www.linkedin.com/in/lance-berks-5161751"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.kasisto.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at an AI company focused on operational efficiency. Values targeted lead generation and multi-channel outreach strategies that can support growing tech companies."
    },
    {
        "contact": {
            "name": "Matt Johnson",
            "email": "johnson@energyhub.com",
            "title": "Business Professional",
            "company": "EnergyHub",
            "linkedin_url": "https://www.linkedin.com/in/matt-johnson-6ab46a"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "http://www.energyhub.com",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at an energy management company interested in operational excellence. Values industry-specific go-to-market strategies and lead generation that work in specialized industries."
    },
    {
        "contact": {
            "name": "Nathan Snyder",
            "email": "nathan.snyder@oosto.com",
            "title": "Business Professional",
            "company": "Oosto",
            "linkedin_url": "https://www.linkedin.com/in/nathanrsnyder"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://oosto.com/",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a vision AI company focused on scaling efficiently. Values targeted lead generation and go-to-market strategies tailored for tech companies with specialized offerings."
    },
    {
        "contact": {
            "name": "Kevin Stinger",
            "email": "kevin@zafran.io",
            "title": "Business Professional",
            "company": "Zafran Security",
            "linkedin_url": "https://www.linkedin.com/in/kevin-stinger-991bb312"
        },
        "company_url": "https://gtmwizards.com",  # This is the SELLER company (gtmWizards)
        "prospect_url": "https://www.zafran.io/",  # This is the PROSPECT company
        "target_geography": "New York",
        "expected_focus": "Professional at a security company focused on operational excellence. Values specialized go-to-market strategies and lead generation that work for security companies in competitive markets."
    }
]

# Functions to manage datasets
def get_or_create_dataset(client, name, description):
    """Get an existing dataset or create a new one if it doesn't exist."""
    try:
        dataset = client.create_dataset(
            dataset_name=name,
            description=description
        )
        logger.info(f"Created new dataset: {name}")
        return dataset
    except langsmith.utils.LangSmithConflictError:
        # If it already exists, retrieve it
        datasets = client.list_datasets(dataset_name=name)
        for dataset in datasets:
            if dataset.name == name:
                logger.info(f"Using existing dataset: {name}")
                return dataset
        # If we get here, something went wrong
        raise ValueError(f"Dataset with name {name} exists but couldn't be retrieved")

def add_examples_if_empty(client, dataset_id, inputs_list, outputs_list):
    """Add examples to a dataset only if it's empty."""
    # Check if dataset already has examples
    examples = list(client.list_examples(dataset_id=dataset_id))
    if not examples:
        logger.info(f"Adding examples to dataset {dataset_id}")
        client.create_examples(
            inputs=inputs_list,
            outputs=outputs_list,
            dataset_id=dataset_id
        )
        return True
    else:
        logger.info(f"Dataset {dataset_id} already has {len(examples)} examples, clearing and recreating")
        # Delete existing examples
        for example in examples:
            client.delete_example(example.id)
        
        # Create new examples
        client.create_examples(
            inputs=inputs_list,
            outputs=outputs_list,
            dataset_id=dataset_id
        )
        return True

# Get or create personality analysis dataset
personality_dataset_name = "Personality Analysis Evaluation - gtmWizards Prospect"
personality_dataset = get_or_create_dataset(
    client=client,
    name=personality_dataset_name,
    description="Evaluates personality analysis for sales outreach effectiveness using gtmWizards as the seller and a prospect as the target"
)

# Add examples to personality analysis dataset - force recreation
add_examples_if_empty(
    client=client,
    dataset_id=personality_dataset.id,
    inputs_list=[{
        "contact": example["contact"],
        "company_url": example["company_url"],
        "prospect_url": example["prospect_url"],
        "target_geography": example["target_geography"]
    } for example in personality_analysis_examples],
    outputs_list=[{
        "expected_focus": example["expected_focus"]
    } for example in personality_analysis_examples]
)

# Define target function for personality analysis evaluation
async def personality_analysis_target(inputs: dict) -> dict:
    """Target function for personality analysis."""
    try:
        contact_data = inputs["contact"]
        company_url = inputs["company_url"]  # gtmWizards URL (seller)
        prospect_url = inputs["prospect_url"]  # Prospect company URL
        target_geography = inputs.get("target_geography")
        
        logger.info(f"Starting company context analysis for seller: {company_url}")
        # First, get the company context using the company context workflow for the SELLER (gtmWizards)
        company_context = await analyze_company_context(
            company_url=company_url,
            model_name="anthropic/claude-3.5-haiku-20241022:beta",
            target_geography=target_geography
        )
        
        logger.info(f"Company context analysis complete for seller: {company_url}")
        logger.info(f"Starting personality analysis for prospect: {contact_data['name']} at {contact_data['company']}")
        
        # Create contact info object for the PROSPECT
        contact = ContactInfo(
            name=contact_data["name"],
            email=contact_data.get("email"),
            linkedin_url=contact_data.get("linkedin_url"),
            company=contact_data.get("company"),
            title=contact_data.get("title")
        )
        
        # Initialize personality analyzer
        analyzer = PersonalityAnalyzer(
            openrouter_api_key=openrouter_api_key,
            tavily_api_key=tavily_api_key,
            model_name="anthropic/claude-3.5-haiku-20241022:beta"
        )
        
        # Enable tracing
        analyzer.enable_tracing()
        
        # Analyze personality with the generated company context
        # The company_context is from gtmWizards (seller)
        # The contact is from the prospect company
        results = await analyzer.analyze_personalities_with_contacts(
            contacts=[contact],
            company_context=company_context
        )
        
        # Get the result for this contact
        result = next(iter(results.values()))
        
        logger.info(f"Personality analysis complete for {contact_data['name']}")
        
        return {
            "personality_analysis": result.personality_analysis,
            "conversation_style": result.conversation_style,
            "professional_interests": result.professional_interests,
            "company_context": company_context  # Include the company context in the output
        }
    except Exception as e:
        logger.error(f"Error in personality analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# Direct evaluation function without using wrappers
async def evaluate_with_llm(prompt: str, schema_class) -> BaseModel:
    """Evaluate using our ChatOpenRouter directly"""
    system_message = "You are an experienced sales coach evaluating personality analyses for sales effectiveness. Be critical, balanced, and constructive."
    
    # Create messages in LangChain format
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    # Implement retry mechanism for overloaded API
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use the invoke method instead of calling directly
            response = await llm.ainvoke(messages)
            
            # Extract content from the response
            content = response.content
            
            # Try to extract structured data using a helper function
            try:
                parsed_data = extract_structured_data(content, schema_class)
                return parsed_data
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                # Fallback with default values
                return create_default_evaluation(schema_class)
                
        except Exception as e:
            error_str = str(e)
            if "Overloaded" in error_str and attempt < max_retries - 1:
                retry_delay_with_jitter = retry_delay * (1 + attempt) + random.uniform(0, 1)
                logger.warning(f"Model overloaded. Retrying evaluation in {retry_delay_with_jitter:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                import time
                time.sleep(retry_delay_with_jitter)
            else:
                # If we've exhausted retries or it's not an overload error, log it
                logger.error(f"Error invoking LLM: {error_str}")
                # Log the full error for debugging
                import traceback
                logger.error(traceback.format_exc())
                # Return default evaluation - don't re-raise the error
                return create_default_evaluation(schema_class)
    
    # If we've exhausted all retries
    logger.error("All evaluation retries failed")
    return create_default_evaluation(schema_class)

def extract_structured_data(content: str, schema_class) -> BaseModel:
    """Extract structured data from LLM response"""
    # Simple parsing approach - extract key-value pairs
    data = {}
    
    # For numerical ratings (1-5)
    for field_name, field in schema_class.__annotations__.items():
        if field_name not in ["strengths", "improvement_areas", "suggested_enhancements"]:
            # Look for field_name: number or similar patterns
            patterns = [
                f"{field_name.replace('_', ' ')}\\s*[:-]\\s*([1-5])\\b",
                f"{field_name}\\s*[:-]\\s*([1-5])\\b",
            ]
            
            for pattern in patterns:
                import re
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    data[field_name] = int(match.group(1))
                    break
            
            # If not found, use default value
            if field_name not in data:
                data[field_name] = 3  # Default middle value
    
    # Extract text fields with improved patterns
    text_fields = {
        "strengths": ["strengths", "strong points", "positive aspects", "strength"],
        "improvement_areas": ["improvement areas", "areas for improvement", "weaknesses", "limitations", "improvement area", "areas that need improvement"],
        "suggested_enhancements": ["suggested enhancements", "recommendations", "suggestions", "enhancements", "suggested enhancement"]
    }
    
    for field_name, patterns in text_fields.items():
        field_content = None
        
        # Try different patterns for section headers
        for pattern in patterns:
            # More flexible regex patterns to capture various formatting styles
            section_patterns = [
                # Pattern for "Header: content" format with optional bullets
                f"{pattern}\\s*[:-]\\s*(.*?)(?=\\n\\n|\\n[A-Za-z]+\\s*:|$)",
                
                # Pattern for "Header\ncontent" format
                f"{pattern}\\s*\\n(.*?)(?=\\n\\n|\\n[A-Za-z]+\\s*:|$)",
                
                # Pattern for sections marked with headers and dashes
                f"{pattern}.*?\\n-+\\s*\\n(.*?)(?=\\n\\n|\\n[A-Za-z]+\\s*:|\\n[A-Za-z]+.*?\\n-+|$)",
                
                # Pattern for bulleted lists without explicit header
                f"{pattern}\\s*:?\\s*\\n((?:\\s*-[^\\n]*\\n)+)",
            ]
            
            import re
            for section_pattern in section_patterns:
                match = re.search(section_pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    field_content = match.group(1).strip()
                    # If it's just a placeholder like "None", ignore it
                    if field_content.lower() in ["none", "n/a", "not applicable"]:
                        field_content = None
                    break
            
            if field_content:
                break
        
        # Last resort: try to find any section of text after the heading
        if not field_content:
            for pattern in patterns:
                import re
                # Look for any content after the heading until a double newline
                match = re.search(f"(?:{pattern}).*?\\n(.*?)(?=\\n\\n|$)", content, re.IGNORECASE | re.DOTALL)
                if match:
                    field_content = match.group(1).strip()
                    if field_content and not field_content.lower() in ["none", "n/a", "not applicable"]:
                        break
        
        # If we found content, add it to the data
        if field_content:
            data[field_name] = field_content
        else:
            # Default fallback message
            data[field_name] = f"No specific {field_name.replace('_', ' ')} provided."
    
    # Create and return instance
    return schema_class(**data)

def create_default_evaluation(schema_class) -> BaseModel:
    """Create a default evaluation when parsing fails"""
    data = {}
    
    # Set default values
    for field_name, field in schema_class.__annotations__.items():
        if field_name not in ["strengths", "improvement_areas", "suggested_enhancements"]:
            data[field_name] = 3  # Middle value
    
    data["strengths"] = "Unable to parse specific strengths from evaluation."
    data["improvement_areas"] = "Unable to parse specific improvement areas from evaluation."
    data["suggested_enhancements"] = "Unable to parse specific enhancement suggestions from evaluation."
    
    return schema_class(**data)

# Create evaluator function for personality analysis
async def personality_analysis_evaluator(outputs: dict, reference_outputs: dict) -> PersonalityAnalysisEval:
    """Evaluates the quality and effectiveness of personality analysis for sales outreach."""
    personality_analysis = outputs.get("personality_analysis", "")
    conversation_style = outputs.get("conversation_style", "")
    professional_interests = outputs.get("professional_interests", [])
    company_context = outputs.get("company_context", {})
    expected_focus = reference_outputs.get("expected_focus", "")
    
    # Extract key elements from company context for the evaluator
    company_name = company_context.get("name", "Unknown Company")
    company_description = company_context.get("description", "No description available")
    company_pain_points = company_context.get("pain_points", [])
    if isinstance(company_pain_points, list):
        company_pain_points = ", ".join(company_pain_points)
    
    # For debugging purposes, log the company context
    logger.info(f"Company context for evaluation: {json.dumps(company_context, indent=2)}")
    if company_name == "Unknown Company" and company_context:
        logger.warning(f"Company name not found in context. Available keys: {list(company_context.keys())}")
    
    # Construct the evaluation prompt
    prompt = f"""
    You are evaluating the quality and practical effectiveness of a personality analysis for sales outreach.
    
    SCENARIO:
    A salesperson from {company_name} (the SELLER) is trying to sell to a prospect (the BUYER).
    The personality analysis is meant to help the {company_name} salesperson understand the prospect better
    and tailor their outreach accordingly.
    
    SELLER COMPANY CONTEXT:
    Company: {company_name}
    Description: {company_description}
    Pain Points Addressed: {company_pain_points}
    
    PROSPECT EXPECTED FOCUS:
    {expected_focus}
    
    GENERATED PERSONALITY ANALYSIS OF THE PROSPECT:
    {personality_analysis}
    
    CONVERSATION STYLE RECOMMENDATIONS FOR TALKING TO THE PROSPECT:
    {conversation_style}
    
    PROSPECT'S PROFESSIONAL INTERESTS:
    {', '.join(professional_interests) if professional_interests else 'None provided'}
    
    Evaluate this analysis based on how useful it would be for a {company_name} sales professional trying to connect with this prospect.
    Focus on practical application rather than theoretical accuracy.
    
    Your evaluation MUST follow this exact format and include all these sections:
    
    Practical Relevance: [score 1-5]
    Conversation Guidance: [score 1-5]
    Pain Point Identification: [score 1-5]
    Personality Insight Depth: [score 1-5]
    Company Context Integration: [score 1-5]
    Actionability: [score 1-5]
    Overall Value: [score 1-5]
    
    Strengths:
    - [First strength]
    - [Second strength]
    - [Additional strengths]
    
    Improvement Areas:
    - [First improvement area]
    - [Second improvement area]
    - [Additional improvement areas]
    
    Suggested Enhancements:
    - [First enhancement suggestion]
    - [Second enhancement suggestion]
    - [Additional enhancement suggestions]
    
    When evaluating, please use these criteria:
    
    1. Practical Relevance: How relevant and applicable are the insights for real sales conversations?
    2. Conversation Guidance: How specific and helpful are the conversation tactics, openers, and approaches?
    3. Pain Point Identification: How accurately and deeply are the prospect's professional pain points and challenges identified?
    4. Personality Insight Depth: How nuanced and specific are the personality insights beyond generic observations?
    5. Company Context Integration: How well does the analysis integrate {company_name}'s value propositions to address the prospect's needs?
    6. Actionability: How immediately actionable are the recommendations for sales outreach?
    7. Overall Value: What is the overall value of this analysis for improving sales conversations?
    
    Be critical but fair - this evaluation is meant to help improve the system, not to artificially inflate scores.
    """
    
    # Get evaluation directly from LLM
    return await evaluate_with_llm(prompt, PersonalityAnalysisEval)

# Helper function to run async target and evaluator functions
async def run_async_evaluation(target_fn, evaluator_fn, inputs, reference_outputs=None):
    """Run async target and evaluator functions and return the results."""
    # Run target function
    outputs = await target_fn(inputs)
    
    # Run evaluator function
    evaluation = await evaluator_fn(outputs, reference_outputs or {})
    
    return {
        "outputs": outputs,
        "evaluation": evaluation
    }

# Run evaluation for personality analysis
async def evaluate_personality_analysis():
    """Run evaluation for personality analysis."""
    logger.info("Starting personality analysis evaluation")
    
    results = []
    examples = list(client.list_examples(dataset_id=personality_dataset.id))
    
    # Limit the number of examples if MAX_EXAMPLES is set
    if MAX_EXAMPLES is not None:
        logger.info(f"Limiting evaluation to first {MAX_EXAMPLES} examples")
        examples = examples[:MAX_EXAMPLES]
    
    for example in examples:
        inputs = example.inputs
        reference_outputs = example.outputs
        
        # Verify inputs have the expected structure
        if "contact" not in inputs or "company_url" not in inputs:
            logger.error(f"Example inputs missing required fields: {inputs}")
            continue
            
        logger.info(f"Evaluating personality analysis for {inputs['contact']['name']} at {inputs['company_url']}")
        
        result = await run_async_evaluation(
            personality_analysis_target,
            personality_analysis_evaluator,
            inputs,
            reference_outputs
        )
        
        # Log and save results
        try:
            client.create_run(
                name="personality-analysis-evaluation",
                inputs=inputs,
                outputs=result["outputs"],
                tags=["evaluation", "personality-analysis"],
                run_type="llm",  # Using "llm" as the run_type
                metadata={
                    "evaluation": result["evaluation"].model_dump()
                }
            )
            logger.info(f"Successfully created run in LangSmith for {inputs['contact']['name']}")
        except Exception as e:
            logger.error(f"Error creating run in LangSmith: {str(e)}")
        
        results.append(result)
    
    # Log results in a readable format
    logger.info(f"Personality analysis evaluation complete. Results:")
    for i, result in enumerate(results):
        eval_data = result["evaluation"].model_dump()
        # Use the name and company from the inputs rather than the examples list
        # which might not match if we're limiting the number of examples
        example_inputs = examples[i].inputs
        contact_name = example_inputs["contact"]["name"]
        prospect_company = example_inputs["contact"]["company"]
        seller_company = "gtmWizards"  # This is the seller company
        logger.info(f"\nResults for prospect {contact_name} at {prospect_company} (seller: {seller_company}):")
        
        # Print scores
        score_fields = [f for f in eval_data.keys() if f not in ["strengths", "improvement_areas", "suggested_enhancements"]]
        for field in score_fields:
            logger.info(f"  {field.replace('_', ' ').title()}: {eval_data[field]}/5")
        
        # Print text feedback
        logger.info(f"  Strengths: {eval_data['strengths']}")
        logger.info(f"  Improvement Areas: {eval_data['improvement_areas']}")
        logger.info(f"  Suggested Enhancements: {eval_data['suggested_enhancements']}")
    
    return results

# Main function to run evaluation
async def run_evaluation():
    """Run personality analysis evaluation and return results."""
    personality_results = await evaluate_personality_analysis()
    return {"personality_analysis_results": personality_results}

# Run evaluation if this file is executed directly
if __name__ == "__main__":
    logger.info("Starting personality analysis evaluation")
    asyncio.run(run_evaluation())
    logger.info("Evaluation complete") 
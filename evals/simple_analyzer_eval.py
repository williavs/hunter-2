"""
Evaluation script for the simplified company analyzer.

This script evaluates the performance of the simple_company_analyzer module
by running it on test cases and evaluating the results using LangSmith.

Usage:
    python -m evals.simple_analyzer_eval                    # Use default example (GTM Wizards)
    python -m evals.simple_analyzer_eval --url https://example.com  # Analyze specific URL
    python -m evals.simple_analyzer_eval --url https://example.com --geo "North America"  # With geography
    python -m evals.simple_analyzer_eval --list             # Use all predefined examples
"""

from langsmith import Client
from pydantic import BaseModel, Field
import asyncio
import os
import sys
import json
import logging
import time
import argparse
import langsmith.utils
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to sys.path so we can import from ai_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our simple company analyzer
from ai_utils.simple_company_analyzer import analyze_company
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', f'simple_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

# Initialize LangSmith client (if available)
try:
    client = Client()
    langsmith_available = True
except:
    logger.warning("LangSmith client not initialized. Will run without LangSmith tracking.")
    langsmith_available = False

# Get API keys
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

# Models to evaluate
MODELS = [
    {
        "name": "Claude 3.5 Haiku",
        "model_id": "anthropic/claude-3.5-haiku-20241022:beta",
        "description": "Fast, efficient model with good quality"
    },
    {
        "name": "Claude 3.5 Sonnet",
        "model_id": "anthropic/claude-3.5-sonnet-20241022:beta",
        "description": "Balanced model for most use cases"
    },
    {
        "name": "Claude 3.7 Sonnet",
        "model_id": "anthropic/claude-3.7-sonnet",
        "description": "Latest Anthropic model with advanced capabilities"
    }
]

# Initialize the OpenRouter clients for evaluation
llm_clients = {}
for model in MODELS:
    llm_clients[model["model_id"]] = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model=model["model_id"],
        temperature=0.1
    )

# Predefined test cases
DEFAULT_EXAMPLES = [
    {
        "company_url": "https://gtmwizards.com",
        "target_geography": "Europe and North America",
        "expected_focus": "B2B lead generation agency that provides go-to-market strategies using a combination of human expertise and AI-powered solutions."
    },
    {
        "company_url": "https://www.salesforce.com",
        "target_geography": "Global",
        "expected_focus": "CRM software company offering cloud-based applications for sales, service, marketing, and more."
    },
    {
        "company_url": "https://www.hubspot.com",
        "target_geography": "Global",
        "expected_focus": "Inbound marketing, sales, and customer service software that helps companies attract visitors, convert leads, and close customers."
    }
]

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate company analysis for any URL.')
    parser.add_argument('--url', type=str, help='Company URL to analyze')
    parser.add_argument('--geo', '--geography', type=str, dest='geography', 
                        help='Target geography for analysis (e.g., "North America", "Europe", "Global")')
    parser.add_argument('--list', action='store_true', 
                        help='Run evaluation on all predefined examples')
    parser.add_argument('--model', type=str, choices=[m["model_id"] for m in MODELS],
                        help='Specific model to use for evaluation')
    return parser.parse_args()

# Define the evaluation schema
class CompanyAnalyzerEval(BaseModel):
    """Evaluation schema for simple company analyzer."""
    # Content quality metrics
    relevance: int = Field(description="Score 1-5 on how relevant the generated context is to the company's actual business model and service offerings", ge=1, le=5)
    pain_points_quality: int = Field(description="Score 1-5 on how well the analysis identifies specific pain points the company claims to solve for their clients", ge=1, le=5)
    differentiation_quality: int = Field(description="Score 1-5 on how well the analysis identifies true differentiators and unique value propositions", ge=1, le=5)
    target_market_accuracy: int = Field(description="Score 1-5 on accuracy of identified target markets, specific industries, and geographical focus", ge=1, le=5)
    sales_insights: int = Field(description="Score 1-5 on useful, actionable sales insights that could help in outreach to this company", ge=1, le=5)
    
    # Output quality metrics
    summarization_clarity: int = Field(description="Score 1-5 on the clarity and conciseness of the summaries provided", ge=1, le=5)
    practical_applicability: int = Field(description="Score 1-5 on how practically useful the analysis would be for sales outreach", ge=1, le=5)
    human_likeness: int = Field(description="Score 1-5 on how human-like and natural the analysis reads (not overly formulaic)", ge=1, le=5)
    
    # Overall metrics
    overall_quality: int = Field(description="Overall quality score 1-5", ge=1, le=5)
    feedback: str = Field(description="Detailed feedback on strengths and critical areas for improvement with specific examples")

# LangSmith dataset functions (if available)
def get_or_create_dataset(client, name, description):
    """Get an existing dataset or create a new one if it doesn't exist."""
    if not langsmith_available:
        return None
        
    try:
        # Try to create a new dataset
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
    if not langsmith_available or not dataset_id:
        return
        
    # Check if dataset already has examples
    examples = list(client.list_examples(dataset_id=dataset_id))
    if not examples:
        logger.info(f"Adding examples to dataset {dataset_id}")
        client.create_examples(
            inputs=inputs_list,
            outputs=outputs_list,
            dataset_id=dataset_id
        )
    else:
        logger.info(f"Dataset {dataset_id} already has {len(examples)} examples, skipping creation")

# Extract domain name for display
def get_domain_name(url):
    """Extract domain name from URL for display purposes."""
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

# Company analysis function
async def analyze_company_target(inputs: dict, model_id: str) -> dict:
    """Target function for simple company analysis."""
    try:
        company_url = inputs["company_url"]
        target_geography = inputs.get("target_geography")
        
        logger.info(f"Starting analysis for {company_url} with target geography {target_geography} using model {model_id}")
        start_time = time.time()
        
        # Use our simple company analyzer
        result = analyze_company(
            company_url=company_url,
            target_geography=target_geography,
            openrouter_api_key=openrouter_api_key,
            tavily_api_key=tavily_api_key
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Completed analysis in {execution_time:.2f} seconds. Result: {result['name']}")
        
        if not result:
            logger.error("Analysis returned empty result!")
            result = {"error": "Analysis returned empty result"}
        
        return {
            "company_context": result,
            "execution_time": execution_time,
            "had_error": "errors" in result and result["errors"],
            "model_id": model_id
        }
    except Exception as e:
        logger.error(f"Error in company analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "company_context": {"error": str(e)}, "had_error": True, "model_id": model_id}

# Evaluation function
async def evaluate_with_llm(prompt: str, schema_class, model_id: str) -> Dict:
    """Evaluate using our OpenRouter client directly"""
    system_message = "You are an expert sales coach evaluating analysis quality. Be critical but fair."
    
    # Call the LLM directly
    try:
        # Create messages format
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        # Get the LLM client for this model
        llm = llm_clients[model_id]
        
        # Use the invoke method
        response = llm.invoke(messages)
        content = response.content
        
        # Parse the response to extract structured data
        try:
            # Try to extract structured data
            parsed_data = extract_structured_data(content, schema_class)
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Fallback with default values
            return create_default_evaluation(schema_class)
    except Exception as e:
        logger.error(f"Error invoking LLM: {str(e)}")
        # Log the full error for debugging
        import traceback
        logger.error(traceback.format_exc())
        return create_default_evaluation(schema_class)

def extract_structured_data(content: str, schema_class) -> Dict:
    """Extract structured data from LLM response"""
    # Simple parsing approach - extract key-value pairs
    data = {}
    
    # For numerical ratings (1-5)
    for field_name, field in schema_class.__annotations__.items():
        if field_name == "feedback":
            continue  # Handle feedback separately
            
        # Look for patterns like "relevance: 4" or "Relevance: 4/5"
        patterns = [
            f"{field_name}: (\d+)",
            f"{field_name.capitalize()}: (\d+)",
            f"{field_name.capitalize()}: (\d+)/5",
            f"{field_name.replace('_', ' ').capitalize()}: (\d+)",
            f"{field_name.replace('_', ' ').capitalize()}: (\d+)/5"
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, content)
            if match:
                data[field_name] = int(match.group(1))
                break
                
        # If not found, try another approach - look for sections
        if field_name not in data:
            section_pattern = f"{field_name.replace('_', ' ').capitalize()}: ([^\\n]+)"
            section_match = re.search(section_pattern, content, re.IGNORECASE)
            if section_match:
                text = section_match.group(1).strip()
                # Try to extract a number
                num_match = re.search(r"(\d+)", text)
                if num_match:
                    data[field_name] = int(num_match.group(1))
    
    # Extract feedback
    feedback_pattern = r"(Feedback|FEEDBACK|Detailed feedback|DETAILED FEEDBACK):\s*([\s\S]+?)(?=\n\s*\w+:|$)"
    feedback_match = re.search(feedback_pattern, content)
    if feedback_match:
        data["feedback"] = feedback_match.group(2).strip()
    else:
        # If no formal feedback section, use the last paragraph
        paragraphs = content.split("\n\n")
        if paragraphs:
            data["feedback"] = paragraphs[-1].strip()
    
    # Fill in missing values with defaults
    for field_name in schema_class.__annotations__:
        if field_name not in data:
            if field_name == "feedback":
                data[field_name] = "No feedback provided."
            else:
                data[field_name] = 3  # Default middle value
    
    return data

def create_default_evaluation(schema_class) -> Dict:
    """Create a default evaluation with middle values."""
    data = {}
    for field_name, field in schema_class.__annotations__.items():
        if field_name == "feedback":
            data[field_name] = "Unable to generate detailed feedback due to an error in the evaluation process."
        else:
            data[field_name] = 3  # Default middle value
    return data

async def company_context_evaluator(outputs: dict, reference_outputs: dict = None, eval_model_id: str = "anthropic/claude-3.5-haiku-20241022:beta") -> Dict:
    """Evaluate the company context analysis."""
    try:
        company_context = outputs.get("company_context", {})
        expected_focus = "" if not reference_outputs else reference_outputs.get("expected_focus", "")
        
        # Construct the prompt for evaluation
        prompt = f"""
Evaluate the quality of this company analysis:

COMPANY URL: {company_context.get('url', 'Unknown URL')}
COMPANY NAME: {company_context.get('name', 'Unknown Name')}
TARGET GEOGRAPHY: {company_context.get('target_geography', 'None specified')}
CONFIDENCE: {company_context.get('confidence', 'Unknown')}
SEARCH QUALITY: {company_context.get('search_quality', 'Unknown')}

DESCRIPTION:
{company_context.get('description', 'No description provided.')}

FULL ANALYSIS:
{company_context.get('analysis', 'No analysis provided.')}

EXPECTED FOCUS (GROUND TRUTH):
{expected_focus}

Please provide a detailed evaluation with specific ratings from 1-5 (5 being the best) for:
- relevance: How relevant is the analysis to the company's business?
- pain_points_quality: How well are the customer pain points identified?
- differentiation_quality: How well are the unique differentiators identified?
- target_market_accuracy: How accurately are target markets identified?
- sales_insights: How useful are the insights for sales outreach?
- summarization_clarity: How clear and concise is the analysis?
- practical_applicability: How practical would this be for sales use?
- human_likeness: How natural and human-like is the analysis?
- overall_quality: Overall quality score

FEEDBACK:
Provide detailed, constructive feedback on strengths and areas for improvement. Use specific examples.
"""
        
        # Evaluate with LLM
        evaluation = await evaluate_with_llm(prompt, CompanyAnalyzerEval, eval_model_id)
        return evaluation
        
    except Exception as e:
        logger.error(f"Error in company context evaluator: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_default_evaluation(CompanyAnalyzerEval)

async def run_evaluation_for_input(inputs, reference_outputs=None, model_id=None, eval_model_id=None):
    """Run evaluation for a single input."""
    model_id = model_id or MODELS[0]["model_id"]
    eval_model_id = eval_model_id or "anthropic/claude-3.5-haiku-20241022:beta"
    
    logger.info(f"Evaluating company: {inputs['company_url']} with model: {model_id}")
    
    # Run the analysis
    start_time = time.time()
    outputs = await analyze_company_target(inputs, model_id)
    total_time = time.time() - start_time
    
    # Create a record for comparison
    record = {
        "company_url": inputs["company_url"],
        "company_domain": get_domain_name(inputs["company_url"]),
        "target_geography": inputs.get("target_geography", "None"),
        "execution_time": outputs.get("execution_time", 0),
        "had_error": outputs.get("had_error", False),
        "model_id": model_id,
        "model_name": next((m["name"] for m in MODELS if m["model_id"] == model_id), "Unknown")
    }
    
    # Add the outputs for reference
    record["outputs"] = outputs
    
    # Only evaluate if there was no error
    if not outputs.get("had_error", False):
        # Evaluate the outputs
        evaluation = await company_context_evaluator(outputs, reference_outputs, eval_model_id)
        record["evaluation"] = evaluation
        
        # Add evaluation metrics to the record
        for metric, value in evaluation.items():
            if metric != "feedback":
                record[f"eval_{metric}"] = value
    
    return record

async def evaluate_companies(company_examples, specific_model_id=None):
    """Evaluate the list of companies with all models or a specific model."""
    logger.info(f"Starting company analysis evaluation for {len(company_examples)} companies")
    
    # Setup LangSmith if available
    dataset = None
    if langsmith_available:
        dataset_name = "Simple Company Analyzer Evaluation"
        dataset = get_or_create_dataset(
            client=client,
            name=dataset_name,
            description="Evaluation of the simple company analyzer with any company"
        )
        
        # Add examples to dataset
        add_examples_if_empty(
            client=client,
            dataset_id=dataset.id if dataset else None,
            inputs_list=[{"company_url": example["company_url"], "target_geography": example.get("target_geography")} 
                     for example in company_examples],
            outputs_list=[{"expected_focus": example.get("expected_focus", "")} 
                     for example in company_examples]
        )
    
    all_results = []
    
    # Use Haiku for evaluation as it's faster and more cost-effective
    eval_model_id = "anthropic/claude-3.5-haiku-20241022:beta"
    
    # Determine which models to evaluate with
    models_to_use = [next(m for m in MODELS if m["model_id"] == specific_model_id)] if specific_model_id else MODELS
    
    # Run for each company example with selected models
    for example in company_examples:
        inputs = {
            "company_url": example["company_url"],
            "target_geography": example.get("target_geography")
        }
        reference_outputs = {
            "expected_focus": example.get("expected_focus", "")
        }
        
        # Run evaluation with each model
        for model in models_to_use:
            model_id = model["model_id"]
            
            # Run evaluation
            result = await run_evaluation_for_input(inputs, reference_outputs, model_id, eval_model_id)
            all_results.append(result)
            
            # Save to LangSmith if available
            if langsmith_available and dataset:
                try:
                    run_id = client.create_run(
                        name=f"simple-company-analyzer-{model['name']}",
                        inputs=inputs,
                        outputs=result["outputs"],
                        tags=["evaluation", "simple-company-analyzer", model["name"], 
                             f"company-{get_domain_name(example['company_url'])}"],
                        run_type="chain",
                        metadata={
                            "evaluation": result.get("evaluation", {}),
                            "execution_time": result.get("execution_time", 0),
                            "model_id": model_id,
                            "model_name": model["name"]
                        }
                    )
                    logger.info(f"Successfully created run in LangSmith with ID: {run_id}")
                except Exception as e:
                    logger.error(f"Error creating run in LangSmith: {str(e)}")
    
    # Create a report
    df = pd.DataFrame(all_results)
    
    # Calculate averages for numerical metrics by model
    model_avg_metrics = {}
    for model in models_to_use:
        model_id = model["model_id"]
        model_results = [r for r in all_results if r.get("model_id") == model_id]
        if not model_results:
            continue
            
        model_df = pd.DataFrame(model_results)
        model_metrics = {}
        
        for metric in ['eval_relevance', 'eval_pain_points_quality', 'eval_differentiation_quality', 
                    'eval_target_market_accuracy', 'eval_sales_insights', 'eval_summarization_clarity',
                    'eval_practical_applicability', 'eval_human_likeness', 'eval_overall_quality']:
            if metric in model_df.columns:
                model_metrics[f"avg_{metric}"] = model_df[metric].mean()
        
        model_metrics["avg_execution_time"] = model_df["execution_time"].mean() if "execution_time" in model_df.columns else 0
        model_avg_metrics[model["name"]] = model_metrics
    
    # Log the summary metrics
    for model_name, metrics in model_avg_metrics.items():
        logger.info(f"Evaluation summary for {model_name}: {json.dumps(metrics, indent=2)}")
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all companies in one report
    report_path = os.path.join('logs', f'company_eval_report_{timestamp}.csv')
    df.to_csv(report_path, index=False)
    logger.info(f"Saved evaluation report to {report_path}")
    
    # Also save detailed results as JSON
    detailed_results = []
    for result in all_results:
        if "outputs" in result and "company_context" in result["outputs"]:
            company_data = result["outputs"]["company_context"]
            evaluation = result.get("evaluation", {})
            
            detailed_result = {
                "company_url": result["company_url"],
                "company_domain": result.get("company_domain", get_domain_name(result["company_url"])),
                "target_geography": result["target_geography"],
                "name": company_data.get("name", "Unknown"),
                "description": company_data.get("description", "None"),
                "confidence": company_data.get("confidence", "Unknown"),
                "search_quality": company_data.get("search_quality", 0),
                "execution_time": result.get("execution_time", 0),
                "had_error": result.get("had_error", False),
                "model_id": result.get("model_id"),
                "model_name": result.get("model_name")
            }
            
            # Add evaluation metrics
            for metric, value in evaluation.items():
                if metric != "feedback":
                    detailed_result[f"eval_{metric}"] = value
                else:
                    detailed_result["feedback"] = value
                    
            detailed_results.append(detailed_result)
    
    # Save detailed results
    detailed_path = os.path.join('logs', f'company_eval_detailed_{timestamp}.json')
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    logger.info(f"Saved detailed results to {detailed_path}")
    
    # Create a comparison table for all models
    comparison_data = []
    for model_name, metrics in model_avg_metrics.items():
        row = {"model": model_name}
        row.update({k: v for k, v in metrics.items()})
        comparison_data.append(row)
        
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join('logs', f'model_comparison_{timestamp}.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved model comparison to {comparison_path}")
    
    return {
        "results": all_results,
        "model_metrics": model_avg_metrics,
        "report_path": report_path,
        "detailed_path": detailed_path,
        "comparison_path": comparison_path
    }

# Main function
async def main():
    """Run the evaluation with command line arguments."""
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    
    # Verify API keys are properly set
    logger.info(f"OpenRouter API key available: {bool(openrouter_api_key)}")
    logger.info(f"Tavily API key available: {bool(tavily_api_key)}")
    
    try:
        # Determine which companies to evaluate
        if args.url:
            # User specified a URL
            company_examples = [{
                "company_url": args.url,
                "target_geography": args.geography if args.geography else "Global",
                "expected_focus": ""  # No expected focus for user-provided URLs
            }]
            print(f"\n=== EVALUATING CUSTOM COMPANY: {args.url} ===")
            
        elif args.list:
            # Use all predefined examples
            company_examples = DEFAULT_EXAMPLES
            print(f"\n=== EVALUATING ALL PREDEFINED EXAMPLES ({len(DEFAULT_EXAMPLES)} companies) ===")
            
        else:
            # Default to first example
            company_examples = [DEFAULT_EXAMPLES[0]]
            print(f"\n=== EVALUATING DEFAULT EXAMPLE: {DEFAULT_EXAMPLES[0]['company_url']} ===")
        
        # Print models to be evaluated
        if args.model:
            model_name = next((m["name"] for m in MODELS if m["model_id"] == args.model), args.model)
            print(f"Using specific model: {model_name}")
        else:
            print("\n=== MODELS TO BE EVALUATED ===")
            for model in MODELS:
                print(f"- {model['name']} ({model['model_id']}): {model['description']}")
        
        # Run evaluation with all models or specified model
        print("\n=== RUNNING EVALUATION ===")
        results = await evaluate_companies(company_examples, args.model)
        
        # Print summary for each model
        print("\n=== EVALUATION SUMMARY BY MODEL ===")
        for model_name, metrics in results["model_metrics"].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")
        
        # Print comparison table
        comparison_path = results.get("comparison_path")
        if comparison_path and os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            print("\n=== MODEL COMPARISON ===")
            print(comparison_df.to_string(index=False))
        
        # Print summary for each company
        print("\n=== COMPANY RESULTS ===")
        for result in results["results"]:
            if "eval_overall_quality" in result:
                print(f"{result['company_domain']} ({result['model_name']}): Overall Quality = {result['eval_overall_quality']}/5")
        
        print(f"\nDetailed results saved to: {results.get('detailed_path', 'unknown')}")
        print(f"CSV report saved to: {results.get('report_path', 'unknown')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(main()) 
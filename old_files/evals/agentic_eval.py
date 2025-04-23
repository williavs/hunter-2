from langsmith import Client
from pydantic import BaseModel, Field
import asyncio
import os
import sys
import json
import logging
import time
import langsmith.utils
import pandas as pd
from datetime import datetime

# Add the parent directory to sys.path so we can import from ai_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom OpenRouter connection and both analysis functions
from old_files.company_context_workflow import ChatOpenRouter, analyze_company_context
from old_files.new_company_context_agent import analyze_company_context_agentic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

# Initialize LangSmith client
client = Client()

# Use our existing OpenRouter connection
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

# Initialize the OpenRouter client with the haiku model for evaluation
llm = ChatOpenRouter(
    openai_api_key=openrouter_api_key,
    model="anthropic/claude-3.5-haiku-20241022:beta",
    temperature=0.1,
    streaming=False
)

# Test dataset with diverse examples
agentic_company_context_examples = [
    {
        "company_url": "https://gtmwizards.com",  # Removed www. which might cause issues
        "target_geography": "Europe and North America",
        "expected_focus": "B2B lead generation agency that provides go-to-market strategies using a combination of human expertise and AI-powered solutions. They offer pipeline building as a service, focusing on precise ICP development, database building, GTM strategy creation, and multi-channel outreach for B2B companies. Their target clients appear to be SMEs looking to scale revenue generation without building in-house sales teams."
    }
]

# Define the enhanced evaluation schema
class AgenticCompanyContextEval(BaseModel):
    """Evaluation schema for agentic company context analysis."""
    # Original metrics with more rigorous expectations
    relevance: int = Field(description="Score 1-5 on how relevant the generated context is to the company's actual business model and service offerings", ge=1, le=5)
    pain_points_quality: int = Field(description="Score 1-5 on how well the analysis identifies specific pain points the company claims to solve for their clients", ge=1, le=5)
    differentiation_quality: int = Field(description="Score 1-5 on how well the analysis identifies true differentiators and unique value propositions", ge=1, le=5)
    target_market_accuracy: int = Field(description="Score 1-5 on accuracy of identified target markets, specific industries, and geographical focus", ge=1, le=5)
    sales_insights: int = Field(description="Score 1-5 on useful, actionable sales insights that could help in outreach to this company", ge=1, le=5)
    
    # New metrics specific to the agentic approach with higher expectations
    search_quality: int = Field(description="Score 1-5 on the quality, relevance and strategic formulation of search queries used", ge=1, le=5)
    information_validation: int = Field(description="Score 1-5 on how thoroughly the agent validates and cross-references information for accuracy", ge=1, le=5)
    resilience: int = Field(description="Score 1-5 on how effectively the agent handles poor search results or missing information", ge=1, le=5)
    
    # New metrics for summary quality
    summarization_clarity: int = Field(description="Score 1-5 on the clarity and conciseness of the summaries provided", ge=1, le=5)
    practical_applicability: int = Field(description="Score 1-5 on how practically useful the analysis would be for sales outreach", ge=1, le=5)
    human_likeness: int = Field(description="Score 1-5 on how human-like and natural the analysis reads (not overly formulaic)", ge=1, le=5)
    
    # Overall metrics
    overall_quality: int = Field(description="Overall quality score 1-5", ge=1, le=5)
    feedback: str = Field(description="Detailed feedback on strengths and critical areas for improvement with specific examples")

# Create or get existing dataset in LangSmith
def get_or_create_dataset(client, name, description):
    """Get an existing dataset or create a new one if it doesn't exist."""
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

# Get or create dataset
agentic_company_dataset_name = "gtmWizards Focused Evaluation"
agentic_company_dataset = get_or_create_dataset(
    client=client,
    name=agentic_company_dataset_name,
    description="Rigorous evaluation of agentic company context analysis specifically for gtmWizards"
)

# Add examples to dataset - only if it doesn't already exist
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
    else:
        logger.info(f"Dataset {dataset_id} already has {len(examples)} examples, skipping creation")

# Add examples to agentic company context dataset
add_examples_if_empty(
    client=client,
    dataset_id=agentic_company_dataset.id,
    inputs_list=[{"company_url": example["company_url"], "target_geography": example["target_geography"]} 
             for example in agentic_company_context_examples],
    outputs_list=[{"expected_focus": example["expected_focus"]} 
             for example in agentic_company_context_examples]
)

# Add debug logging for the target functions
async def legacy_company_context_target(inputs: dict) -> dict:
    """Target function for legacy company context analysis."""
    try:
        company_url = inputs["company_url"]
        target_geography = inputs.get("target_geography")
        
        logger.info(f"Starting legacy analysis for {company_url} with target geography {target_geography}")
        start_time = time.time()
        
        # Use our existing company context analysis function
        result = await analyze_company_context(
            company_url=company_url,
            model_name="anthropic/claude-3.5-haiku-20241022:beta",
            target_geography=target_geography
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Completed legacy analysis in {execution_time:.2f} seconds. Result length: {len(json.dumps(result))}")
        
        if not result or result == {}:
            logger.error("Legacy analysis returned empty result!")
            result = {"error": "Analysis returned empty result"}
        
        return {
            "company_context": result,
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Error in legacy company context analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "company_context": {"error": str(e)}}

async def agentic_company_context_target(inputs: dict) -> dict:
    """Target function for agentic company context analysis."""
    try:
        company_url = inputs["company_url"]
        target_geography = inputs.get("target_geography")
        
        logger.info(f"Starting agentic analysis for {company_url} with target geography {target_geography}")
        start_time = time.time()
        
        # Use our new agentic company context analysis function
        result = await analyze_company_context_agentic(
            company_url=company_url,
            model_name="anthropic/claude-3.5-haiku-20241022:beta",
            target_geography=target_geography
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Completed agentic analysis in {execution_time:.2f} seconds. Result length: {len(json.dumps(result))}")
        
        if not result or result == {}:
            logger.error("Agentic analysis returned empty result!")
            result = {"error": "Analysis returned empty result"}
        
        had_error = "error" in result
        
        return {
            "company_context": result,
            "execution_time": execution_time,
            "had_error": had_error
        }
    except Exception as e:
        logger.error(f"Error in agentic company context analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "company_context": {"error": str(e)}, "had_error": True}

# Direct evaluation function without using wrappers
async def evaluate_with_llm(prompt: str, schema_class) -> BaseModel:
    """Evaluate using our ChatOpenRouter directly"""
    system_message = "You are an expert sales coach evaluating analysis quality. Be critical but fair."
    
    # Call the LLM directly using the proper invoke method
    try:
        # Create messages in LangChain format
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        # Use the invoke method instead of calling directly
        response = await llm.ainvoke(messages)
        
        # Extract content from the response
        content = response.content
        
        # Use schema_class to parse the content
        try:
            # Try to extract structured data using a helper function
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

def extract_structured_data(content: str, schema_class) -> BaseModel:
    """Extract structured data from LLM response with improved parsing for more fields"""
    # Simple parsing approach - extract key-value pairs
    data = {}
    
    # For numerical ratings (1-5)
    for field_name, field in schema_class.__annotations__.items():
        if field_name != "feedback":
            # Look for field_name: number or similar patterns
            patterns = [
                f"{field_name.replace('_', ' ')}\\s*[:-]\\s*([1-5])\\b",
                f"{field_name}\\s*[:-]\\s*([1-5])\\b",
                f"\\b{field_name.replace('_', '[- ]')}\\s*[:-]\\s*([1-5])\\b",
                f"\\b{field_name.replace('_', ' ').title()}\\s*[:-]\\s*([1-5])\\b",
                f"\\d+\\.\\s*.*?{field_name.replace('_', ' ')}.*?[:-]\\s*([1-5])\\b",
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
    
    # Extract feedback (assuming it's the last part after "feedback" or similar)
    import re
    feedback_patterns = [
        r"feedback\s*[:-]\s*(.*?)(?:\n\n|\Z)",
        r"feedback[:\n]+\s*(.*?)(?:\n\n|\Z)",
        r"feedback.*?\n\s*(.*?)(?:\n\n|\Z)",
        r"detailed.*?feedback[:\n]+\s*(.*?)(?:\n\n|\Z)",
    ]
    
    for pattern in feedback_patterns:
        feedback_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if feedback_match:
            data["feedback"] = feedback_match.group(1).strip()
            break
    
    if "feedback" not in data:
        data["feedback"] = "No specific feedback provided."
    
    # Create and return instance
    return schema_class(**data)

def create_default_evaluation(schema_class) -> BaseModel:
    """Create a default evaluation when parsing fails"""
    data = {}
    
    # Set default values
    for field_name, field in schema_class.__annotations__.items():
        if field_name != "feedback":
            data[field_name] = 3  # Middle value
    
    data["feedback"] = "Unable to parse specific feedback from evaluation."
    
    return schema_class(**data)

# Create evaluator function
async def agentic_company_context_evaluator(outputs: dict, reference_outputs: dict) -> AgenticCompanyContextEval:
    """Evaluates the quality of agentic company context analysis with higher standards."""
    company_context = outputs.get("company_context", {})
    expected_focus = reference_outputs.get("expected_focus", "")
    execution_time = outputs.get("execution_time", 0)
    had_error = outputs.get("had_error", False)
    
    # Construct the evaluation prompt with more human-like critical tone
    prompt = f"""
    You are an experienced sales and marketing director evaluating an AI assistant's analysis of a company for sales prospecting purposes. Be extremely critical and demanding - this analysis will be used for real business decisions.
    
    COMPANY BEING ANALYZED: gtmWizards
    
    EXPECTED FOCUS:
    {expected_focus}
    
    GENERATED COMPANY CONTEXT:
    {json.dumps(company_context, indent=2)}
    
    ADDITIONAL METADATA:
    Execution Time: {execution_time:.2f} seconds
    Error Occurred: {had_error}
    
    Evaluate this analysis as if you were a demanding sales leader who needs accurate, insightful information. Score and critique the following dimensions on a 1-5 scale (where 3 is merely adequate, 4 is good, and 5 is truly exceptional):
    
    1. Relevance (1-5): How accurately does the analysis capture the company's actual business model and service offerings? Does it miss any critical aspects of their business?
    
    2. Pain Points Quality (1-5): How well does the analysis identify the specific client pain points that gtmWizards claims to solve? Does it capture both the obvious and more subtle challenges they address?
    
    3. Differentiation Quality (1-5): How thoroughly does the analysis identify gtmWizards' unique value propositions and differentiators in the market? Does it distinguish between generic claims and true differentiators?
    
    4. Target Market Accuracy (1-5): How precisely does it identify gtmWizards' target customers, industries, and geographical focus? Is the segmentation accurate?
    
    5. Sales Insights (1-5): How actionable and useful are the insights for someone preparing to engage with this company? Would they genuinely help in crafting an effective outreach strategy?
    
    6. Search Quality (1-5): Based on the results, how strategic and effective were the search queries likely to have been? Did they target the right information?
    
    7. Information Validation (1-5): How thoroughly did the analysis cross-check and validate information rather than accepting single sources?
    
    8. Resilience (1-5): How effectively did the analysis work around information gaps or potentially conflicting data?
    
    9. Summarization Clarity (1-5): How clear, concise, and well-organized is the information? Does it present a coherent narrative or just disjointed facts?
    
    10. Practical Applicability (1-5): How immediately useful would this analysis be for a salesperson preparing for outreach? Does it provide genuine competitive advantage?
    
    11. Human-likeness (1-5): How human-like and natural is the analysis? Does it read like an experienced sales researcher wrote it, or is it formulaic and AI-generated?
    
    12. Overall Quality (1-5): Considering all factors, how would you rate the overall quality and usefulness of this analysis?
    
    FEEDBACK: Provide detailed, critical feedback that would help improve the analysis. Be specific about what information is missing, what conclusions are questionable, and what would make this more useful for a sales professional. Point out any inaccuracies or missed opportunities. Be as demanding as a real sales director would be.
    
    Format your response with each dimension clearly labeled with the score, and end with the detailed feedback paragraph.
    """
    
    # Get evaluation directly from LLM
    return await evaluate_with_llm(prompt, AgenticCompanyContextEval)

# Helper function to run async target and evaluator functions
async def run_comparison_evaluation(inputs, reference_outputs=None):
    """Run comparative evaluation of legacy and agentic approaches."""
    
    # Run both analysis approaches
    logger.info(f"Running legacy analysis for {inputs['company_url']}")
    legacy_outputs = await legacy_company_context_target(inputs)
    
    logger.info(f"Running agentic analysis for {inputs['company_url']}")
    agentic_outputs = await agentic_company_context_target(inputs)
    
    # Evaluate only the agentic approach with our enhanced schema
    logger.info(f"Evaluating agentic analysis for {inputs['company_url']}")
    agentic_evaluation = await agentic_company_context_evaluator(
        agentic_outputs, 
        reference_outputs or {}
    )
    
    # Prepare comparison data
    comparison = {
        "company_url": inputs["company_url"],
        "target_geography": inputs.get("target_geography", "Not specified"),
        "legacy_execution_time": legacy_outputs.get("execution_time", 0),
        "agentic_execution_time": agentic_outputs.get("execution_time", 0),
        "legacy_error": "error" in legacy_outputs,
        "agentic_error": "error" in agentic_outputs,
        "agentic_evaluation": agentic_evaluation.model_dump()
    }
    
    return {
        "legacy_outputs": legacy_outputs,
        "agentic_outputs": agentic_outputs,
        "agentic_evaluation": agentic_evaluation,
        "comparison": comparison
    }

# Run evaluation
async def evaluate_agentic_company_context():
    """Run evaluation for agentic company context analysis."""
    logger.info("Starting agentic company context analysis evaluation")
    
    results = []
    comparisons = []
    examples = client.list_examples(dataset_id=agentic_company_dataset.id)
    
    for example in examples:
        inputs = example.inputs
        reference_outputs = example.outputs
        
        logger.info(f"Evaluating company: {inputs['company_url']}")
        
        result = await run_comparison_evaluation(
            inputs,
            reference_outputs
        )
        
        # Log and save results to LangSmith
        try:
            run_id = client.create_run(
                name="agentic-company-context-analysis",
                inputs=inputs,
                outputs=result["agentic_outputs"],
                tags=["evaluation", "agentic-company-context"],
                run_type="chain",  # Using "chain" as we're evaluating a multi-step process
                metadata={
                    "agentic_evaluation": result["agentic_evaluation"].model_dump(),
                    "legacy_execution_time": result["legacy_outputs"].get("execution_time", 0),
                    "agentic_execution_time": result["agentic_outputs"].get("execution_time", 0),
                }
            )
            logger.info(f"Successfully created run in LangSmith with ID: {run_id}")
        except Exception as e:
            logger.error(f"Error creating run in LangSmith: {str(e)}")
        
        results.append(result)
        comparisons.append(result["comparison"])
    
    # Create a comparison report
    df = pd.DataFrame(comparisons)
    
    # Calculate averages for numerical metrics
    avg_metrics = {
        "avg_legacy_execution_time": df["legacy_execution_time"].mean(),
        "avg_agentic_execution_time": df["agentic_execution_time"].mean(),
        "avg_relevance": df["agentic_evaluation"].apply(lambda x: x["relevance"]).mean(),
        "avg_pain_points_quality": df["agentic_evaluation"].apply(lambda x: x["pain_points_quality"]).mean(),
        "avg_differentiation_quality": df["agentic_evaluation"].apply(lambda x: x["differentiation_quality"]).mean(),
        "avg_target_market_accuracy": df["agentic_evaluation"].apply(lambda x: x["target_market_accuracy"]).mean(),
        "avg_sales_insights": df["agentic_evaluation"].apply(lambda x: x["sales_insights"]).mean(),
        "avg_search_quality": df["agentic_evaluation"].apply(lambda x: x["search_quality"]).mean(),
        "avg_information_validation": df["agentic_evaluation"].apply(lambda x: x["information_validation"]).mean(),
        "avg_resilience": df["agentic_evaluation"].apply(lambda x: x["resilience"]).mean(),
        "avg_overall_quality": df["agentic_evaluation"].apply(lambda x: x["overall_quality"]).mean(),
    }
    
    # Log the summary metrics
    logger.info(f"Evaluation summary: {json.dumps(avg_metrics, indent=2)}")
    
    # Save the comparison report to CSV
    report_path = os.path.join('logs', f'agentic_eval_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(report_path, index=False)
    logger.info(f"Saved evaluation report to {report_path}")
    
    return {
        "results": results,
        "comparisons": comparisons,
        "summary_metrics": avg_metrics,
        "report_path": report_path
    }

# Main function to run evaluation
async def run_evaluation():
    """Run agentic company context evaluation focusing only on gtmWizards."""
    logger.info("Starting focused evaluation on gtmWizards.com")
    
    # Get or create dataset
    agentic_company_dataset_name = "gtmWizards Focused Evaluation"
    agentic_company_dataset = get_or_create_dataset(
        client=client,
        name=agentic_company_dataset_name,
        description="Rigorous evaluation of agentic company context analysis specifically for gtmWizards"
    )
    
    # Add examples to agentic company context dataset
    add_examples_if_empty(
        client=client,
        dataset_id=agentic_company_dataset.id,
        inputs_list=[{"company_url": example["company_url"], "target_geography": example["target_geography"]} 
                for example in agentic_company_context_examples],
        outputs_list=[{"expected_focus": example["expected_focus"]} 
                for example in agentic_company_context_examples]
    )
    
    # Run the evaluation
    return await evaluate_agentic_company_context()

# Run evaluation if this file is executed directly
if __name__ == "__main__":
    logger.info("Starting rigorous gtmWizards.com evaluation")
    try:
        # Verify API keys are properly set
        logger.info(f"OpenRouter API key available: {bool(openrouter_api_key)}")
        logger.info(f"Tavily API key available: {bool(tavily_api_key)}")
        
        # Run with the standard model only first to debug
        print("\n=== RUNNING WITH STANDARD MODEL (HAIKU) ===")
        logger.info("Initializing Haiku model")
        
        # Set environment variables directly to ensure they're available
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        llm = ChatOpenRouter(
            openai_api_key=openrouter_api_key,
            model="anthropic/claude-3.5-haiku-20241022:beta",
            temperature=0.1,
            streaming=False
        )
        
        logger.info("Starting evaluation with Haiku model")
        results_haiku = asyncio.run(run_evaluation())
        logger.info("Completed evaluation with Haiku model")
        
        # Only proceed to Sonnet if Haiku was successful
        if results_haiku and "comparisons" in results_haiku and len(results_haiku["comparisons"]) > 0:
            print("\n=== RUNNING WITH MORE CAPABLE MODEL (SONNET) ===")
            logger.info("Initializing Sonnet model")
            llm = ChatOpenRouter(
                openai_api_key=openrouter_api_key,
                model="anthropic/claude-3.5-sonnet-20241022:beta",
                temperature=0.1,
                streaming=False
            )
            
            logger.info("Starting evaluation with Sonnet model")
            results_sonnet = asyncio.run(run_evaluation())
            logger.info("Completed evaluation with Sonnet model")
            
            # Compare results between models
            print("\n=== MODEL COMPARISON ===")
            haiku_metrics = results_haiku["summary_metrics"]
            sonnet_metrics = results_sonnet["summary_metrics"]
            
            print("Metric            | Haiku | Sonnet | Difference")
            print("------------------|-------|--------|----------")
            for metric in haiku_metrics:
                if metric.startswith("avg_"):
                    metric_name = metric[4:]
                    haiku_val = haiku_metrics[metric]
                    sonnet_val = sonnet_metrics[metric]
                    diff = sonnet_val - haiku_val
                    print(f"{metric_name:18} | {haiku_val:.2f} | {sonnet_val:.2f} | {diff:+.2f}")
            
            print("\nDetailed reports saved to:")
            print(f"- Haiku: {results_haiku['report_path']}")
            print(f"- Sonnet: {results_sonnet['report_path']}")
        else:
            logger.error("Haiku evaluation failed or returned empty results - skipping Sonnet evaluation")
            print("\n!!! EVALUATION WITH HAIKU MODEL FAILED - SKIPPING SONNET !!!")
            print("Check the logs for more details on the errors")
        
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        logger.exception("Detailed exception:") 
from langsmith import Client
from pydantic import BaseModel, Field
import asyncio
import os
import sys
import json
import logging
import langsmith.utils

# Add the parent directory to sys.path so we can import from ai_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom OpenRouter connection
from old_files.company_context_workflow import ChatOpenRouter, analyze_company_context

# Setup logging
logging.basicConfig(level=logging.INFO)
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

# Initialize the OpenRouter client with the haiku model
llm = ChatOpenRouter(
    openai_api_key=openrouter_api_key,
    model="anthropic/claude-3.5-haiku-20241022:beta",
    temperature=0.1,
    streaming=False
)

# Define evaluation schema
class CompanyContextEval(BaseModel):
    """Evaluation schema for company context analysis."""
    relevance: int = Field(description="Score 1-5 on how relevant the generated context is to the company's actual business", ge=1, le=5)
    pain_points_quality: int = Field(description="Score 1-5 on how well the analysis identifies specific pain points the company claims to solve", ge=1, le=5)
    differentiation_quality: int = Field(description="Score 1-5 on how well the analysis identifies true differentiators", ge=1, le=5)
    target_market_accuracy: int = Field(description="Score 1-5 on accuracy of identified target markets and geography", ge=1, le=5)
    sales_insights: int = Field(description="Score 1-5 on useful sales insights that could help in outreach", ge=1, le=5)
    overall_quality: int = Field(description="Overall quality score 1-5", ge=1, le=5)
    feedback: str = Field(description="Specific feedback on strengths and areas for improvement")

# Create test dataset - just one example for simplicity
company_context_examples = [
    {
        "company_url": "https://www.datadog.com",
        "target_geography": "North America",
        "expected_focus": "Cloud monitoring and analytics platform focusing on infrastructure monitoring, application performance, and log management for DevOps teams. Primary pain points addressed include cloud visibility, troubleshooting complex systems, and reducing downtime."
    }
]

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
company_dataset_name = "Company Context Analysis Evaluation"
company_dataset = get_or_create_dataset(
    client=client,
    name=company_dataset_name,
    description="Evaluates company context analysis workflow"
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

# Add examples to company context dataset
add_examples_if_empty(
    client=client,
    dataset_id=company_dataset.id,
    inputs_list=[{"company_url": example["company_url"], "target_geography": example["target_geography"]} 
            for example in company_context_examples],
    outputs_list=[{"expected_focus": example["expected_focus"]} 
            for example in company_context_examples]
)

# Define target function for evaluation
async def company_context_target(inputs: dict) -> dict:
    """Target function for company context analysis."""
    try:
        company_url = inputs["company_url"]
        target_geography = inputs.get("target_geography")
        
        # Use our existing company context analysis function
        result = await analyze_company_context(
            company_url=company_url,
            model_name="anthropic/claude-3.5-haiku-20241022:beta",
            target_geography=target_geography
        )
        
        return {"company_context": result}
    except Exception as e:
        logger.error(f"Error in company context analysis: {str(e)}")
        return {"error": str(e)}

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
    """Extract structured data from LLM response"""
    # Simple parsing approach - extract key-value pairs
    data = {}
    
    # For numerical ratings (1-5)
    for field_name, field in schema_class.__annotations__.items():
        if field_name != "feedback":
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
    
    # Extract feedback (assuming it's the last part after "feedback" or similar)
    import re
    feedback_match = re.search(r"feedback\s*[:-]\s*(.*?)(?:\n\n|\Z)", content, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        data["feedback"] = feedback_match.group(1).strip()
    else:
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
async def company_context_evaluator(outputs: dict, reference_outputs: dict) -> CompanyContextEval:
    """Evaluates the quality of company context analysis."""
    company_context = outputs.get("company_context", {})
    expected_focus = reference_outputs.get("expected_focus", "")
    
    # Construct the evaluation prompt
    prompt = f"""
    You are evaluating the quality of a company context analysis for sales purposes.
    
    EXPECTED FOCUS:
    {expected_focus}
    
    GENERATED COMPANY CONTEXT:
    {json.dumps(company_context, indent=2)}
    
    Evaluate the analysis on these dimensions:
    1. Relevance (1-5): How relevant is the generated context to the company's actual business?
    2. Pain Points Quality (1-5): How well does the analysis identify specific pain points the company claims to solve?
    3. Differentiation Quality (1-5): How well does the analysis identify true differentiators?
    4. Target Market Accuracy (1-5): How accurately are target markets and geography identified?
    5. Sales Insights (1-5): How useful are the sales insights for outreach?
    6. Overall Quality (1-5): The overall quality of the analysis.
    
    Also provide specific feedback on strengths and areas for improvement.
    
    Format your response with each dimension rating clearly labeled with the dimension name followed by the score,
    and end with a paragraph of feedback.
    """
    
    # Get evaluation directly from LLM
    return await evaluate_with_llm(prompt, CompanyContextEval)

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

# Run evaluation
async def evaluate_company_context():
    """Run evaluation for company context analysis."""
    logger.info("Starting company context analysis evaluation")
    
    results = []
    examples = client.list_examples(dataset_id=company_dataset.id)
    
    for example in examples:
        inputs = example.inputs
        reference_outputs = example.outputs
        
        result = await run_async_evaluation(
            company_context_target,
            company_context_evaluator,
            inputs,
            reference_outputs
        )
        
        # Log and save results
        try:
            client.create_run(
                name="company-context-analysis",
                inputs=inputs,
                outputs=result["outputs"],
                tags=["evaluation", "company-context"],
                run_type="llm",  # Using "llm" as the run_type since we're evaluating LLM outputs
                metadata={
                    "evaluation": result["evaluation"].model_dump()
                }
            )
            logger.info("Successfully created run in LangSmith")
        except Exception as e:
            logger.error(f"Error creating run in LangSmith: {str(e)}")
        
        results.append(result)
    
    logger.info(f"Company context evaluation complete. Results: {json.dumps([r['evaluation'].model_dump() for r in results], indent=2)}")
    return results

# Main function to run evaluation
async def run_evaluation():
    """Run company context evaluation and return results."""
    company_results = await evaluate_company_context()
    return {"company_context_results": company_results}

# Run evaluation if this file is executed directly
if __name__ == "__main__":
    logger.info("Starting simplified company context evaluation")
    asyncio.run(run_evaluation())
    logger.info("Evaluation complete")
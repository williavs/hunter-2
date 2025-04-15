import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Union
import pandas as pd
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Set up logging
logger = logging.getLogger(__name__)

class SPEAREmailOutput(BaseModel):
    """Output format for all generated emails."""
    emailbody1: str = Field(description="Social proof/case study focused email")
    emailbody2: str = Field(description="Business pain/fear focused email")
    emailbody3: str = Field(description="Innovation/differentiation focused email")
    emailbody4: str = Field(description="ROI/metrics focused email")


class SPEAREmailGenerator:
    """
    Generator for JMM-style SPEAR emails with different strategic focuses.
    Generates all four email types at once in parallel.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4.1",
                 max_concurrent: int = 10):
        """
        Initialize the SPEAR Email Generator with model settings.
        
        Args:
            model_name: Model to use for generation
            max_concurrent: Maximum concurrent operations
        """
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.llm = None
        self._setup_llm()
        
        # Create semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Set up the email prompt template
        self._setup_prompt_template()
    
    def _setup_llm(self):
        """Set up the language model client."""
        try:
            # Initialize ChatOpenAI with OpenAI API
            self.llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=self.model_name,
                temperature=0.7,
                streaming=False,
                timeout=90  # Adding a longer timeout for reliability
            )
            logger.info(f"LLM initialized with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _setup_prompt_template(self):
        """Set up the prompt template for generating all emails at once."""
        
        # System prompt with JMM methodology guidance
        self.system_prompt = """
        You are an expert at writing ultra-short, high-impact SPEAR emails following the Justin Michael Method (JMM).
        
        Follow these specific JMM guidelines:
        1. Create painfully short emails (3 sentences max)
        2. Focus on business value and pain points
        3. Be specific with metrics (use precise numbers like 32.6% not 30%)
        4. Keep emails compact with minimal spacing (intentional "ugliness")
        5. Don't use pleasantries like "hope you're doing well"
        6. Create subtle tension or fear (without being pushy)
        7. Include specific customer examples with real metrics
        8. Avoid traditional email structure and formality
        9. Use short, punchy sentences with minimal fluff
        10. Each email should be 3 sentences maximum

        IMPORTANT: NOBODY CARES ABOUT WHAT THE SELLING COMPANY DOES. THEY ONLY CARE ABOUT WHAT THEY CAN DO FOR THEIR PROBLEMS. WHEN POSSIBLE EMAILS SHOW PROSPECTS WHAT THEY CAN DO WITH WHAT THEY CAN DO SELLERS PRODUCT. 

        IMPORTANT: NEVER FOCUS ON SELLING COMPANY OR SELLING COMPANY FEATURES, NO FEATURE DUMPING EVER. 
        
        IMPORTANT: Only use the provided case studies for social proof examples. DO NOT invent or hallucinate customer stories, companies, or metrics. If no case studies are provided, use generic statements without naming specific companies.
        
        You will generate four different JMM-style SPEAR emails for the same contact, each with a different strategic focus.
        Direct the emails to the individual contact, addressing their specific industry, role, and potential needs.
        
        YOUR RESPONSE MUST BE VALID JSON. NO OTHER TEXT BEFORE OR AFTER. 
        NO MARKDOWN. NO CODE BLOCKS. JUST PURE JSON THAT CAN BE PARSED DIRECTLY.
        
        RESPONSE FORMAT:
        {
          "emailbody1": "Social proof email focusing on similar customers who have succeeded with our product",
          "emailbody2": "Business pain/fear email emphasizing risks or opportunity costs",
          "emailbody3": "Innovation/differentiation email highlighting unique capabilities (IMPORTANT: THIS IS NOT A FEATURE DUMP, IT SHOULD FOCUS ON THE BUYING COMPANY, NOT THE SELLING COMPANY)",
          "emailbody4": "ROI/metrics email focusing on specific business outcomes with precise numbers"
        }
        
        Use proper JSON with double quotes around keys and values, escape quotes within text values with backslash, and ensure there are no trailing commas or syntax errors.
        """
        
        # Human prompt template for generating all emails at once
        self.prompt_template = """
        Create four different JMM-style SPEAR emails for this contact, each with a different strategic focus.
        
        Personalize the emails using these contact details:
        {contact_context}
        
        Case Studies (use these EXACT examples, do not hallucinate):
        {case_studies}
        
        For any email, make inferences about the potential needs based on their role, department, and other details.
        Use information like performance metrics, age, salary, job title, etc. to craft highly targeted emails.
        Use any available metrics in the contact data to create specific, personalized email content.
        
        The four required emails are:
        1. Social Proof Email: Highlight similar customers who've succeeded with our product. ONLY reference companies and metrics from the provided case studies.
        2. Business Pain/Fear Email: Emphasize risks or opportunity costs of not acting
        3. Innovation/Differentiation Email: Showcase unique capabilities competitors don't have(IMPORTANT: THIS IS NOT A FEATURE DUMP, IT SHOULD FOCUS ON THE BUYING COMPANY, NOT THE SELLING COMPANY)
        4. ROI/Metrics Email: Focus on specific business outcomes with precise numbers
        
        Remember to follow JMM principles: painfully short (max 3 sentences), precise metrics (32.6% not 30%), compressed format, and no pleasantries.
        Create metrics and business results that would be relevant to this specific contact.
        
        YOUR ENTIRE RESPONSE MUST BE A SINGLE VALID JSON OBJECT, NOTHING ELSE.
        
        THE EXACT JSON STRUCTURE SHOULD BE:
        {{
          "emailbody1": "Your social proof email text here",
          "emailbody2": "Your business pain/fear email text here",
          "emailbody3": "Your innovation/differentiation email text here ",
          "emailbody4": "Your ROI/metrics email text here"
        }}

        IMPORTANT: NOBODY CARES ABOUT WHAT THE SELLING COMPANY DOES. THEY ONLY CARE ABOUT WHAT THEY CAN DO FOR THEIR PROBLEMS. WHEN POSSIBLE EMAILS SHOW PROSPECTS WHAT THEY CAN DO WITH WHAT THEY CAN DO SELLERS PRODUCT. 

        IMPORTANT: NEVER FOCUS ON SELLING COMPANY OR SELLING COMPANY FEATURES, NO FEATURE DUMPING EVER. 
        
        
        Your response must ONLY be the JSON object above. No explanations, no markdown, no code blocks - JUST THE JSON.
        All string values should use properly escaped double quotes: \\" for quotes within text.
        Make sure the JSON is valid and can be parsed by json.loads().
        """
    
    async def test_api_key(self) -> bool:
        """Test if the OpenRouter API key is valid."""
        try:
            test_message = "Test message to validate API key."
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=test_message)
            ]
            await self.llm.agenerate([messages])
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False
    
    async def _generate_all_emails(self, contact_context: Dict[str, Any], company_context: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate all four email types at once for a single contact."""
        try:
            # Extract key information from contact context for better emails
            contact_info = contact_context.copy()
            
            # Add inferred data (like industry) if not present
            if ('Industry' not in contact_info or not contact_info.get('Industry')) and 'Department' in contact_info:
                contact_info['Industry'] = self._infer_industry_from_department(contact_info.get('Department', ''))
            
            # Get case studies from company context
            case_studies = ""
            if company_context and "case_studies" in company_context:
                case_studies = company_context["case_studies"]
            
            # Format the prompt with contact context and case studies
            formatted_prompt = self.prompt_template.format(
                contact_context=json.dumps(contact_info),
                case_studies=case_studies if case_studies else "No specific case studies provided. Do not reference or name any specific companies in the social proof email."
            )
            
            # Set up messages for the model
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=formatted_prompt)
            ]
            
            # Generate the emails
            response = await self.llm.agenerate([messages])
            content = response.generations[0][0].text.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {content}")
            
            try:
                # Parse the JSON response
                result = json.loads(content)
                
                # Ensure all required fields are present
                default_emails = {
                    "emailbody1": "Error generating social proof email.",
                    "emailbody2": "Error generating business pain email.",
                    "emailbody3": "Error generating innovation email.",
                    "emailbody4": "Error generating ROI email."
                }
                
                # Merge with defaults to ensure all keys exist
                for key in default_emails:
                    if key not in result or not result[key].strip():
                        result[key] = default_emails[key]
                        logger.warning(f"Missing or empty {key} in LLM response, using default.")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw content causing error: {content}")
                
                return {
                    "emailbody1": "Error: Invalid JSON response from email generation service.",
                    "emailbody2": "Error: Invalid JSON response from email generation service.",
                    "emailbody3": "Error: Invalid JSON response from email generation service.",
                    "emailbody4": "Error: Invalid JSON response from email generation service."
                }
            
        except Exception as e:
            logger.error(f"Error generating emails: {str(e)}")
            return {
                "emailbody1": f"Error: {str(e)}",
                "emailbody2": f"Error: {str(e)}",
                "emailbody3": f"Error: {str(e)}",
                "emailbody4": f"Error: {str(e)}"
            }
    
    def _infer_industry_from_department(self, department: str) -> str:
        """Infer a potential industry based on department."""
        department = department.lower() if department else ""
        
        industry_map = {
            'sales': 'Sales Technology',
            'marketing': 'Marketing Technology',
            'engineering': 'Software Development',
            'hr': 'Human Resources',
            'finance': 'Financial Services',
            'legal': 'Legal Services',
            'operations': 'Operations Management',
            'it': 'Information Technology',
            'support': 'Customer Service',
            'product': 'Product Development',
            'research': 'Research & Development'
        }
        
        for key, value in industry_map.items():
            if key in department:
                return value
        
        return "Technology"  # Default fallback
    
    async def generate_spear_emails(self, contacts_df: pd.DataFrame, company_context: Dict[str, Any] = None, selected_indices: List[int] = None) -> pd.DataFrame:
        """
        Generate SPEAR emails for selected contacts in the dataframe.
        
        Args:
            contacts_df: DataFrame containing contact information
            company_context: Optional dictionary with company information (not required)
            selected_indices: Indices of rows to process (None for all rows)
        
        Returns:
            DataFrame with added email columns
        """
        # Make a copy of the dataframe to avoid modifying the original
        df = contacts_df.copy()
        
        # Add email columns if they don't exist
        for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
            if col not in df.columns:
                df[col] = ""
        
        # Determine which rows to process
        indices_to_process = selected_indices if selected_indices is not None else df.index.tolist()
        
        # Validate indices to ensure they're within range
        valid_indices = [idx for idx in indices_to_process if idx < len(df)]
        if len(valid_indices) < len(indices_to_process):
            logger.warning(f"Removed {len(indices_to_process) - len(valid_indices)} out-of-range indices")
            indices_to_process = valid_indices
        
        # Process contacts in true parallel
        async def process_contact(idx):
            try:
                # Convert the row to a dictionary
                contact_context = df.iloc[idx].to_dict()
                
                # Generate all emails at once
                async with self.semaphore:
                    result = await self._generate_all_emails(contact_context, company_context)
                
                # Return the results instead of modifying df directly
                return idx, result, None  # None means no error
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing contact at index {idx}: {error_msg}")
                return idx, None, error_msg  # Return the error
        
        # Process all contacts concurrently in true parallel
        tasks = [process_contact(idx) for idx in indices_to_process]
        results = await asyncio.gather(*tasks)
        
        # Now safely update the dataframe sequentially after all async operations are complete
        for idx, email_results, error in results:
            if error:
                # Log the error and add it to the dataframe
                logger.error(f"Failed to generate emails for contact at index {idx}: {error}")
                df.at[idx, "error"] = error
            elif email_results:
                # Add email results to the dataframe
                for key, value in email_results.items():
                    df.at[idx, key] = value
        
        return df


async def generate_spear_emails(df: pd.DataFrame, company_context: Dict[str, Any] = None, selected_indices: List[int] = None, model_name: str = "gpt-4.1") -> pd.DataFrame:
    """
    Generate SPEAR emails for contacts in a DataFrame.
    
    Args:
        df: DataFrame with contact information
        company_context: Optional company context dictionary
        selected_indices: Optional list of DataFrame indices to process (if None, process all)
        model_name: Model to use for generation
        
    Returns:
        DataFrame with original data plus generated emails
    """
    try:
        # Validate input dataframe
        if df is None or df.empty:
            logger.error("Input dataframe is None or empty")
            return pd.DataFrame()
        
        # Check if API key is available
        if not os.environ.get("OPENROUTER_API_KEY"):
            logger.error("OpenRouter API key not found in environment variables")
            error_df = df.copy()
            for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
                if col not in error_df.columns:
                    error_df[col] = "Error: OpenRouter API key not found"
            return error_df
            
        # Validate selected indices
        if selected_indices:
            # Make sure indices are within range
            valid_indices = [idx for idx in selected_indices if idx < len(df)]
            if len(valid_indices) < len(selected_indices):
                logger.warning(f"Removed {len(selected_indices) - len(valid_indices)} out-of-range indices")
                selected_indices = valid_indices
                
            if not selected_indices:
                logger.error("No valid indices to process")
                return df.copy()
        
        logger.info(f"Initializing SPEAREmailGenerator with model {model_name}")
        generator = SPEAREmailGenerator(model_name=model_name)
        
        # Log the inputs for debugging
        logger.info(f"Processing {len(selected_indices) if selected_indices else 'all'} contacts")
        
        # Generate emails
        result_df = await generator.generate_spear_emails(df, company_context, selected_indices)
        
        # Validate the results
        if result_df is not None:
            # Check if we have any non-empty email bodies
            if selected_indices:
                has_content = False
                for idx in selected_indices:
                    if idx >= len(result_df):
                        continue
                    for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
                        if col in result_df.columns:
                            value = result_df.iloc[idx][col]
                            if isinstance(value, str) and value.strip():
                                has_content = True
                                break
                
                if not has_content:
                    logger.warning("No email content was generated for any selected contacts")
            
            # Log success
            logger.info(f"Successfully generated emails for {len(selected_indices) if selected_indices else 'all'} contacts")
            return result_df
        else:
            logger.error("Email generation returned None dataframe")
            return df.copy()  # Return a copy of the original dataframe if result is None
    except Exception as e:
        logger.error(f"Failed to generate emails: {str(e)}", exc_info=True)
        # Return original dataframe with error columns
        error_df = df.copy()
        for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
            if col not in error_df.columns:
                error_df[col] = f"Error: {str(e)}"
        return error_df 
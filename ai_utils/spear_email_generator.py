import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Union
import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import dotenv_values
from ai_utils import prompt_library
import streamlit as st

# Set up logging
logger = logging.getLogger(__name__)

class SPEAREmailGenerator:
    """
    Generator for SPEAR emails using dynamic prompt templates from the prompt library.
    Supports any number and type of prompts, including those with advanced output formats (e.g., JSON).
    """
    SYSTEM_PROMPT = (
        "You are a world-class sales email generator. Always embody the following principles and style, regardless of the prompt or context:\n"
        "- Relentless persistence: like a honey badger, never discouraged by rejection, always moving forward.\n"
        "- Resilience and resourcefulness: find creative, multi-channel paths to decision-makers.\n"
        "- Strategic patience: know when to advance, when to hold position.\n"
        "- Pattern recognition: spot and leverage patterns in objections, buying behavior, and stakeholder dynamics.\n"
        "- Use pattern interruption: break through the noise with unexpected, memorable, or counterintuitive approaches.\n"
        "- Employ metaphors and vivid imagery to create memorable, bold communication.\n"
        "- Use humor, self-deprecation, and challenger questions to diffuse tension and provoke thought.\n"
        "- Always lead with 'what's in it for them' (WIFM), not about you or your company.\n"
        "- Use future visioning: help prospects imagine a dramatically improved future state.\n"
        "- Use referential value: talk about how others like them have succeeded, not just what you do.\n"
        "- Be direct, punchy, and concise—no walls of text, no fluff, no jargon.\n"
        "- Use short paragraphs, line breaks, and formatting for readability.\n"
        "- Vary your call to action, avoid standard 'can we meet?' closes.\n"
        "- Use empathy, open questions, and adaptive mirroring in objection handling.\n"
        "- Detach from the outcome—never sound desperate or 'salesy.'\n"
        "- Use challenger/contrarian questions to provoke reflection and engagement.\n"
        "- If the prompt requests a structured output (e.g., JSON), return only that structure.\n"
        "- Always deliver value first—don't make the email about your company until it's clear why the recipient should care.\n"
        "- Use clever transitions and personalization when possible.\n"
    )

    def __init__(self, 
                 model_name: str = "gpt-4.1", 
                 max_concurrent: int = 10,
                 ai_mode: str = "Proxy",
                 openai_api_key: Optional[str] = None,
                 proxy_api_key: Optional[str] = None,
                 proxy_base_url: str = "https://llm.data-qa.justworks.com"
        ):
            self.model_name = model_name
            self.max_concurrent = max_concurrent
            self.ai_mode = ai_mode
            self.openai_api_key = openai_api_key
            self.proxy_api_key = proxy_api_key
            self.proxy_base_url = proxy_base_url
            self.llm = None
            self._setup_llm()
            self.semaphore = asyncio.Semaphore(max_concurrent)

    def _setup_llm(self):
        try:
            # Use instance attributes for API key and model selection
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
                temperature=0.7,
                streaming=False,
                timeout=90
            )
            if base_url:
                llm_kwargs["base_url"] = base_url
            self.llm = ChatOpenAI(**llm_kwargs)
            logger.info(f"LLM initialized with model {model} (mode: {mode})")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _build_user_prompt(self, contact_context, company_context, prompt_templates):
        """
        Dynamically build a user prompt that injects all selected email prompt instructions.
        If only one prompt, use its instructions directly. If multiple, enumerate them and specify the output structure.
        Args:
            contact_context: Dict containing info about the individual contact.
            company_context: Dict containing info about the user's company (from hunter.py analysis) and potentially 'case_studies'.
            prompt_templates: List of prompt template dicts to use.
        """
        prompt_blocks = []
        for i, template in enumerate(prompt_templates, 1):
            prompt_blocks.append(f"EMAIL TYPE {i}: {template['name']}\nInstructions: {template['prompt_text']}")
        prompt_section = "\n\n".join(prompt_blocks)
        # Output contract
        if len(prompt_templates) == 1:
            output_note = ("\n\nReturn ONLY the output for this email type. If the prompt requests JSON, return only the JSON object.")
        else:
            ids = [t['id'] for t in prompt_templates]
            output_note = (f"\n\nReturn a JSON object with a key for each email type: {ids}. "
                           "Each value should be the output for that email type (plain text or JSON as specified in the instructions). ")
        # Context
        context_block = (
            f"Contact context: {json.dumps(contact_context)}\n"
            f"Company context: {json.dumps(company_context) if company_context else '{}'}\n"
        )
        return f"{prompt_section}\n\n{context_block}{output_note}"

    async def generate_emails_with_prompts(self, contact_context: Dict[str, Any], prompt_templates: List[Dict], company_context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate emails for a contact using a list of prompt templates from the prompt library.
        Each template should have 'id', 'name', 'description', and 'prompt_text'.
        Handles both plain text and JSON-structured outputs.
        """
        results = {}
        tasks = []
        semaphore = self.semaphore

        # Build the user prompt once for all selected templates
        user_prompt = self._build_user_prompt(contact_context, company_context, prompt_templates)
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        async def generate_all():
            try:
                response = await self.llm.agenerate([messages])
                content = response.generations[0][0].text.strip()
                # Try to parse JSON if multiple prompts or if JSON is expected
                if len(prompt_templates) > 1 or 'Return as JSON' in user_prompt or content.startswith('{'):
                    try:
                        parsed = json.loads(content)
                        # Map outputs to template ids
                        for i, template in enumerate(prompt_templates):
                            val = parsed.get(template['id'])
                            results[template['id']] = val if val is not None else parsed.get(str(i+1), "")
                    except Exception:
                        # Fallback: assign raw content to all
                        for template in prompt_templates:
                            results[template['id']] = content
                else:
                    # Single prompt: assign output directly
                    results[prompt_templates[0]['id']] = content
            except Exception as e:
                for template in prompt_templates:
                    results[template['id']] = f"Error: {str(e)}"

        await generate_all()
        return results

async def generate_emails_for_contacts_with_prompts(df, prompt_templates, 
    company_context=None, 
    selected_indices=None, 
    model_name="gpt-4.1", 
    max_concurrent=10,
    ai_mode: str = "Proxy",
    openai_api_key: Optional[str] = None,
    proxy_api_key: Optional[str] = None,
    proxy_base_url: str = "https://llm.data-qa.justworks.com",
):
    """
    Generate emails for contacts in a DataFrame using up to 4 selected prompt templates from the prompt library.
    Adds a column for each template id to the DataFrame. Handles both plain text and JSON-structured outputs.
    """
    if df is None or df.empty or not prompt_templates:
        return pd.DataFrame() if df is None else df.copy()
    # Limit to maximum 4 email types
    prompt_templates = prompt_templates[:4]
    generator = SPEAREmailGenerator(
        model_name=model_name, # Keep for now, but _setup_llm overrides based on mode
        max_concurrent=max_concurrent,
        ai_mode=ai_mode,
        openai_api_key=openai_api_key,
        proxy_api_key=proxy_api_key,
        proxy_base_url=proxy_base_url
    )
    df = df.copy()
    indices_to_process = selected_indices if selected_indices is not None else df.index.tolist()
    # Add columns for each template id if not present
    for template in prompt_templates:
        col = template['id']
        if col not in df.columns:
            df[col] = ""
    async def process_contact(idx):
        contact_context = df.iloc[idx].to_dict()
        emails = await generator.generate_emails_with_prompts(contact_context, prompt_templates, company_context)
        return idx, emails
    tasks = [process_contact(idx) for idx in indices_to_process]
    results = await asyncio.gather(*tasks)
    for idx, emails in results:
        for template in prompt_templates:
            col = template['id']
            val = emails.get(col, "")
            if isinstance(val, dict):
                df.at[idx, col] = json.dumps(val, ensure_ascii=False, indent=2)
            else:
                df.at[idx, col] = val
    return df

async def generate_spear_emails(df: pd.DataFrame, company_context: Dict[str, Any] = None, selected_indices: List[int] = None, model_name: str = "gpt-4.1") -> pd.DataFrame:
    """
    Generate SPEAR emails for contacts in a DataFrame.
    Supports up to 4 email types.
    """
    try:
        if df is None or df.empty:
            logger.error("Input dataframe is None or empty")
            return pd.DataFrame()
        if not getattr(st.session_state, "proxy_api_key", None):
            logger.error("API key not found in session state")
            error_df = df.copy()
            # Only support up to 4 email columns
            for i in range(1, 5):
                col = f"emailbody{i}"
                if col not in error_df.columns:
                    error_df[col] = "Error: API key not found"
            return error_df
        if selected_indices:
            valid_indices = [idx for idx in selected_indices if idx < len(df)]
            if len(valid_indices) < len(selected_indices):
                logger.warning(f"Removed {len(selected_indices) - len(valid_indices)} out-of-range indices")
                selected_indices = valid_indices
            if not selected_indices:
                logger.error("No valid indices to process")
                return df.copy()
        logger.info(f"Initializing SPEAREmailGenerator with model {model_name}")
        generator = SPEAREmailGenerator(model_name=model_name)
        logger.info(f"Processing {len(selected_indices) if selected_indices else 'all'} contacts")
        # Only use up to 4 email columns
        prompt_templates = prompt_library.get_selected_prompts()[:4]
        result_df = await generate_emails_for_contacts_with_prompts(
            df, prompt_templates, company_context, selected_indices, model_name=model_name
        )
        if result_df is not None:
            if selected_indices:
                has_content = False
                for idx in selected_indices:
                    if idx >= len(result_df):
                        continue
                    for template in prompt_templates:
                        col = template['id']
                        if col in result_df.columns:
                            value = result_df.iloc[idx][col]
                            if isinstance(value, str) and value.strip():
                                has_content = True
                                break
                    if has_content:
                        break
                if not has_content:
                    logger.warning("No email content was generated for any selected contacts")
            logger.info(f"Successfully generated emails for {len(selected_indices) if selected_indices else 'all'} contacts")
            return result_df
        else:
            logger.error("Email generation returned None dataframe")
            return df.copy()
    except Exception as e:
        logger.error(f"Failed to generate emails: {str(e)}", exc_info=True)
        error_df = df.copy()
        for i in range(1, 5):
            col = f"emailbody{i}"
            if col not in error_df.columns:
                error_df[col] = f"Error: {str(e)}"
        return error_df 
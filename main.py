import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import base64
import asyncio
from io import StringIO
from utils.scraper import process_websites_parallel, website_mapping_dialog, guess_website_column
# Import directly from the renamed file
from ai_utils.personality_analyzer import analyze_personality, PersonalityAnalyzer
# Import the new company context workflow
from ai_utils.company_context_workflow import analyze_company_context
# Import the company scraper
from utils.company_scraper import company_scraper_dialog, CompanyScraper
# Import helper functions
from utils.data_helpers import load_csv_data, get_download_link, has_name_components
# Import API utilities
from utils.api_utils import test_api_keys

import os
import logging
from dotenv import load_dotenv
import requests
from utils.logging_config import configure_logging, configure_langsmith_tracing

# Use the centralized logging configuration
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables from .env file")
load_dotenv()

# Configure LangSmith tracing if API key is available
configure_langsmith_tracing()

# Log environment variable status (without exposing full keys)
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")
logger.info(f"OPENROUTER_API_KEY loaded: {'Yes' if openrouter_key else 'No'}")
if openrouter_key:
    logger.debug(f"OPENROUTER_API_KEY starts with: {openrouter_key[:10]}...")
    logger.debug(f"OPENROUTER_API_KEY length: {len(openrouter_key)}")
logger.info(f"TAVILY_API_KEY loaded: {'Yes' if tavily_key else 'No'}")

# Define the fixed model name
FIXED_MODEL = "anthropic/claude-3.5-haiku-20241022:beta"

# ==========================
# Core Application Settings
# ==========================
# Only set page config when running this file directly (not through navigation)
# This prevents conflicts with st.navigation
if __name__ == "__main__":
    st.set_page_config(
        page_title="Contact Data Enrichment Tool",
        page_icon="📇",
        layout="wide", 
        initial_sidebar_state="expanded", 
        # Use wide layout for better data display
    )

# ==========================
# Helper Functions
# ==========================
async def run_personality_analysis(df, company_context=None):
    """Run personality analysis in async context"""
    # Make sure we have the correct name column information
    name_column = st.session_state.name_column
    logger.debug(f"Using name column: {name_column}")
    
    # If we have combined first and last names, make sure the full_name column exists
    if st.session_state.get("has_combined_names", False) and "full_name" in df.columns:
        logger.debug("Using combined first and last names in 'full_name' column")
        # The full_name column is already in the DataFrame
    elif st.session_state.get("website_mapping", {}).get("has_separate_names", False):
        # We need to combine first and last names
        first_name_column = st.session_state.website_mapping.get("first_name_column")
        last_name_column = st.session_state.website_mapping.get("last_name_column")
        
        if first_name_column and last_name_column and first_name_column in df.columns and last_name_column in df.columns:
            logger.debug(f"Combining first name column '{first_name_column}' and last name column '{last_name_column}'")
            # Create a copy of the dataframe
            df = df.copy()
            
            # Combine first and last names into a new column
            df['full_name'] = df.apply(
                lambda row: f"{row[first_name_column]} {row[last_name_column]}".strip(), 
                axis=1
            )
            
            # Update the name column to use the full_name
            name_column = "full_name"
            st.session_state.name_column = name_column
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparing for personality analysis...")
    
    # Count rows with website content for logging
    rows_with_content = len(df[df['website_content'].notna() & (df['website_content'] != "")])
    logger.info(f"Analyzing {len(df)} contacts, of which {rows_with_content} have website content")
    
    try:
        # Debug check for API keys before analysis
        logger.debug("Checking API keys before personality analysis")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        
        if not openrouter_key:
            logger.error("OPENROUTER_API_KEY not found in environment")
            st.error("OpenRouter API key not found. Please set it in the API Keys Configuration section.")
            return df
        
        if not tavily_key:
            logger.error("TAVILY_API_KEY not found in environment")
            st.error("Tavily API key not found. Please set it in the API Keys Configuration section.")
            return df
        
        logger.debug(f"Starting personality analysis with API keys (first chars): OpenRouter={openrouter_key[:8]}..., Tavily={tavily_key[:8]}...")
        # Use the OpenRouter version of the analyzer with the fixed model
        status_text.text(f"Analyzing personalities with Claude via OpenRouter for {len(df)} contacts ({rows_with_content} with website content)...")
        progress_bar.progress(10)
        
        # Log company context if available
        if company_context:
            # Check if we have website content
            has_website_content = "website_content" in company_context and company_context["website_content"]
            if has_website_content:
                logger.info(f"Using company context with website content for analysis: {company_context.get('name', 'Unknown')}")
                status_text.text(f"Analyzing personalities with enhanced company website content...")
                
                # Display a more informative message about the website content
                content_length = len(company_context["website_content"])
                approx_pages = max(1, content_length // 5000)  # Rough estimate of page count
                logger.debug(f"Company website content: {content_length} characters, approximately {approx_pages} pages")
            else:
                logger.debug(f"Using company context for analysis: {company_context.get('name', 'Unknown')}")
        else:
            logger.debug("No company context provided for analysis")
        
        # Pass company context to the analyze_personality function
        result_df = await analyze_personality(df, model_name=FIXED_MODEL, company_context=company_context)
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("Personality analysis complete!")
        return result_df
    
    except Exception as e:
        logger.error(f"Error during personality analysis: {str(e)}")
        st.error(f"An error occurred during personality analysis: {str(e)}")
        return df
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "website_column" not in st.session_state:
    st.session_state.website_column = None
if "name_column" not in st.session_state:
    st.session_state.name_column = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "personality_analysis_complete" not in st.session_state:
    st.session_state.personality_analysis_complete = False
if "scraped_df" not in st.session_state:
    st.session_state.scraped_df = None
if "show_mapping_dialog" not in st.session_state:
    st.session_state.show_mapping_dialog = False
if "has_combined_names" not in st.session_state:
    st.session_state.has_combined_names = False
    
# Check for API keys and display warning if missing
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")


    
# Enhanced sidebar title (moved to streamlit_app.py as it's common)
# st.sidebar.markdown("""
# <h1 style="text-align: center; color: #FF5722; font-weight: bold; margin-bottom: 5px;">HUNTER</h1>
# <p style="text-align: center; color: #757575; font-size: 14px; margin-top: 0;">AI-Powered Sales Intelligence</p>
# """, unsafe_allow_html=True)

# Use native Streamlit components for the title and subtitle



# Use an icon to indicate missing API keys in the sidebar title
if not openrouter_key or not tavily_key:
    st.title("SETUP API KEYS IN SIDEBAR")
    st.sidebar.subheader("⚠️ API Keys Configuration (Required)")
    st.sidebar.link_button("Get Tavily API Key", "https://app.tavily.com/")
    st.sidebar.link_button("Get OpenRouter API Key", "https://openrouter.ai/settings/keys")

    # API Keys section
    with st.sidebar.expander("API Keys Configuration", expanded=not openrouter_key or not tavily_key):
        # OpenRouter API key input
        openrouter_api_key = st.text_input("OpenRouter API Key", 
                                        value=os.environ.get("OPENROUTER_API_KEY", ""), 
                                        type="password",
                                        help="Required for personality analysis with OpenRouter")
        
        tavily_api_key = st.text_input("Tavily API Key", 
                                    value=os.environ.get("TAVILY_API_KEY", ""), 
                                    type="password",
                                    help="Required for web search in personality analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save API Keys", key="save_api_keys_button"):
                # Remove any quotes or extra whitespace
                openrouter_api_key = openrouter_api_key.strip().replace('"', '')
                tavily_api_key = tavily_api_key.strip().replace('"', '')
                
                os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
                os.environ["TAVILY_API_KEY"] = tavily_api_key
                
                # Log key information (safely)
                if openrouter_api_key:
                    key_preview = openrouter_api_key[:8] + "..." if len(openrouter_api_key) > 8 else "[empty]"
                    logger.debug(f"Saved OpenRouter API key (first chars): {key_preview}")
                    logger.debug(f"OpenRouter API key length: {len(openrouter_api_key)}")
                
                st.success("API keys saved for this session!")
        
        with col2:
            if st.button("Test API Keys", key="test_api_keys_button"):
                with st.spinner("Testing API keys..."):
                    valid, message = asyncio.run(test_api_keys())
                    if valid:
                        st.success(message)
                    else:
                        st.error(message)
else:
    st.sidebar.success("LLM Juiced 🔥")

# Add Company Context section
with st.expander("Step 1. Setup Company Context", expanded=False):
    
    # Determine current workflow state
    has_scraped_content = "website_content" in st.session_state.get("company_context", {})
    has_generated_context = bool(st.session_state.get("company_context", {}).get("description", ""))
    context_approved = st.session_state.get("context_approved", False)
    
    # Display current state/progress
    col_status = st.columns(3)
    with col_status[0]:
        if has_scraped_content:
            st.success("✅ 1. Website scraped")
        else:
            st.info("1. Website scraping needed")
    
    with col_status[1]:
        if has_generated_context:
            st.success("✅ 2. Context generated")
        else:
            st.info("2. Context generation needed")
            
    with col_status[2]:
        if context_approved:
            st.success("✅ 3. Context approved")
        else:
            st.info("3. Context approval needed")
    
    # Step 1: Website Input & Combined Scrape+Generate Process
    st.markdown("### Step 1: Enter Company Website")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_url = st.text_input(
            "Company Website URL",
            value=st.session_state.get("company_url", ""),
            placeholder="https://example.com",
            help="Enter your company website URL to scrape and analyze"
        )
        
        # Add target geography field below company URL
        target_geography = st.text_input(
            "Target Geography (Where are you selling?)",
            value=st.session_state.get("target_geography", ""),
            placeholder="e.g., North America, EMEA, APAC, Global, etc.",
            help="Specify your target market or where you're selling - problems differ by region"
        )
        
    with col2:
        combined_process_button = st.button(
            "Scrape & Analyze", 
            key="scrape_analyze_btn",
            use_container_width=True,
            disabled=not company_url
        )
        
        # Store URL and geography in session state
        if company_url:
            st.session_state.company_url = company_url
        if target_geography:
            st.session_state.target_geography = target_geography
        
    # Handle the combined scraping and analysis process
    if combined_process_button:
        with st.spinner("Step 1: Scraping website content..."):
            try:
                # First open the scraper dialog to get content
                scraper = CompanyScraper(company_url, max_pages=10)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Define progress callback
                def update_progress(current, total):
                    progress = min(current / total, 1.0) * 0.5  # First 50% for scraping
                    progress_bar.progress(progress)
                    status_text.text(f"Scraping page {current} of {total}...")
                
                # Run the scraper
                pages_content = asyncio.run(scraper.scrape_website(progress_callback=update_progress))
                
                # Get content from the scraper
                combined_content = scraper.get_combined_content()
                
                # Update progress
                progress_bar.progress(0.5)  # 50% complete after scraping
                status_text.text("Scraping complete. Now analyzing content...")
                
                # Save to session state
                if "company_context" not in st.session_state:
                    st.session_state.company_context = {}
                
                # Store the scraped content in session state
                st.session_state.company_context["website_content"] = combined_content
                st.session_state.company_context["url"] = company_url
                # Add target geography to the context
                if target_geography:
                    logger.info(f"Setting user-specified target geography: '{target_geography}'")
                    st.session_state.company_context["target_geography"] = target_geography
                
                # Now run the company context analysis with the target geography
                logger.info(f"Calling analyze_company_context with target_geography: '{target_geography}'")
                try:
                    company_context = asyncio.run(analyze_company_context(
                        company_url, 
                        model_name=FIXED_MODEL,
                        target_geography=target_geography
                    ))
                    
                    # Update progress
                    progress_bar.progress(1.0)  # 100% complete
                    status_text.text("Scraping and analysis complete!")
                    
                    # Check if we got a valid dictionary back
                    if not isinstance(company_context, dict):
                        logger.error(f"Invalid company_context returned: {type(company_context)}")
                        st.error(f"Error analyzing company: Invalid response format")
                        # Use what we have in session state as a fallback
                        company_context = st.session_state.company_context
                    
                    # Check if there's an error field in the context
                    if "error" in company_context:
                        logger.warning(f"Company analysis returned with error: {company_context.get('error')}")
                        st.warning(f"Company analysis completed with warning: {company_context.get('description')}")
                    
                    if company_context:
                        # Update session state
                        st.session_state.company_name = company_context.get("name", "")
                        
                        # Copy over the website_content we just saved
                        website_content = st.session_state.company_context.get("website_content", "")
                        
                        # Preserve the user-specified target geography if provided
                        user_geography = st.session_state.company_context.get("target_geography", "")
                        logger.info(f"Current target geography in session state: '{user_geography}'")
                        
                        # Update with the analyzed context while preserving website content
                        st.session_state.company_context = company_context
                        st.session_state.company_context["website_content"] = website_content
                        
                        # Override the detected geography with user-specified if available
                        if user_geography:
                            logger.info(f"Preserving user-specified target geography: '{user_geography}'")
                            st.session_state.company_context["target_geography"] = user_geography
                            
                        logger.info(f"Final target geography in context: '{st.session_state.company_context.get('target_geography', 'Unknown')}'")
                        
                        # Mark context as not approved yet
                        st.session_state.context_approved = False
                        
                        # Show success message
                        st.success("Successfully scraped and analyzed your company website!")
                        st.rerun()  # Rerun to show the approval section
                    else:
                        st.error("Unable to analyze company website. Please try again.")
                except Exception as e:
                    logger.error(f"Error in company analysis: {str(e)}")
                    logger.exception("Detailed exception:")
                    st.error(f"Error analyzing company: {str(e)}")
                    # We can still proceed with what we have from the scraping
                    if "company_context" in st.session_state and st.session_state.company_context:
                        st.info("However, we've saved your scraped website content. You can edit the context manually.")
                        st.session_state.context_approved = False
                        st.rerun()
            except Exception as e:
                logger.error(f"Error in scrape and analyze process: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up progress indicators
                try:
                    progress_bar.empty()
                    status_text.empty()
                except:
                    pass
    
    # Step 2: Review and approve the generated context
    if has_scraped_content and has_generated_context and not context_approved:
        st.markdown("### Step 2: Review & Approve Generated Context")
        st.info("Please review the generated company context below. You can edit it before approving.")
        
        company_context = st.session_state.get("company_context", {})
        company_name = st.text_input(
            "Company Name",
            value=company_context.get("name", ""),
            key="review_company_name"
        )
        
        company_description = st.text_area(
            "Company Description",
            value=company_context.get("description", ""),
            height=150,
            key="review_company_description"
        )
        
        # Add target geography field to the review section
        company_geography = st.text_input(
            "Target Geography",
            value=company_context.get("target_geography", ""),
            placeholder="e.g., North America, EMEA, APAC, Global, etc.",
            key="review_company_geography",
            help="Specify your target market or where you're selling - this affects how problems are framed"
        )
        
        # Display scraped content preview - use a container instead of an expander
        st.markdown("#### Website Content Preview")
        show_content = st.checkbox("Show scraped content", value=False)
        
        if show_content:
            website_content = company_context.get("website_content", "")
            content_preview = website_content[:2000] + "..." if len(website_content) > 2000 else website_content
            st.text_area(
                "Scraped Content",
                value=content_preview,
                height=200,
                disabled=True
            )
            if website_content:
                content_length = len(website_content)
                approx_pages = max(1, content_length // 5000)  # Rough estimate of page count
                st.info(f"📄 {approx_pages} pages of content scraped ({content_length} characters)")
        
        # Approval button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Approve Context", key="approve_context_btn", use_container_width=True):
                # Update the context with edited values
                st.session_state.company_context["name"] = company_name
                st.session_state.company_context["description"] = company_description
                st.session_state.company_context["target_geography"] = company_geography
                
                # Mark as approved
                st.session_state.context_approved = True
                st.success("Company context approved and ready for personality analysis!")
                st.rerun()
        
        with col2:
            if st.button("Start Over", key="reset_context_btn", use_container_width=True):
                # Reset the context
                if "company_context" in st.session_state:
                    del st.session_state.company_context
                st.session_state.context_approved = False
                st.info("Context reset. Please start the process again.")
                st.rerun()
    
    # Step 3: Display approved context
    elif context_approved:
        st.markdown("### Approved Company Context")
        st.success("This approved context will be used for all personality analyses.")
        
        company_context = st.session_state.get("company_context", {})
        
        st.markdown(f"**Company Name**: {company_context.get('name', '')}")
        st.markdown("**Company Description**:")
        st.markdown(f"_{company_context.get('description', '')}_")
        st.markdown(f"**Target Geography**: {company_context.get('target_geography', 'Global')}")
        
        # Show scraped content metrics if available
        if "website_content" in company_context and company_context["website_content"]:
            content_length = len(company_context["website_content"])
            approx_pages = max(1, content_length // 5000)
            st.info(f"📄 Enhanced with website content from {approx_pages} pages ({content_length} characters)")
        
            # Show content preview with checkbox rather than expander
            show_content = st.checkbox("Show website content", value=False)
            if show_content:
                website_content = company_context["website_content"]
                content_preview = website_content[:2000] + "..." if len(website_content) > 2000 else website_content
                st.text_area(
                    "Website Content",
                    value=content_preview,
                    height=200,
                    disabled=True
                )
        
        # Option to edit again
        if st.button("Edit Context", key="edit_context_btn"):
            st.session_state.context_approved = False
            st.rerun()
    
    # Display manual entry option if no context exists yet
    elif not has_scraped_content and not has_generated_context:
        st.markdown("### Or Enter Context Manually")
        manual_company_name = st.text_input("Company Name", value="", key="manual_company_name")
        manual_company_description = st.text_area(
            "Company Description", 
            value="",
            height=150,
            placeholder="Describe your company, products/services, target market and value proposition...",
            key="manual_company_description"
        )
        
        # Add target geography field to manual entry
        manual_target_geography = st.text_input(
            "Target Geography", 
            value="",
            placeholder="e.g., North America, EMEA, APAC, Global, etc.",
            help="Specify your target market or where you're selling - problems differ by region",
            key="manual_target_geography"
        )
        
        if st.button("Save Manual Context", key="save_manual_context_btn"):
            if manual_company_description.strip():
                # Save to session state
                if "company_context" not in st.session_state:
                    st.session_state.company_context = {}
                
                st.session_state.company_context["name"] = manual_company_name
                st.session_state.company_context["description"] = manual_company_description
                st.session_state.company_context["target_geography"] = manual_target_geography
                
                # Mark as approved
                st.session_state.context_approved = True
                st.success("Manual company context saved and approved!")
                st.rerun()
            else:
                st.error("Please enter a company description.")

# File upload section
uploaded_file = st.file_uploader("Step 2. Upload your contact CSV file", type="csv")

if uploaded_file is not None:
    # Load data if not already loaded
    if st.session_state.df is None:
        df = load_csv_data(uploaded_file)
        if df is not None:
            # Check if the dataframe has separate first and last name columns
            if has_name_components(df):
                st.info("We detected separate first and last name columns in your CSV. You'll be able to combine them in the next step.")
                st.session_state.has_combined_names = True
            
            st.session_state.df = df
            st.session_state.processing_complete = False
            st.session_state.personality_analysis_complete = False
            # Automatically show the mapping dialog when file is uploaded
            st.session_state.show_mapping_dialog = True
    
    # Use the loaded dataframe
    df = st.session_state.df
    
    if df is not None:
        # Display data in data_editor
        st.subheader("Contact Data")
        
        # Guess website column if not already selected
        if st.session_state.website_column is None:
            website_col = guess_website_column(df)
            st.session_state.website_column = website_col
        else:
            website_col = st.session_state.website_column
        
        # Display the dataframe with data_editor
        column_config = {
            website_col: st.column_config.TextColumn("Website URL"),
            "website_content": st.column_config.TextColumn("Website Content", width="large"),
            "website_links": st.column_config.TextColumn("Website Links", width="large"),
        }
        
        # Add configuration for the full_name column if it exists
        if "full_name" in df.columns:
            column_config["full_name"] = st.column_config.TextColumn("Full Name")
        
        # Add personality columns to config if analysis has been done
        if st.session_state.personality_analysis_complete:
            column_config.update({
                "personality_analysis": st.column_config.TextColumn("Personality Analysis", width="large"),
                "conversation_style": st.column_config.TextColumn("Conversation Style"),
                "professional_interests": st.column_config.TextColumn("Professional Interests"),
                "company_context": st.column_config.TextColumn("Company Context", help="The company context used for this contact's analysis")
            })
        
        # If we have scraped data, use that instead
        if st.session_state.scraped_df is not None:
            df = st.session_state.scraped_df
            st.session_state.df = df
            st.session_state.scraped_df = None
            st.session_state.processing_complete = True
        
        # Determine which columns should be disabled in the editor
        disabled_columns = ["website_content", "website_links"]
        
        if st.session_state.personality_analysis_complete:
            disabled_columns.extend(["personality_analysis", "conversation_style", "professional_interests", "company_context"])
        
        if st.session_state.has_combined_names and "full_name" in df.columns:
            disabled_columns.append("full_name")
        
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            height=400,
            column_config=column_config,
            disabled=disabled_columns
        )
        
        # Update the session state with edited data
        st.session_state.df = edited_df
        
        # Process websites section
        st.subheader("Process Websites")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show the mapping dialog if triggered
            if st.session_state.show_mapping_dialog:
                website_mapping_dialog(df)
                # Dialog will handle the processing and storing results
                # The dialog sets st.session_state.scraped_df when complete
                # And performs a st.rerun() to close the dialog
                st.session_state.show_mapping_dialog = False
            
            # Simple button to start website scraping
            if not st.session_state.processing_complete:
                if st.button("Start Website Scraping", key="start_scraping_button"):
                    st.session_state.show_mapping_dialog = True
                    st.rerun()
            
            # Button to remove rows with failed website scraping
            if st.session_state.processing_complete and "website_content" in st.session_state.df.columns:
                if st.button("Remove Rows with Failed Website Scraping"):
                    # Filter out rows where website_content indicates an error or is empty
                    # Look for multiple error patterns and empty content cases
                    mask = ~(
                        # Existing error patterns
                        st.session_state.df["website_content"].str.startswith("Error:", na=False) |
                        st.session_state.df["website_content"].str.startswith("No URL provided", na=False) |
                        st.session_state.df["website_content"].str.startswith("Invalid URL", na=False) |
                        st.session_state.df["website_content"].str.contains("failed to scrape", case=False, na=False) |
                        st.session_state.df["website_content"].str.contains("timed out", case=False, na=False) |
                        st.session_state.df["website_content"].str.contains("access denied", case=False, na=False) |
                        # General error detection
                        st.session_state.df["website_content"].str.contains("error", case=False, na=False) |
                        # Additional checks for empty content
                        st.session_state.df["website_content"].isna() |  # Null/NaN values
                        (st.session_state.df["website_content"] == "") |  # Empty strings
                        st.session_state.df["website_content"].str.isspace().fillna(False) |  # Whitespace only
                        (st.session_state.df["website_content"].str.len() < 50)  # Very short content (likely errors)
                    )
                    
                    total_before = len(st.session_state.df)
                    st.session_state.df = st.session_state.df[mask]
                    rows_removed = total_before - len(st.session_state.df)
                    
                    if rows_removed > 0:
                        st.success(f"Removed {rows_removed} rows with failed or empty website scraping.")
                    else:
                        st.info("No rows with failed website scraping to remove.")
                    st.rerun()
        
        # Personality Analysis section - only enabled after website processing
        with col2:
            # Add a slider to control how many rows to analyze when processing is complete
            if st.session_state.processing_complete:
                total_rows = len(df)
                max_rows_to_analyze = st.slider(
                    "Number of rows to analyze:",
                    min_value=1,
                    max_value=total_rows,
                    value=min(total_rows, 10),  # Default to 10 rows or total if less
                    step=1,
                    key="max_rows_slider"
                )
                st.info(f"Will analyze {max_rows_to_analyze} out of {total_rows} rows. Adjust slider to control costs and processing time.")
            
            analyze_button = st.button(
                "Analyze Personalities", 
                disabled=not st.session_state.processing_complete,
                key="analyze_personalities_button"
            )
            
            if analyze_button:
                # Get the number of rows to analyze from the slider
                max_rows = st.session_state.get("max_rows_slider", len(df))
                
                # Limit the dataframe to the selected number of rows
                analysis_df = df.head(max_rows)
                
                # Count rows with website content for informational purposes
                rows_with_content = len(analysis_df[analysis_df['website_content'].notna() & (analysis_df['website_content'] != "")])
                
                # Display info about the analysis
                st.info(f"Analyzing {max_rows} contacts using Claude via OpenRouter. {rows_with_content} contacts have website content. This may take a few minutes.")
                
                # Check for API keys
                openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                tavily_key = os.environ.get("TAVILY_API_KEY")
                
                if not openrouter_key or not tavily_key:
                    logger.error(f"API keys missing: OpenRouter={'Missing' if not openrouter_key else 'Present'}, Tavily={'Missing' if not tavily_key else 'Present'}")
                    st.error("Please set your OpenRouter and Tavily API keys in the API Keys Configuration section.")
                # Ensure name column is set
                elif not st.session_state.name_column:
                    st.error("Please select a name column in the Website Mapping Dialog first.")
                else:
                    logger.debug("API keys are present, starting personality analysis with OpenRouter")
                    
                    # Get company context if available
                    company_context = st.session_state.get("company_context", None)
                    context_approved = st.session_state.get("context_approved", False)
                    
                    if company_context:
                        # Check if context is approved
                        if not context_approved:
                            st.warning("⚠️ Your company context has not been approved yet. Please go to the Company Context Configuration section and approve it first.")
                            # Disable the analysis button
                            st.stop()
                            
                        # Check if we have website content
                        has_website_content = "website_content" in company_context and company_context["website_content"]
                        if has_website_content:
                            logger.info(f"Using company context with website content for analysis: {company_context['name']}")
                            content_length = len(company_context["website_content"])
                            approx_pages = max(1, content_length // 5000)  # Rough estimate of page count
                            st.info(f"Using enhanced company context with website content (~{approx_pages} pages) for more personalized analysis.")
                        else:
                            logger.info(f"Using company context for analysis: {company_context['name']}")
                    else:
                        logger.info("No company context provided for analysis")
                        st.warning("No company context provided. For more personalized results, add your company information in the Company Context Configuration section.")
                    
                    # Run the personality analysis asynchronously with company context
                    result_df = asyncio.run(run_personality_analysis(analysis_df, company_context=company_context))
                    
                    # Merge the results back into the full dataframe
                    # First, create a copy of the full dataframe
                    full_df = df.copy()
                    
                    # Update only the rows that were analyzed
                    for idx in result_df.index:
                        if idx in full_df.index:
                            full_df.loc[idx, 'personality_analysis'] = result_df.loc[idx, 'personality_analysis']
                            full_df.loc[idx, 'conversation_style'] = result_df.loc[idx, 'conversation_style']
                            full_df.loc[idx, 'professional_interests'] = result_df.loc[idx, 'professional_interests']
                            if 'company_context' in result_df.columns:
                                full_df.loc[idx, 'company_context'] = result_df.loc[idx, 'company_context']
                    
                    # Count how many rows were actually analyzed (have non-empty personality_analysis)
                    analyzed_count = len(result_df[result_df['personality_analysis'].notna() & (result_df['personality_analysis'] != "")])
                    
                    # Store the result in session state
                    st.session_state.df = full_df
                    st.session_state.personality_analysis_complete = True
                    st.success(f"Successfully analyzed personalities for {analyzed_count} contacts!")
                    st.rerun()
        
        # Download section
        st.subheader("Download Data")
        st.markdown(f'<a href="{get_download_link(df)}" download="enriched_contacts.csv" class="button">Download CSV</a>', unsafe_allow_html=True)

# Call the main function if running this file directly
if __name__ == "__main__":
    # All content now at module level, so we don't need to call anything here
    pass
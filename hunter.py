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
# Import the simple company analyzer instead of the workflow
from ai_utils.simple_company_analyzer import analyze_company_context
# Import the company scraper
from utils.company_scraper import company_scraper_dialog, CompanyScraper
# Import helper functions
from utils.data_helpers import load_csv_data, get_download_link, has_name_components


# Import URL sanitizer
from utils.url_sanitizer import sanitize_dataframe_urls, sanitize_url

import os
from dotenv import load_dotenv
import requests
from utils.logging_config import configure_langsmith_tracing

# Load environment variables from .env file
load_dotenv()

# Configure LangSmith tracing if API key is available
configure_langsmith_tracing()



# Helper function to keep session state variables permanent
def keep_permanent_session_vars():
    """Prevents Streamlit from clearing session state variables with p_ prefix"""
    for key in list(st.session_state.keys()):
        if key.startswith("p_"):
            st.session_state[key] = st.session_state[key]

# Function to show the main page UI
def show_main_page():
    """Contains all the UI components for the main page"""
    # Initialize permanent session state variables
    if "p_df" not in st.session_state:
        st.session_state.p_df = None
    
    # Also add p_show_mapping_dialog to preserve dialog state between page navigation
    if "p_show_mapping_dialog" not in st.session_state:
        st.session_state.p_show_mapping_dialog = False
    
    # ==========================
    # Core Application Settings
    # ==========================
    # Only set page config when running this file directly (not through navigation)
    # This prevents conflicts with st.navigation
    
    # Initialize session state variables if not already set
    if "df" not in st.session_state:
        st.session_state.df = None
    
    # Using the p_ prefix convention for permanent session state variables
    if "p_df" not in st.session_state:
        st.session_state.p_df = None
    
    if "company_context" not in st.session_state:
        st.session_state.company_context = {}
    
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
        # If p_show_mapping_dialog is True, sync this value
        if "p_show_mapping_dialog" in st.session_state and st.session_state.p_show_mapping_dialog:
            st.session_state.show_mapping_dialog = True
    if "has_combined_names" not in st.session_state:
        st.session_state.has_combined_names = False
        
 
# ==========================
# Helper Functions
# ==========================
async def run_personality_analysis(df, company_context=None):
    """Run personality analysis in async context"""
    # Ensure id column exists for robust matching
    df = df.copy()
    if "id" not in df.columns:
        import uuid
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    name_column = st.session_state.name_column
    # If we have combined first and last names, make sure the full_name column exists
    if st.session_state.get("has_combined_names", False) and "full_name" in df.columns:
        pass
    elif st.session_state.get("website_mapping", {}).get("has_separate_names", False):
        first_name_column = st.session_state.website_mapping.get("first_name_column")
        last_name_column = st.session_state.website_mapping.get("last_name_column")
        if first_name_column and last_name_column and first_name_column in df.columns and last_name_column in df.columns:
            df['full_name'] = df.apply(
                lambda row: f"{row[first_name_column]} {row[last_name_column]}".strip(), 
                axis=1
            )
            name_column = "full_name"
            st.session_state.name_column = name_column
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        total_rows = len(df)
        status_text.text(f"Analyzing {total_rows} contacts...")
        ai_mode = st.session_state.get("ai_mode", "Proxy")
        openai_api_key = st.session_state.get("openai_api_key", "")
        proxy_api_key = st.session_state.get("proxy_api_key", "")
        proxy_base_url = st.session_state.get("proxy_base_url", "https://llm.data-qa.justworks.com")
        result_df = await analyze_personality(
            df=df, 
            company_context=company_context,
            ai_mode=ai_mode,
            openai_api_key=openai_api_key,
            proxy_api_key=proxy_api_key,
            proxy_base_url=proxy_base_url
        )
        progress_bar.progress(1.0)
        status_text.text(f"Analysis complete for {total_rows} contacts")
        if company_context and "company_context" not in result_df.columns:
            company_name = company_context.get('name', 'Unknown')
            company_desc = company_context.get('description', '')
            result_df['company_context'] = f"{company_name}: {company_desc}"
        return result_df
    except Exception as e:
        st.error(f"An error occurred during personality analysis: {str(e)}")
        return df
    finally:
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

# Enhanced sidebar title (moved to streamlit_app.py as it's common)
# st.sidebar.markdown("""
# <h1 style="text-align: center; color: #FF5722; font-weight: bold; margin-bottom: 5px;">HUNTER</h1>
# <p style="text-align: center; color: #757575; font-size: 14px; margin-top: 0;">AI-Powered Sales Intelligence</p>
# """, unsafe_allow_html=True)

# Use native Streamlit components for the title and subtitle





# Add Company Context section
with st.expander("Step 1. Setup Company Context", expanded=False):
    
    # Determine current workflow state
    has_url_entered = "url" in st.session_state.get("company_context", {})
    has_generated_context = bool(st.session_state.get("company_context", {}).get("description", ""))
    context_approved = st.session_state.get("context_approved", False)
    
    # Display current state/progress
    col_status = st.columns(3)
    with col_status[0]:
        if has_url_entered:
            st.success("✅ 1. URL entered")
        else:
            st.info("1. Enter company URL")
    
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
    
    # Step 1: Website Input & Analysis Process
    st.markdown("### Step 1: Enter Company Website")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_url = st.text_input(
            "Company Website URL",
            value=st.session_state.get("company_url", ""),
            placeholder="https://example.com",
            help="Enter your company website URL to scrape and analyze"
        )
        
        # Add company name field
        company_name = st.text_input(
            "Company Name",
            value=st.session_state.get("company_name", ""),
            placeholder="e.g., Acme Corporation",
            help="Enter the company name for more accurate analysis"
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
            "Analyze", 
            key="scrape_analyze_btn_main",
            use_container_width=True,
            disabled=not company_url or not company_name
        )
        
        # Store URL, company name, and geography in session state
        if company_url:
            st.session_state.company_url = company_url
        if company_name:
            st.session_state.company_name = company_name
        if target_geography:
            st.session_state.target_geography = target_geography
        
    # Handle the analysis process (no scraping)
    if combined_process_button:
        with st.spinner("Analyzing company..."):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress to indicate start
                progress_bar.progress(0.1)
                status_text.text("Starting company analysis...")
                
                # Initialize company context in session state
                if "company_context" not in st.session_state:
                    st.session_state.company_context = {}
                
                # Update company URL in context
                st.session_state.company_context["url"] = company_url
                
                # Add target geography to the context
                if target_geography:
                    st.session_state.company_context["target_geography"] = target_geography
                
                # Update progress
                progress_bar.progress(0.3)
                status_text.text("Running company analysis...")
                
                # Run analysis on the company website
                with st.spinner("Analyzing company..."):
                    try:
                        # Get API/mode parameters from session state
                        ai_mode = st.session_state.get("ai_mode", "Proxy")
                        openai_api_key = st.session_state.get("openai_api_key", "")
                        proxy_api_key = st.session_state.get("proxy_api_key", "")
                        proxy_base_url = st.session_state.get("proxy_base_url", "https://llm.data-qa.justworks.com")
                        
                        # Analyze the company using the provided information
                        company_context = asyncio.run(analyze_company_context(
                            company_url=company_url,
                            company_name=company_name,
                     
                            target_geography=target_geography,
                            ai_mode=ai_mode,
                            openai_api_key=openai_api_key,
                            proxy_api_key=proxy_api_key,
                            proxy_base_url=proxy_base_url,
                        ))
                        
                        # Store the context in session state
                        # The company_context dict contains keys like 'name', 'description', 'url', 'target_geography', etc.
                        st.session_state.company_context = company_context
                        st.session_state.context_generated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing company: {str(e)}")
                        if "OpenAI API key" in str(e):
                            st.info("Please provide a valid OpenAI API key in the sidebar.")
                        st.session_state.company_context = {
                            "name": "Error",
                            "url": company_url,
                            "description": f"Error during analysis: {str(e)}",
                            "error": str(e)
                        }
            except Exception as e:
                st.error(f"Error in analysis process: {str(e)}")
            finally:
                # Clean up progress indicators
                try:
                    progress_bar.empty()
                    status_text.empty()
                except:
                    pass
    
    # Step 2: Review and approve the generated context
    if has_url_entered and has_generated_context and not context_approved:
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
            height=600,
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
        
        # Add feedback field for context adjustments
        context_feedback = st.text_area(
            "Look off? What needs to be adjusted?",
            value="",
            height=100,
            key="context_feedback",
            help="Provide specific feedback on what needs to be adjusted in the company context. Our AI will refine it based on your input."
        )
        
        # Add button to adjust context based on feedback
        if context_feedback.strip():
            if st.button("Adjust Context Based on Feedback", key="adjust_context_btn"):
                with st.spinner("Adjusting company context based on your feedback..."):
                    try:
                        # Import the context adjuster
                        from utils.context_adjuster import adjust_company_context
                        
                        # Call the adjuster with the current context and feedback
                        adjusted_context = adjust_company_context(
                            company_context,
                            context_feedback
                        )
                        
                        # Update the session state with adjusted context
                        if adjusted_context:
                            st.session_state.company_context = adjusted_context
                            st.success("Context adjusted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to adjust context. Please try again or adjust manually.")
                    except Exception as e:
                        st.error(f"Error adjusting context: {str(e)}")
        
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
        
        # Option to edit again
        if st.button("Edit Context", key="edit_context_btn"):
            st.session_state.context_approved = False
            st.rerun()
    
    # Display manual entry option if no context exists yet
    elif not has_url_entered and not has_generated_context:
        st.markdown("### Or Enter Context Manually")
        manual_company_name = st.text_input("Company Name", value="", key="manual_company_name")
        manual_company_description = st.text_area(
            "Company Description", 
            value="",
            height=600,
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

# Initialize the df variable in the session state if it doesn't exist yet
if "df" not in st.session_state:
    st.session_state.df = None

# Initialize the website_column variable in the session state if it doesn't exist yet
if "website_column" not in st.session_state:
    st.session_state.website_column = None

# Initialize other commonly used session state variables
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
if "company_context" not in st.session_state:
    st.session_state.company_context = {}

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
            
            # Guess the website column
            website_col = guess_website_column(df)
            
            # Sanitize URLs in the website column
            if website_col in df.columns:
                df, cleaned_count = sanitize_dataframe_urls(df, website_col)
                if cleaned_count > 0:
                    st.success(f"✅ Sanitized {cleaned_count} URLs in your data to ensure proper formatting.")
            
            st.session_state.df = df
            # Also store in permanent session state with p_ prefix
            st.session_state.p_df = df.copy()
            st.session_state.processing_complete = False
            st.session_state.personality_analysis_complete = False
            # Automatically show the mapping dialog when file is uploaded
            st.session_state.show_mapping_dialog = True
            # Also set the permanent version
            st.session_state.p_show_mapping_dialog = True
    
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
        
        # Restore permanent state if available
        if "p_df" in st.session_state and st.session_state.p_df is not None:
            st.session_state.df = st.session_state.p_df
        if "p_processing_complete" in st.session_state:
            st.session_state.processing_complete = st.session_state.p_processing_complete
        
        if st.session_state.scraped_df is not None:
            df = st.session_state.scraped_df
            st.session_state.df = df
            st.session_state.scraped_df = None
            st.session_state.processing_complete = True
            st.session_state.p_df = df.copy()
            st.session_state.p_processing_complete = True
        
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
        st.session_state.p_df = edited_df.copy()
        
        # Process websites section
        st.subheader("Process Websites")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # First check the regular dialog flag, then check the permanent one as fallback
            # This restores the original behavior when returning from other pages
            if st.session_state.show_mapping_dialog or st.session_state.p_show_mapping_dialog:
                # Reset both flags before showing dialog to avoid infinite loops
                st.session_state.show_mapping_dialog = False
                st.session_state.p_show_mapping_dialog = False
                
                website_mapping_dialog(df)
                # Dialog will handle the processing and storing results
                # The dialog sets st.session_state.scraped_df when complete
                # And performs a st.rerun() to close the dialog
            
            # Simple button to start website scraping
            if not st.session_state.processing_complete:
                if st.button("Start Website Scraping", key="start_scraping_button"):
                    st.session_state.show_mapping_dialog = True
                    st.session_state.p_show_mapping_dialog = True
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
            # Debug output for troubleshooting
            st.write("processing_complete:", st.session_state.get("processing_complete"))
            st.write("df shape:", None if st.session_state.df is None else st.session_state.df.shape)
            # Add user feedback if button is disabled
            if not st.session_state.processing_complete:
                st.warning("Website scraping must be completed before running personality analysis.")
            elif st.session_state.df is None or len(st.session_state.df) == 0:
                st.warning("No data available for analysis. Please upload and process your contact file.")
            # Add a slider to control how many rows to analyze when processing is complete
            if st.session_state.processing_complete:
                total_rows = len(df)
                
                # Handle case when there are no rows or only one row
                if total_rows <= 0:
                    st.warning("No rows available to analyze. Please upload data with valid website URLs.")
                elif total_rows == 1:
                    max_rows_to_analyze = 1
                    st.info("Only one row available to analyze.")
                else:
                    max_rows_to_analyze = st.slider(
                        "Number of rows to analyze:",
                        min_value=1,
                        max_value=total_rows,
                        value=min(total_rows, 10),
                        step=1,
                        key="max_rows_slider"
                    )
                    st.info(f"Will analyze {max_rows_to_analyze} out of {total_rows} rows. Adjust slider to control costs and processing time.")
            
            analyze_button = st.button(
                "Analyze Personalities", 
                disabled=not st.session_state.processing_complete or len(df) <= 0,  # Disable if no rows
                key="analyze_personalities_button"
            )
            
            if analyze_button:
                # Get the number of rows to analyze from the slider, ensure it's at least 1
                max_rows = min(st.session_state.get("max_rows_slider", 1), max(1, len(df)))
                analysis_df = df.head(max_rows)
                rows_with_content = len(analysis_df[analysis_df['website_content'].notna() & (analysis_df['website_content'] != "")])
                st.info(f"Analyzing {max_rows} contacts using Claude via OpenRouter. {rows_with_content} contacts have website content.")
                st.warning(f"This will take a few minutes. You can navigate to another page, but keep this one open until the AI finsihes.")
                company_context = st.session_state.get("company_context", None)
                context_approved = st.session_state.get("context_approved", False)
                if not context_approved:
                    st.warning("Please approve your company context in Step 1 before analyzing personalities.")
                    st.info("If you've already set up your company context, click 'Approve' to continue.")
                else:
                    import sys
                    import asyncio
                    try:
                        with st.spinner(f"Analyzing personalities for {max_rows} contacts..."):
                            # Use correct async pattern for Streamlit
                            if sys.version_info >= (3, 7):
                                try:
                                    loop = asyncio.get_running_loop()
                                    # If we're in an event loop, create a task and await it
                                    task = asyncio.create_task(run_personality_analysis(
                                        df=analysis_df,
                                        company_context=company_context
                                    ))
                                    full_df = asyncio.run_coroutine_threadsafe(task, loop).result()
                                except RuntimeError:
                                    # No running event loop, safe to use asyncio.run
                                    full_df = asyncio.run(run_personality_analysis(
                                        df=analysis_df,
                                        company_context=company_context
                                    ))
                            else:
                                # Fallback for older Python
                                full_df = asyncio.run(run_personality_analysis(
                                    df=analysis_df,
                                    company_context=company_context
                                ))
                            analyzed_count = sum(1 for _, row in full_df.iterrows() 
                                               if "personality_analysis" in full_df.columns 
                                               and pd.notna(row["personality_analysis"]) 
                                               and row["personality_analysis"].strip() != ""
                                               and not row["personality_analysis"].strip().startswith("Error"))
                            st.session_state.df = full_df
                            st.session_state.p_df = full_df.copy()
                            st.session_state.personality_analysis_complete = True
                            st.success(f"Successfully analyzed personalities for {analyzed_count} contacts!")
                            st.success("Proceed to the Spear page to write emails.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error during personality analysis: {e}")
                        import traceback
                        st.exception(traceback.format_exc())
        
        
        # Download section
        st.subheader("Download Data")
        st.markdown(f'<a href="{get_download_link(df)}" download="enriched_contacts.csv" class="button">Download CSV</a>', unsafe_allow_html=True)

# Add a button to navigate to the SPEAR page
st.markdown("---")
st.subheader("Next Steps")
if st.button("Generate emails for contacts", use_container_width=True):
    # Navigate to the SPEAR page
    import streamlit as st
    st.switch_page("spear.py")

# Keep our permanent session vars at the top of the app
keep_permanent_session_vars()

# Call the main function if running this file directly
if __name__ == "__main__":
    # All content now at module level, so we don't need to call anything here
    pass
else:
    # Only run the UI when loaded as a page through navigation, not when imported
    show_main_page()
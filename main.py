from fastapi import utils
import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import base64
import asyncio
from io import StringIO
from utils.scraper import process_websites_parallel, website_mapping_dialog, guess_website_column
from ai_utils.personality_analyzer import analyze_personality, PersonalityAnalyzer
import os
import logging
from dotenv import load_dotenv

# Set up logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app_debug.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables from .env file")
load_dotenv()

# Log environment variable status (without exposing full keys)
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")
logger.info(f"ANTHROPIC_API_KEY loaded: {'Yes' if anthropic_key else 'No'}")
if anthropic_key:
    logger.info(f"ANTHROPIC_API_KEY starts with: {anthropic_key[:10]}...")
    logger.info(f"ANTHROPIC_API_KEY length: {len(anthropic_key)}")
logger.info(f"TAVILY_API_KEY loaded: {'Yes' if tavily_key else 'No'}")

# ==========================
# Core Application Settings
# ==========================
st.set_page_config(
    page_title="Contact Data Enrichment Tool",
    page_icon="📇",
    layout="wide"  # Use wide layout for better data display
)

# ==========================
# Helper Functions
# ==========================
def load_csv_data(uploaded_file):
    """Load CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        # Add empty website_content and website_links columns right away
        if 'website_content' not in df.columns:
            df['website_content'] = ""
        if 'website_links' not in df.columns:
            df['website_links'] = ""
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def get_download_link(df):
    """Generate a download link for the processed dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

async def test_api_keys():
    """Test if the API keys are valid"""
    try:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        
        if not anthropic_key:
            return False, "Anthropic API key is missing"
        
        if not tavily_key:
            return False, "Tavily API key is missing"
        
        # Test the Anthropic API key
        logger.debug("Creating PersonalityAnalyzer to test API keys")
        analyzer = PersonalityAnalyzer(
            anthropic_api_key=anthropic_key,
            tavily_api_key=tavily_key
        )
        
        # Test the API key
        key_valid = await analyzer.test_anthropic_api_key()
        
        if not key_valid:
            return False, "Anthropic API key validation failed"
        
        return True, "API keys are valid"
    except Exception as e:
        logger.error(f"Error testing API keys: {str(e)}")
        return False, f"Error testing API keys: {str(e)}"

# Run async functions in Streamlit
async def run_personality_analysis(df):
    """Run personality analysis in async context"""
    with st.spinner("Analyzing personalities based on website content..."):
        # Debug check for API keys before analysis
        logger.debug("Checking API keys before personality analysis")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        
        if not anthropic_key:
            logger.error("ANTHROPIC_API_KEY not found in environment")
            st.error("Anthropic API key not found. Please set it in the API Keys Configuration section.")
            return df
        
        if not tavily_key:
            logger.error("TAVILY_API_KEY not found in environment")
            st.error("Tavily API key not found. Please set it in the API Keys Configuration section.")
            return df
            
        logger.debug(f"Starting personality analysis with API keys (first chars): Anthropic={anthropic_key[:8]}..., Tavily={tavily_key[:8]}...")
        result_df = await analyze_personality(df)
        return result_df

# ==========================
# Main Application
# ==========================
def main():
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
        
    st.title("Contact Data Enrichment Tool")
    
    # API Keys section
    with st.expander("API Keys Configuration"):
        anthropic_api_key = st.text_input("Anthropic API Key", 
                                        value=os.environ.get("ANTHROPIC_API_KEY", ""), 
                                        type="password",
                                        help="Required for personality analysis")
        
        tavily_api_key = st.text_input("Tavily API Key", 
                                     value=os.environ.get("TAVILY_API_KEY", ""), 
                                     type="password",
                                     help="Required for web search in personality analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save API Keys"):
                # Remove any quotes or extra whitespace
                anthropic_api_key = anthropic_api_key.strip().replace('"', '')
                tavily_api_key = tavily_api_key.strip().replace('"', '')
                
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                os.environ["TAVILY_API_KEY"] = tavily_api_key
                
                # Log key information (safely)
                if anthropic_api_key:
                    key_preview = anthropic_api_key[:8] + "..." if len(anthropic_api_key) > 8 else "[empty]"
                    logger.debug(f"Saved Anthropic API key (first chars): {key_preview}")
                    logger.debug(f"Anthropic API key length: {len(anthropic_api_key)}")
                
                st.success("API keys saved for this session!")
        
        with col2:
            if st.button("Test API Keys"):
                with st.spinner("Testing API keys..."):
                    valid, message = asyncio.run(test_api_keys())
                    if valid:
                        st.success(message)
                    else:
                        st.error(message)
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your contact CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data if not already loaded
        if st.session_state.df is None:
            df = load_csv_data(uploaded_file)
            if df is not None:
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
            
            # Allow user to select a different website column
            website_col = st.selectbox(
                "Select website column:", 
                options=df.columns.tolist(),
                index=df.columns.tolist().index(website_col) if website_col in df.columns else 0
            )
            st.session_state.website_column = website_col
            
            # Display the dataframe with data_editor
            column_config = {
                website_col: st.column_config.TextColumn("Website URL"),
                "website_content": st.column_config.TextColumn("Website Content", width="large"),
                "website_links": st.column_config.TextColumn("Website Links", width="large"),
            }
            
            # Add personality columns to config if analysis has been done
            if st.session_state.personality_analysis_complete:
                column_config.update({
                    "personality_analysis": st.column_config.TextColumn("Personality Analysis", width="large"),
                    "conversation_style": st.column_config.TextColumn("Conversation Style"),
                    "professional_interests": st.column_config.TextColumn("Professional Interests")
                })
            
            # If we have scraped data, use that instead
            if st.session_state.scraped_df is not None:
                df = st.session_state.scraped_df
                st.session_state.df = df
                st.session_state.scraped_df = None
                st.session_state.processing_complete = True
            
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                height=400,
                column_config=column_config,
                disabled=["website_content", "website_links", "personality_analysis", "conversation_style", "professional_interests"] 
                         if st.session_state.personality_analysis_complete else 
                         ["website_content", "website_links"]
            )
            
            # Update the session state with edited data
            st.session_state.df = edited_df
            
            # Process websites section
            st.subheader("Process Websites")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Open Website Mapping Dialog"):
                    st.session_state.show_mapping_dialog = True
                    st.rerun()
                
                # Show the mapping dialog if triggered
                if st.session_state.show_mapping_dialog:
                    website_mapping_dialog(df)
                    # Dialog will handle the processing and storing results
                    # The dialog sets st.session_state.scraped_df when complete
                    # And performs a st.rerun() to close the dialog
                    st.session_state.show_mapping_dialog = False
            
            # Personality Analysis section - only enabled after website processing
            with col2:
                analyze_button = st.button(
                    "Analyze Personalities", 
                    disabled=not st.session_state.processing_complete
                )
                
                if analyze_button:
                    # Check for API keys
                    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
                    tavily_key = os.environ.get("TAVILY_API_KEY")
                    
                    if not anthropic_key or not tavily_key:
                        logger.error(f"API keys missing: Anthropic={'Missing' if not anthropic_key else 'Present'}, Tavily={'Missing' if not tavily_key else 'Present'}")
                        st.error("Please set your Anthropic and Tavily API keys in the API Keys Configuration section.")
                    # Ensure name column is set
                    elif not st.session_state.name_column:
                        st.error("Please select a name column in the Website Mapping Dialog first.")
                    else:
                        logger.debug("API keys are present, starting personality analysis")
                        # Run the personality analysis asynchronously
                        result_df = asyncio.run(run_personality_analysis(st.session_state.df))
                        
                        # Store the result in session state
                        st.session_state.df = result_df
                        st.session_state.personality_analysis_complete = True
                        st.success("Successfully analyzed personalities!")
                        st.rerun()
            
            # Download section
            st.subheader("Download Data")
            st.markdown(f'<a href="{get_download_link(df)}" download="enriched_contacts.csv" class="button">Download CSV</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
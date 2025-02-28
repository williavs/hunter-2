import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import base64
from io import StringIO
from utils.scraper import process_websites_parallel

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
        # Add empty website_content column right away
        if 'website_content' not in df.columns:
            df['website_content'] = ""
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def guess_website_column(df):
    """Make a best guess at which column contains website data"""
    website_keywords = ['website', 'company website', 'web', 'url', 'site', 'domain']
    
    best_match = None
    best_score = 0
    
    for col in df.columns:
        for keyword in website_keywords:
            score = fuzz.ratio(col.lower(), keyword)
            if score > best_score:
                best_score = score
                best_match = col
    
    return best_match

def get_download_link(df):
    """Generate a download link for the processed dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

# ==========================
# Main Application
# ==========================
def main():
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "website_column" not in st.session_state:
        st.session_state.website_column = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
        
    st.title("Contact Data Enrichment Tool")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your contact CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data if not already loaded
        if st.session_state.df is None:
            df = load_csv_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.processing_complete = False
        
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
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    website_col: st.column_config.TextColumn("Website URL"),
                    "website_content": st.column_config.TextColumn("Website Content", width="large")
                },
                disabled=["website_content"] if not st.session_state.processing_complete else None
            )
            
            # Update the session state with edited data
            st.session_state.df = edited_df
            
            # Process websites section
            st.subheader("Process Websites")
            
            col1, col2 = st.columns(2)
            with col1:
                max_workers = st.slider("Parallel Workers", min_value=1, max_value=20, value=10, 
                                       help="Number of parallel processes for scraping websites")
            with col2:
                timeout = st.slider("Timeout (seconds)", min_value=5, max_value=30, value=10,
                                  help="Maximum time to wait for website response")
            
            if st.button("Process Websites"):
                with st.spinner("Scraping website content in parallel..."):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    try:
                        # Process websites in parallel
                        result_df = process_websites_parallel(
                            df, 
                            website_col,
                            max_workers=max_workers,
                            timeout=timeout
                        )
                        
                        # Update progress bar to 100%
                        progress_bar.progress(100)
                        
                        # Store the result in session state
                        st.session_state.df = result_df
                        st.session_state.processing_complete = True
                        st.success(f"Successfully processed {len(df)} websites!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing websites: {str(e)}")
            
            # Download section
            st.subheader("Download Data")
            st.markdown(f'<a href="{get_download_link(df)}" download="enriched_contacts.csv" class="button">Download CSV</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
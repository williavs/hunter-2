import requests
import html2text
import pandas as pd
import concurrent.futures
from urllib.parse import urlparse, urljoin
import time
import logging
import urllib3
import streamlit as st
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

# Suppress InsecureRequestWarning messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize HTML to text converter
h2t = html2text.HTML2Text()
h2t.ignore_links = True
h2t.ignore_images = True
h2t.ignore_tables = False
h2t.ignore_emphasis = True
h2t.body_width = 0  # No wrapping

def guess_website_column(df):
    """Make a best guess at which column contains website data using fuzzy matching"""
    website_keywords = ['website', 'company website', 'web', 'url', 'site', 'domain']
    
    best_match = None
    best_score = 0
    
    for col in df.columns:
        for keyword in website_keywords:
            score = fuzz.ratio(col.lower(), keyword)
            if score > best_score:
                best_score = score
                best_match = col
    
    # Fallback to first column if no good match
    if not best_match and len(df.columns) > 0:
        best_match = df.columns[0]
        
    return best_match

@st.dialog("Website Field Mapping", width="large")
def website_mapping_dialog(df):
    """
    Modal dialog for mapping website columns and configuring scraping options.
    
    Args:
        df: DataFrame containing the uploaded contact data
    """
    st.write("Please map required columns and configure scraping options")
    
    # Get all columns from the dataframe
    all_columns = list(df.columns)
    
    # Use fuzzy matching to guess the website column
    guessed_website_col = guess_website_column(df)
    
    # Use fuzzy matching to guess the name column
    guessed_name_col = guess_name_column(df)
    
    # Column selection section
    st.subheader("Column Mapping")
    col1, col2 = st.columns(2)
    
    with col1:
        # Website column selection
        website_column = st.selectbox(
            "Select the column containing website URLs:",
            options=all_columns,
            index=all_columns.index(guessed_website_col) if guessed_website_col in all_columns else 0,
            help="Choose the column that contains website URLs for scraping"
        )
    
    with col2:
        # Name column selection
        name_column = st.selectbox(
            "Select the column containing contact names:",
            options=all_columns,
            index=all_columns.index(guessed_name_col) if guessed_name_col in all_columns else 0,
            help="Choose the column that contains contact names (required for personality analysis)"
        )
    
    # Configuration options
    st.subheader("Scraping Configuration")
    col1, col2 = st.columns(2)
    with col1:
        max_workers = st.slider("Max parallel workers:", 1, 20, 10, 
                              help="Number of websites to process in parallel")
        
    with col2:
        timeout = st.slider("Request timeout (seconds):", 3, 30, 10,
                         help="Time to wait for website response")
    
    # Additional options section
    with st.expander("Advanced Options"):
        extract_links = st.checkbox("Extract links from websites", value=True,
                                  help="Also extract links from each website")
        max_content_length = st.number_input(
            "Maximum content length (characters):",
            min_value=1000,
            max_value=200000,
            value=50000,
            help="Limit on how much content to process from each website"
        )
    
    # Store settings in session state
    if st.button("Start Website Scraping"):
        # Save mapping settings
        st.session_state.website_mapping = {
            "website_column": website_column,
            "name_column": name_column,
            "max_workers": max_workers,
            "timeout": timeout,
            "extract_links": extract_links,
            "max_content_length": max_content_length
        }
        
        # Ensure the name column exists in the dataframe
        if name_column not in df.columns:
            st.error(f"Name column '{name_column}' not found in the data.")
            return
        
        # Trigger scraping
        with st.spinner("Scraping websites. This might take a while..."):
            result_df = process_websites_parallel(
                df, 
                website_column, 
                max_workers=max_workers,
                timeout=timeout,
                max_content_length=max_content_length
            )
            
            # Store the result in session state
            st.session_state.scraped_df = result_df
            st.session_state.name_column = name_column
            
            # Rerun to close the dialog and update the main app
            st.rerun()


def guess_name_column(df):
    """Make a best guess at which column contains name data using fuzzy matching"""
    name_keywords = ['name', 'contact name', 'full name', 'person', 'contact']
    
    best_match = None
    best_score = 0
    
    for col in df.columns:
        for keyword in name_keywords:
            score = fuzz.ratio(col.lower(), keyword)
            if score > best_score:
                best_score = score
                best_match = col
    
    # Fallback to first column if no good match
    if not best_match and len(df.columns) > 0:
        best_match = df.columns[0]
        
    return best_match

def clean_url(url):
    """Ensure URL is properly formatted with protocol"""
    if not url:
        return None
    
    url = url.strip()
    if not url:
        return None
        
    # Add http:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = f'http://{url}'
    
    return url

def scrape_website(url, timeout=10, max_content_length=50000):
    """
    Scrape content from a single website
    
    Args:
        url: Website URL to scrape
        timeout: Request timeout in seconds
        max_content_length: Maximum content length to process
        
    Returns:
        Tuple of (extracted text content, list of links) or (error message, empty list)
    """
    if not url:
        return "No URL provided", []
    
    url = clean_url(url)
    if not url:
        return "Invalid URL", []
    
    try:
        # Set user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request - using verify=False but warnings are suppressed
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        
        # Store the full HTML for link extraction
        html_content = response.text[:max_content_length]  # Limit content size
        
        # Extract links using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        base_url = urlparse(url).scheme + "://" + urlparse(url).netloc
        
        # Find all link elements and extract href attributes
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle relative URLs
            if href.startswith('/'):
                full_url = urljoin(base_url, href)
                links.append(full_url)
            # Skip fragment identifiers or javascript
            elif href.startswith('#') or href.startswith('javascript:'):
                continue
            # Add complete URLs directly
            else:
                links.append(href)
        
        # Remove duplicates and sort
        links = sorted(list(set(links)))
        
        # Convert HTML to text
        text_content = h2t.handle(html_content)
        
        # Clean up the text (remove excessive newlines, etc.)
        text_content = '\n'.join(line for line in text_content.splitlines() if line.strip())
        
        return text_content[:max_content_length], links  # Return both content and links
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching website: {str(e)}", []
    except Exception as e:
        return f"Error processing website: {str(e)}", []

def process_websites_parallel(df, website_column, max_workers=10, timeout=10, max_content_length=50000):
    """
    Process all websites in the dataframe in parallel
    
    Args:
        df: Pandas DataFrame containing website URLs
        website_column: Column name containing website URLs
        max_workers: Maximum number of parallel workers
        timeout: Request timeout in seconds
        max_content_length: Maximum content length to process
        
    Returns:
        DataFrame with added 'website_content' and 'website_links' columns
    """
    if website_column not in df.columns:
        raise ValueError(f"Column '{website_column}' not found in dataframe")
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Get the list of URLs to process
    urls = df[website_column].tolist()
    
    # Create a list to store results
    results = []
    
    logger.info(f"Starting to process {len(urls)} websites with {max_workers} parallel workers")
    
    # Create a progress bar placeholder in Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store futures
        future_to_url = {executor.submit(scrape_website, url, timeout, max_content_length): i 
                         for i, url in enumerate(urls)}
        
        # Process results as they complete
        completed = 0
        total = len(future_to_url)
        
        for future in concurrent.futures.as_completed(future_to_url):
            idx = future_to_url[future]
            try:
                content, links = future.result()
                results.append((idx, content, links))
            except Exception as e:
                logger.error(f"Error processing URL at index {idx}: {str(e)}")
                results.append((idx, f"Error: {str(e)}", []))
            
            # Update progress
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Processing websites: {completed}/{total}")
    
    # Clear the status text
    status_text.empty()
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    
    # Add results as new columns
    result_df['website_content'] = [r[1] for r in results]
    result_df['website_links'] = [', '.join(r[2]) if r[2] else "" for r in results]
    
    logger.info(f"Completed processing {len(urls)} websites")
    return result_df

if __name__ == "__main__":
    # Example usage
    test_df = pd.DataFrame({
        'name': ['Example 1', 'Example 2'],
        'website': ['https://example.com', 'https://httpbin.org']
    })
    
    result = process_websites_parallel(test_df, 'website', max_workers=2)
    print(result.head()) 
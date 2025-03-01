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

# Use the centralized logger
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
    Simplified modal dialog for mapping website columns.
    
    Args:
        df: DataFrame containing the uploaded contact data
    """
    st.write("Please confirm the website and contact name columns")
    
    # Get all columns from the dataframe
    all_columns = list(df.columns)
    
    # Use fuzzy matching to guess the website column
    guessed_website_col = guess_website_column(df)
    
    # Check if the dataframe has separate first and last name columns
    has_separate_names, first_name_col, last_name_col = detect_name_columns(df)
    
    # Use fuzzy matching to guess the name column if no separate names detected
    guessed_name_col = guess_name_column(df) if not has_separate_names else None
    
    # Column selection section
    col1, col2 = st.columns(2)
    
    with col1:
        # Website column selection
        website_column = st.selectbox(
            "Select the column containing website URLs:",
            options=all_columns,
            index=all_columns.index(guessed_website_col) if guessed_website_col in all_columns else 0
        )
    
    # Handle name column selection based on whether we have separate first/last name columns
    if has_separate_names:
        st.info("We detected separate first and last name columns in your data. Please confirm them below.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # First name column selection
            first_name_column = st.selectbox(
                "Select the column containing first names:",
                options=all_columns,
                index=all_columns.index(first_name_col) if first_name_col in all_columns else 0
            )
        
        with col2:
            # Last name column selection
            last_name_column = st.selectbox(
                "Select the column containing last names:",
                options=all_columns,
                index=all_columns.index(last_name_col) if last_name_col in all_columns else 0
            )
            
        name_column = "full_name"  # This will be created during processing
    else:
        # Regular name column selection
        with col2:
            name_column = st.selectbox(
                "Select the column containing contact names:",
                options=all_columns,
                index=all_columns.index(guessed_name_col) if guessed_name_col in all_columns else 0
            )
    
    # Fixed configuration values
    max_workers = 20
    timeout = 10
    extract_links = True
    max_content_length = 50000
    
    # Store settings in session state
    if st.button("Start Website Scraping", key="dialog_start_scraping_button"):
        # Save mapping settings
        st.session_state.website_mapping = {
            "website_column": website_column,
            "name_column": name_column,
            "has_separate_names": has_separate_names,
            "first_name_column": first_name_column if has_separate_names else None,
            "last_name_column": last_name_column if has_separate_names else None,
            "max_workers": max_workers,
            "timeout": timeout,
            "extract_links": extract_links,
            "max_content_length": max_content_length
        }
        
        # If we have separate name columns, create a combined full name column
        if has_separate_names:
            # Create a copy of the dataframe
            result_df = df.copy()
            
            # Combine first and last names into a new column
            result_df['full_name'] = result_df.apply(
                lambda row: f"{row[first_name_column]} {row[last_name_column]}".strip(), 
                axis=1
            )
            
            # Use the new dataframe for processing
            df = result_df
        
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

def detect_name_columns(df):
    """
    Detect if the dataframe has separate first name and last name columns.
    
    Returns:
        tuple: (has_separate_names, first_name_col, last_name_col)
    """
    first_name_keywords = ['first name', 'firstname', 'first', 'given name', 'given', 'fname']
    last_name_keywords = ['last name', 'lastname', 'last', 'surname', 'family name', 'family', 'lname']
    
    # Find best match for first name
    first_name_col = None
    first_name_score = 0
    
    # Find best match for last name
    last_name_col = None
    last_name_score = 0
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for exact matches first (higher priority)
        if col_lower in first_name_keywords or any(keyword == col_lower for keyword in first_name_keywords):
            first_name_col = col
            first_name_score = 100  # Perfect match
            continue
            
        if col_lower in last_name_keywords or any(keyword == col_lower for keyword in last_name_keywords):
            last_name_col = col
            last_name_score = 100  # Perfect match
            continue
        
        # Then check for fuzzy matches
        for keyword in first_name_keywords:
            score = fuzz.ratio(col_lower, keyword)
            if score > first_name_score:
                first_name_score = score
                first_name_col = col
        
        for keyword in last_name_keywords:
            score = fuzz.ratio(col_lower, keyword)
            if score > last_name_score:
                last_name_score = score
                last_name_col = col
    
    # Check for common patterns in column names
    if not first_name_col or not last_name_col:
        for col in df.columns:
            col_lower = col.lower()
            # Check for columns that contain "first" or "last"
            if "first" in col_lower and (first_name_score < 80 or not first_name_col):
                first_name_col = col
                first_name_score = max(first_name_score, 80)
            if "last" in col_lower and (last_name_score < 80 or not last_name_col):
                last_name_col = col
                last_name_score = max(last_name_score, 80)
    
    # Determine if we have separate name columns
    # We consider it valid if both columns are found with a reasonable confidence
    has_separate_names = (first_name_score > 70 and last_name_score > 70 and 
                         first_name_col is not None and last_name_col is not None and
                         first_name_col != last_name_col)  # Make sure they're not the same column
    
    # If we have a full name column already, don't use separate names
    has_full_name = any(col.lower() in ['name', 'full name', 'fullname'] for col in df.columns)
    if has_full_name:
        has_separate_names = False
    
    return has_separate_names, first_name_col, last_name_col

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
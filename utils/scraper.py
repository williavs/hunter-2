import requests
import html2text
import pandas as pd
import concurrent.futures
from urllib.parse import urlparse
import time
import logging
import urllib3
import streamlit as st

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
        Extracted text content or error message
    """
    if not url:
        return "No URL provided"
    
    url = clean_url(url)
    if not url:
        return "Invalid URL"
    
    try:
        # Set user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request - using verify=False but warnings are suppressed
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        
        # Convert HTML to text
        html_content = response.text[:max_content_length]  # Limit content size
        text_content = h2t.handle(html_content)
        
        # Clean up the text (remove excessive newlines, etc.)
        text_content = '\n'.join(line for line in text_content.splitlines() if line.strip())
        
        return text_content[:max_content_length]  # Ensure we don't return too much content
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching website: {str(e)}"
    except Exception as e:
        return f"Error processing website: {str(e)}"

def process_websites_parallel(df, website_column, max_workers=10, timeout=10):
    """
    Process all websites in the dataframe in parallel
    
    Args:
        df: Pandas DataFrame containing website URLs
        website_column: Column name containing website URLs
        max_workers: Maximum number of parallel workers
        timeout: Request timeout in seconds
        
    Returns:
        DataFrame with added 'website_content' column
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
        future_to_url = {executor.submit(scrape_website, url, timeout): i 
                         for i, url in enumerate(urls)}
        
        # Process results as they complete
        completed = 0
        total = len(future_to_url)
        
        for future in concurrent.futures.as_completed(future_to_url):
            idx = future_to_url[future]
            try:
                content = future.result()
                results.append((idx, content))
            except Exception as e:
                logger.error(f"Error processing URL at index {idx}: {str(e)}")
                results.append((idx, f"Error: {str(e)}"))
            
            # Update progress
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Processing websites: {completed}/{total}")
    
    # Clear the status text
    status_text.empty()
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    
    # Add results as a new column
    result_df['website_content'] = [r[1] for r in results]
    
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
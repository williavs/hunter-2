"""
URL Sanitizer Utility

This module provides functions for cleaning and validating URLs to ensure they're properly formatted
before sending them to scraping functions.
"""

import re
from urllib.parse import urlparse
import pandas as pd

def sanitize_url(url_value):
    """
    Clean and validate a URL string.
    
    Args:
        url_value: A value that should be a URL (might be string, float, int, etc.)
        
    Returns:
        str: A cleaned URL string with proper formatting, or empty string if invalid
    """
    # Handle non-string values (like NaN, None, floats, etc.)
    if url_value is None:
        return ""
    
    # Convert to string if it's not already
    if not isinstance(url_value, str):
        # Check if it's NaN
        if pd.isna(url_value):
            return ""
        # Try converting to string
        try:
            url_value = str(url_value)
        except:
            return ""
    
    # Remove leading/trailing whitespace
    url_value = url_value.strip()
    
    # Remove surrounding quotes
    url_value = re.sub(r'^[\'"]|[\'"]$', '', url_value)
    
    # If empty after cleaning, return empty string
    if not url_value:
        return ""
    
    # Add http:// prefix if missing
    if not url_value.startswith(('http://', 'https://')):
        url_value = 'https://' + url_value
    
    # Basic URL validation
    try:
        result = urlparse(url_value)
        # Check if URL has valid format with netloc (domain)
        if result.netloc:
            return url_value
        return ""
    except:
        return ""

def sanitize_dataframe_urls(df, url_column):
    """
    Sanitize URLs in a specific column of a dataframe.
    
    Args:
        df: Pandas DataFrame containing URLs
        url_column: Column name that contains the URLs
        
    Returns:
        DataFrame: The DataFrame with sanitized URLs
    """
    if url_column not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Apply sanitize_url to each value in the URL column
    result_df[url_column] = result_df[url_column].apply(sanitize_url)
    
    # Count how many URLs were fixed
    cleaned_count = sum(1 for i, row in result_df.iterrows() 
                      if df.loc[i, url_column] != result_df.loc[i, url_column])
    
    return result_df, cleaned_count 
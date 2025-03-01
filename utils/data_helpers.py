"""
Data Helper Functions

This module provides utility functions for data loading, processing, and manipulation
used throughout the Email GTM Wizard application.
"""

import base64
import pandas as pd
import streamlit as st


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


def has_name_components(df):
    """Check if the dataframe has first and last name columns but no full name column"""
    # Define common column names for first and last names
    first_name_patterns = ['first name', 'firstname', 'first', 'given name', 'given', 'fname']
    last_name_patterns = ['last name', 'lastname', 'last', 'surname', 'family name', 'family', 'lname']
    
    # Check for first name column - exact matches first
    has_first = any(col.lower() in first_name_patterns for col in df.columns)
    
    # If no exact match, check for columns containing the keywords
    if not has_first:
        has_first = any(any(pattern in col.lower() for pattern in first_name_patterns) for col in df.columns)
    
    # Check for last name column - exact matches first
    has_last = any(col.lower() in last_name_patterns for col in df.columns)
    
    # If no exact match, check for columns containing the keywords
    if not has_last:
        has_last = any(any(pattern in col.lower() for pattern in last_name_patterns) for col in df.columns)
    
    # Check for full name column
    has_full = any(col.lower() in ['name', 'full name', 'fullname', 'contact name', 'contact'] for col in df.columns)
    
    # Make sure we don't have the same column matching both first and last name patterns
    if has_first and has_last:
        first_cols = [col for col in df.columns if col.lower() in first_name_patterns or 
                     any(pattern in col.lower() for pattern in first_name_patterns)]
        last_cols = [col for col in df.columns if col.lower() in last_name_patterns or 
                    any(pattern in col.lower() for pattern in last_name_patterns)]
        
        # If the same column matches both patterns, it's probably not separate first/last names
        if all(col in last_cols for col in first_cols) and all(col in first_cols for col in last_cols):
            return False
    
    return has_first and has_last and not has_full 
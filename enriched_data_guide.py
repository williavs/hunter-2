import streamlit as st

def maintain_session_states():
    """Initialize or maintain session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "website_column" not in st.session_state:
        st.session_state.website_column = None
    if "email_column" not in st.session_state:
        st.session_state.email_column = None

def show_enriched_data_guide():
    maintain_session_states()
    
    st.title("How to Use the Enriched Data")
    
    st.markdown("""
    ## Leveraging Personality Data for Sales and Marketing
    
    After using HUNTER and SPEAR, you've enriched your contact data with personality insights and generated personalized email content. 
    Here's how to effectively use this information across your sales and marketing efforts.
    """)
    
    # Important Note About Emails
    st.info("""
    ### Important Note About Generated Emails
    
    The emails produced by SPEAR contain only the body content of the email. This design gives you maximum flexibility for:
    
    - Adding your own personalized greeting (e.g., "Hi Tim," or "Hello John,")
    - Creating your own subject lines in your Sales Engagement Platform
    - Adding your own signature and sign-off
    - Using the email content as a variable in your email templates
    
    When downloading your enriched data, you can rename the email columns to whatever naming convention works best with your email platform.
    """)
    
    # Email Platforms Section
    st.header("Using Emails in Sales Engagement Platforms")
    
    st.markdown("""
    ### Implementation Steps:
    
    1. **Set Up Email Templates in Your SEP** (Outreach, SalesLoft, etc.):
       - Create templates that use variables/merge fields for the email body content
       - Design your own subject lines and greetings
       - Add your standard signature
    
    2. **Import Process Best Practices**:
       - Most SEPs like Outreach or SalesLoft have an 'Import Contacts' feature for uploading CSVs
       - These tools can handle duplications and update existing records
       - If working with new accounts/contacts that DO NOT exist in your CRM (Salesforce, HubSpot):
         * Create the contacts or accounts in your CRM FIRST
         * THEN upload to your SEP
         * This ensures all data is properly synced between systems
    
    3. **Using the Email Content**:
       - Map the email column from your CSV to the appropriate variable in your email template
       - Preview a few emails before sending to ensure correct formatting
       - Consider A/B testing different subject lines with the same personalized body content
    """)
    
    
    
    st.divider()
    st.markdown("""
    **Need further assistance?** Contact us for personalized guidance on implementing these strategies.
    """)

# This is how the page is run when accessed through streamlit
if __name__ == "__main__":
    show_enriched_data_guide()
else:
    # When loaded through streamlit navigation
    show_enriched_data_guide() 
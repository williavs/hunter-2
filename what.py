import streamlit as st

# Helper function to keep session state variables permanent
def keep_permanent_session_vars():
    """Prevents Streamlit from clearing session state variables with p_ prefix"""
    for key in list(st.session_state.keys()):
        if key.startswith("p_"):
            st.session_state[key] = st.session_state[key]

def show_what_page():
    """Display the WHAT? welcome page"""
    # Call our helper to keep permanent vars
    keep_permanent_session_vars()
    
    # Create the title with emoji
    st.title("WHAT IS HUNTER?")
    
    # Add a subtitle
    st.subheader("A Deep Sales Intelligence App.")

    # What is this tool section
    st.markdown("""
    ## What is this tool? 🔍

    > Allows sellers to research prospects and create personalized emails in bulk.
    """)

    # Use tabs for instructions
    tab1, tab2 = st.tabs(["How to Use", "Getting Started"])

    with tab1:
        # How to use section
        st.markdown("""
        ## How to use this tool: 📝

        ### 1. **HUNTER** 🎯
            - Navigate to the HUNTER page, follow the instructions to add your selling companies context
            - Upload your CSV of your PROSPECTS company data (export from 6Sense, Salesforce, Clay etc.)
            - HUNTER will scrape company websites for you automatically, then use the context to perform a Deep Research Analysis on each contact

        ### 2. **SPEAR** ⚡
            - Navigate to the SPEAR page
            - Select the contacts you want to write an email to
            - SPEAR will generate 4 email drafts for you
            - Name columns and export CSV with the results
        """)

    with tab2:
        # Getting started section
        st.markdown("""
        ### Getting Started:

        1. Set up your API keys in the sidebar
        2. Define your company context in HUNTER
        3. Upload your prospect CSV with website URLs
        4. Run personality analysis
        5. Generate targeted emails in SPEAR
        6. Download your results and get back to work!
        """)

    # Use columns for info and warnings
    col_info, col_warning = st.columns(2)

    with col_info:
        # Before You Start section
        st.info("""
        ### Before You Start

        This app requires two API keys to function:

        - **OpenRouter API Key**: Provides access to Claude 3.7 Sonnet, which powers the personality analysis and email generation. [Get your OpenRouter API key here](https://openrouter.ai/settings/keys).

        - **Tavily API Key**: Enables intelligent web search to gather information about your prospects. [Get your Tavily API key here](https://app.tavily.com/).

        """)

    with col_warning:
        # Things to Keep in Mind section
        st.warning("""
        ### Things to Keep in Mind

        - **Start Small**: Begin with a small contact list (5-10 contacts) to learn how the system works
        - **Size Limit**: The system cannot handle lists larger than 50 rows of contacts (lots of AI processing happening)
        - **SEP Preparation**: Check what custom contact fields are available in your Sales Engagement Platform before trying to import
        - **Email Content**: Generated emails contain only the body text - you'll need to add greetings and signatures in your SEP
        """)

    # Add a visual separator
    st.markdown("---")

    # Add buttons to navigate to other pages
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Start Analyzing Prospects", use_container_width=True):
            st.switch_page("hunter.py")
    
    with col_btn2:
        if st.button("Learn about the AI used", use_container_width=True):
            st.switch_page("methodology.py")

# This ensures the page is displayed when loaded directly or through navigation
if __name__ == "__main__":
    show_what_page()
else:
    # When loaded through streamlit navigation
    show_what_page() 
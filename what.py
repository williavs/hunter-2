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
    
    # Main description
    st.markdown("""
    ### What is this tool?
    
    Allows sellers to research prospects and create personalized emails in bulk. 
    
    ### How it works:
    
    1. **HUNTER** - Upload your contact data and analyze prospect personalities
    2. **SPEAR** - Generate personalized emails based on the analysis
    """)
    
    # Before You Start section
    st.info("""
    ### Before You Start
    
    This app requires two API keys to function:
    
    - **OpenRouter API Key**: Provides access to Claude 3.7 Sonnet, which powers the personality analysis and email generation. [Get your OpenRouter API key here](https://openrouter.ai/settings/keys).
    
    - **Tavily API Key**: Enables intelligent web search to gather information about your prospects. [Get your Tavily API key here](https://app.tavily.com/).
    
    This is a generalized prototype application. If you'd like a custom implementation for your company, [contact Willy VanSickle](mailto:willyv3@v3-ai.com) or check out the [@V3Consult page](about.py).
    """)
    
    st.markdown("""
    ### Getting Started:
    
    1. Set up your API keys in the sidebar
    2. Define your company context in HUNTER
    3. Upload your prospect CSV with website URLs
    4. Run personality analysis
    5. Generate targeted emails in SPEAR
    6. Download your results and get back to work! 
    """)
    
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
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Analyzing Prospects", use_container_width=True):
            st.switch_page("main.py")
    
    with col2:
        if st.button("Learn about the AI used", use_container_width=True):
            st.switch_page("methodology.py")
    

# This ensures the page is displayed when loaded directly or through navigation
if __name__ == "__main__":
    show_what_page()
else:
    # When loaded through streamlit navigation
    show_what_page() 
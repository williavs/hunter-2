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
    
    ### Getting Started:
    
    1. Set up your API keys in the sidebar
    2. Define your company context in HUNTER
    3. Upload your prospect CSV with website URLs
    4. Run personality analysis
    5. Generate targeted emails in SPEAR
    6. Download your results and get back to work! 
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
import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="HUNTER",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to keep session state variables permanent
def keep_permanent_session_vars():
    """Prevents Streamlit from clearing session state variables with p_ prefix"""
    for key in list(st.session_state.keys()):
        if key.startswith("p_"):
            st.session_state[key] = st.session_state[key]

# Call our helper function to ensure permanent session state variables persist
keep_permanent_session_vars()

# Import modules after setting page config
# Don't import main directly to avoid duplicate UI elements
# import main

st.sidebar.title(":red[HUNTER]")
st.sidebar.markdown(":grey[DEEP SALES INTELLIGENCE]")

# Define the pages for navigation
pages = {
    "Tools": [
        st.Page("what.py", title="WHAT?"),
        st.Page("hunter.py", title="HUNTER"),
        st.Page("spear.py", title="SPEAR"),
    ],
    "Resources": [

        st.Page("methodology.py", title="For the AI Nerds"),
        st.Page("enriched_data_guide.py", title="How to Use the Enriched Data")
    ],
}

# Create the navigation
pg = st.navigation(pages, expanded=True)

# Run the selected page
pg.run() 
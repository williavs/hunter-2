import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="HUNTER",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title(":red[HUNTER]")
st.sidebar.markdown(":grey[Deep Sales Intelligence]")

# Define the pages for navigation
pages = {
    "GTM Tools": [
        st.Page("main.py", title="Contact Enrichment"),
        st.Page("honeybadger_strategies.py", title="Honey Badger Strategies"),
    ],
    "Resources": [
        st.Page("about.py", title="About"),
    ],
}

# Create the navigation
pg = st.navigation(pages, expanded=True)

# Run the selected page
pg.run() 
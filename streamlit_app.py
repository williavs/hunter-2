import streamlit as st
from about import show_about_page
from methodology import show_methodology_page
import main

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="GTM Wizards",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title(":red[HUNTER]")
st.sidebar.markdown(":grey[Deep Sales Intelligence]")

# Define the pages for navigation
pages = {
    "GTM Tools": [
        st.Page("main.py", title="Contact Enrichment"),
        st.Page("honeybadger_strategies.py", title="Honey Badger -JMM"),
    ],
    "Resources": [
        st.Page("about.py", title="About"),
        st.Page("methodology.py", title="Technical Methodology"),
    ],
}

# Create the navigation
pg = st.navigation(pages, expanded=False)

# Run the selected page
pg.run() 
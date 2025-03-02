import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="GTM Wizards",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules after setting page config
from about import show_about_page
from methodology import show_methodology_page
import main

st.sidebar.title(":red[HUNTER]")
st.sidebar.markdown(":grey[Deep Sales Intelligence]")

# Define the pages for navigation
pages = {
    "Tool": [
        st.Page("main.py", title="HUNTER"),
        st.Page("honeybadger_strategies.py", title="Honey Badger -JMM"),
    ],
    "Resources": [
        st.Page("about.py", title="About"),
        st.Page("honeybadger_strategies.py", title="Honey Badger -JMM"),
        st.Page("methodology.py", title="For the AI Nerds"),
    ],
}

# Create the navigation
pg = st.navigation(pages, expanded=False)

# Run the selected page
pg.run() 
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

# --- API Settings Expander and Mode Toggle ---
with st.sidebar.expander("AI API Settings", expanded=False):
    options = ["Proxy", "OpenAI"]
    default_mode = st.session_state.get("ai_mode", options[0])
    try:
        default_index = options.index(default_mode)
    except ValueError:
        default_index = 0
    mode = st.radio(
        "Model Provider Mode:",
        options=options,
        index=default_index,
        horizontal=True,
        key="ai_mode_radio"
    )
    st.session_state["ai_mode"] = mode

    if mode == "OpenAI":
        if st.session_state.get("openai_api_key", "") and not st.session_state.get("edit_openai_api_key", False):
            st.success("OpenAI API Key saved.")
            if st.button("Edit OpenAI API Key", key="edit_openai_api_key_btn"):
                st.session_state["edit_openai_api_key"] = True
        else:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("openai_api_key", ""),
                type="password",
                key="openai_api_key_input"
            )
            if st.button("Save OpenAI API Key", key="save_openai_api_key_btn"):
                st.session_state["openai_api_key"] = openai_api_key
                st.session_state["edit_openai_api_key"] = False
                st.success("OpenAI API Key saved.")
    else:
        if st.session_state.get("proxy_api_key", "") and not st.session_state.get("edit_proxy_api_key", False):
            st.success("Proxy API Key saved.")
            if st.button("Edit Proxy API Key", key="edit_proxy_api_key_btn"):
                st.session_state["edit_proxy_api_key"] = True
        else:
            proxy_api_key = st.text_input(
                "Proxy API Key",
                value=st.session_state.get("proxy_api_key", ""),
                type="password",
                key="proxy_api_key_input"
            )
            if st.button("Save Proxy API Key", key="save_proxy_api_key_btn"):
                st.session_state["proxy_api_key"] = proxy_api_key
                st.session_state["edit_proxy_api_key"] = False
                st.success("Proxy API Key saved.")
        if st.session_state.get("proxy_base_url", "") and not st.session_state.get("edit_proxy_base_url", False):
            st.success(f"Proxy Base URL: {st.session_state['proxy_base_url']}")
            if st.button("Edit Proxy Base URL", key="edit_proxy_base_url_btn"):
                st.session_state["edit_proxy_base_url"] = True
        else:
            proxy_base_url = st.text_input(
                "Proxy Base URL",
                value=st.session_state.get("proxy_base_url", "https://llm.data-qa.justworks.com"),
                key="proxy_base_url_input"
            )
            if st.button("Save Proxy Base URL", key="save_proxy_base_url_btn"):
                st.session_state["proxy_base_url"] = proxy_base_url
                st.session_state["edit_proxy_base_url"] = False
                st.success("Proxy Base URL saved.")

    st.caption("These keys are stored only in your session and never sent to any server except the selected provider.")

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
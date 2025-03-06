# Simple First Steps to Improve the Codebase

Let's focus on the most critical issues that can be fixed quickly without rewriting the entire application. These are straightforward changes that will make the code more maintainable while preserving the current functionality.

## Top 5 Issues to Address First

### 1. Fix the Page Rendering Pattern

**Current Problem:**
```python
# At the end of each page file:
if __name__ == "__main__":
    show_page()
if __name__ != "__main__":
    show_page()
```

This pattern calls `show_page()` regardless of how the file is accessed, which is redundant and confusing.

**Simple Solution:**
```python
# Replace with:
if __name__ == "__main__" or __name__ != "__main__":
    show_page()

# Or even better, just:
show_page()
```

### 2. Extract Deeply Nested Logic to Functions

**Current Problem:**
```python
if st.session_state.p_email_generation_complete:
    # 5 levels of nested conditionals to check if emails contain content
    # ... (20+ lines of complex checks)
```

**Simple Solution:**
Create helper functions for these checks:

```python
def has_email_content(indices, df):
    """Check if any of the selected rows have email content."""
    for idx in indices:
        for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
            if col in df.columns and idx < len(df):
                value = df.iloc[idx][col]
                if isinstance(value, str) and value.strip() and not value.startswith("Error:"):
                    return True
    return False

# Then use it:
if st.session_state.p_email_generation_complete:
    if has_email_content(st.session_state.p_selected_rows, st.session_state.p_spear_df):
        st.success("Emails generated successfully!")
    else:
        st.warning("Generation completed but no content was produced.")
```

### 3. Reduce Manual Reruns

**Current Problem:**
```python
st.session_state.p_generating_emails = True
st.session_state.p_email_generation_complete = False
st.rerun()
```

These manual rerun triggers make it hard to follow the application flow.

**Simple Solution:**
Use Streamlit's natural flow where possible and only use `st.rerun()` when absolutely necessary:

```python
if st.button("Generate Emails"):
    # Do the work directly here
    st.session_state.results = generate_emails(selected_rows)
    # No rerun needed in many cases
```

### 4. Consolidate Duplicate Code

**Current Problem:**
Each page has its own copy of similar initialization code, data loading logic, and UI patterns.

**Simple Solution:**
Create a shared utility file with common functions:

```python
# In utils.py
def initialize_session_state(keys_with_defaults):
    """Initialize session state variables if they don't exist."""
    for key, default in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Then in each page:
from utils import initialize_session_state

initialize_session_state({
    "p_spear_df": pd.DataFrame(),
    "p_selected_rows": [],
    "p_generating_emails": False,
    "p_email_generation_complete": False
})
```

### 5. Simplify Dialog Management

**Current Problem:**
```python
@st.dialog("Add Case Studies")
def show_case_studies_dialog():
    # ...dialog content...
    if st.button("Save Case Studies"):
        st.session_state.p_case_studies = case_studies
        st.session_state.p_dismissed_case_studies_dialog = True
        st.rerun()
```

Multiple state variables track dialog state, creating complexity.

**Simple Solution:**
Let Streamlit dialogs manage their own visibility:

```python
@st.dialog("Add Case Studies")
def show_case_studies_dialog():
    # ...dialog content...
    if st.button("Save Case Studies"):
        st.session_state.p_case_studies = case_studies
        st.rerun()  # Only rerun to close the dialog

# Only show the dialog when needed:
if not st.session_state.get("p_case_studies"):
    show_case_studies_dialog()
```

## Implementation Approach

1. **Start Small**: Fix one file at a time, beginning with the most used pages
2. **Test Frequently**: After each change, test the application to ensure functionality is preserved
3. **No Big Rewrites**: Focus on targeted improvements rather than rewriting the whole application
4. **Incremental Progress**: Gradually apply these patterns across the codebase

## Expected Benefits

- **Improved Readability**: Code will be easier to understand
- **Better Maintainability**: Easier to add features and fix bugs
- **Reduced Complexity**: Fewer moving parts and state variables
- **Preserved Functionality**: Users won't notice any changes in behavior

These simple changes will significantly improve the codebase without requiring a complete rewrite or architectural overhaul. 
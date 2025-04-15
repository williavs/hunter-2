# Streamlit Application Development: Best Practices Guide

Dear AI Assistant,

When developing Streamlit applications, it's important to work with Streamlit's design philosophy rather than against it. Streamlit is built around a simple execution model where the entire script runs from top to bottom on each interaction. Embracing this model leads to cleaner, more maintainable code.

## Core Streamlit Principles

### 1. Embrace the Top-to-Bottom Execution Flow

Streamlit's execution model is intentionally simple: your script runs from top to bottom on each interaction. This is a feature, not a limitation.

✅ **DO**
```python
if st.button("Calculate"):
    result = perform_calculation()
    st.write(f"Result: {result}")
```

❌ **DON'T** Create complex state machines with manual rerun triggers
```python
if st.session_state.get("calculating", False):
    result = perform_calculation()
    st.session_state.result = result
    st.session_state.calculating = False
    st.rerun()  # Avoid unnecessary reruns

if st.button("Calculate"):
    st.session_state.calculating = True
    st.rerun()

if "result" in st.session_state:
    st.write(f"Result: {st.session_state.result}")
```

### 2. Use Session State Appropriately

Session state is for persistence between reruns, not for controlling application flow.

✅ **DO**
```python
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Increment"):
    st.session_state.count += 1

st.write(f"Count: {st.session_state.count}")
```

❌ **DON'T** Create obscure naming schemes or unnecessary state variables
```python
# Avoid this pattern
if "p_count" not in st.session_state:
    st.session_state.p_count = 0
if "p_incrementing" not in st.session_state:
    st.session_state.p_incrementing = False

if st.button("Increment") or st.session_state.p_incrementing:
    st.session_state.p_incrementing = False
    st.session_state.p_count += 1

st.write(f"Count: {st.session_state.p_count}")
```

### 3. Leverage Streamlit's Multipage App Structure

For multi-page applications, use Streamlit's built-in pages feature rather than creating your own navigation system.

✅ **DO** Use the standard directory structure:
```
📁 streamlit_app/
├── 📄 Home.py
└── 📁 pages/
    ├── 📄 01_Page1.py
    ├── 📄 02_Page2.py
    └── 📄 03_Page3.py
```

❌ **DON'T** Create custom page rendering logic
```python
# Avoid this pattern at the end of every file
if __name__ == "__main__":
    show_page()
if __name__ != "__main__":
    show_page()
```

## UI Design Patterns

### 1. Modularize UI Components

Break your UI into logical functions rather than deeply nested conditions.

✅ **DO**
```python
def show_data_input():
    st.header("Data Input")
    # Input widgets here

def show_results(data):
    st.header("Results")
    # Display results here

# Main app flow
data = show_data_input()
if data:
    show_results(data)
```

❌ **DON'T** Use deeply nested conditionals
```python
# Avoid deeply nested rendering logic
if not df.empty:
    st.header("Data")
    if st.session_state.get("show_chart", False):
        if "selected_columns" in st.session_state:
            if len(st.session_state.selected_columns) > 0:
                # Chart rendering logic
```

### 2. Simplify UI State Management

Let Streamlit handle widget states naturally.

✅ **DO**
```python
selected_option = st.selectbox("Choose an option", options)
if selected_option:
    st.write(f"You selected {selected_option}")
```

❌ **DON'T** Manually track widget values
```python
# Avoid manual widget state tracking
if st.selectbox("Choose an option", options, key="options_select"):
    selection = st.session_state.options_select
    st.session_state.p_last_selection = selection
    st.rerun()
if "p_last_selection" in st.session_state:
    st.write(f"You selected {st.session_state.p_last_selection}")
```

### 3. Use Streamlit Forms for Batch Operations

When you need to submit multiple inputs at once, use Streamlit's form container.

✅ **DO**
```python
with st.form("data_form"):
    name = st.text_input("Name")
    age = st.number_input("Age")
    submitted = st.form_submit_button("Submit")
    
if submitted:
    st.write(f"Hello {name}, you are {age} years old")
```

❌ **DON'T** Create your own form logic with state variables
```python
# Avoid custom form submission logic
name = st.text_input("Name", key="name_input")
age = st.number_input("Age", key="age_input")

if st.button("Submit") or st.session_state.get("p_form_submitted", False):
    st.session_state.p_form_submitted = False
    st.write(f"Hello {st.session_state.name_input}, you are {st.session_state.age_input} years old")
```

## Layout Best Practices

### 1. Use Containers for Logical Grouping

✅ **DO**
```python
with st.container():
    st.header("Analysis Section")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Value 1", 42)
    with col2:
        st.metric("Value 2", 123)
```

### 2. Prefer Simple Layouts

Use columns judiciously and avoid deeply nested layout containers.

✅ **DO**
```python
col1, col2 = st.columns(2)
with col1:
    st.write("Column 1 content")
with col2:
    st.write("Column 2 content")
```

❌ **DON'T** Create overly complex nested layouts
```python
# Avoid excessive nesting
col1, col2 = st.columns(2)
with col1:
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        subsubcol1, subsubcol2 = st.columns(2)
        # This becomes hard to maintain
```

## Data Handling Recommendations

### 1. Use Caching Effectively

Leverage Streamlit's caching system for expensive operations.

✅ **DO**
```python
@st.cache_data
def load_data():
    # Expensive data loading operation
    return pd.read_csv("large_file.csv")

data = load_data()  # This will only run once and then be cached
```

### 2. Simplify Dataframe Operations

Perform transformations in clear, separate functions rather than inline.

✅ **DO**
```python
def filter_data(df, min_value):
    return df[df["value"] > min_value]

filtered_data = filter_data(df, min_value)
st.dataframe(filtered_data)
```

## Error Handling

### 1. Use Streamlit's Native Feedback Mechanisms

✅ **DO**
```python
try:
    result = process_data(data)
    st.success("Data processed successfully!")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
```

## Final Recommendations

1. **Keep it simple**: Streamlit's power comes from its simplicity. Don't overcomplicate it.
2. **Follow the flow**: Work with Streamlit's execution model, not against it.
3. **Modularize**: Break your app into logical functions for better maintainability.
4. **Limit state**: Use session state only when needed for persistence, not for control flow.
5. **Test frequently**: Run your app often while developing to catch issues early.

Remember, a good Streamlit app feels natural and intuitive to use. By following these guidelines, you'll create applications that are not only easier to maintain but also provide a better experience for your users.

Best regards,
An AI assistant committed to clean Streamlit development 
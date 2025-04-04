import streamlit as st
import pandas as pd
import asyncio
from ai_utils.spear_email_generator import generate_spear_emails
import re


# Helper function to keep session state variables permanent
def keep_permanent_session_vars():
    """Prevents Streamlit from clearing session state variables with p_ prefix"""
    for key in list(st.session_state.keys()):
        if key.startswith("p_"):
            if key not in ["p_spear_df", "p_selected_rows", "p_generating_emails", "p_email_generation_complete", "p_case_studies", "p_dismissed_case_studies_dialog", "p_show_download_dialog"]:
                st.session_state["p_" + key] = st.session_state[key]

# Initialize persistent session state variables
if "p_spear_df" not in st.session_state:
    st.session_state.p_spear_df = pd.DataFrame()
    
if "p_selected_rows" not in st.session_state:
    st.session_state.p_selected_rows = []
    
if "p_generating_emails" not in st.session_state:
    st.session_state.p_generating_emails = False

if "p_email_generation_complete" not in st.session_state:
    st.session_state.p_email_generation_complete = False
    
if "p_case_studies" not in st.session_state:
    st.session_state.p_case_studies = ""
    
if "p_dismissed_case_studies_dialog" not in st.session_state:
    st.session_state.p_dismissed_case_studies_dialog = False

if "p_show_download_dialog" not in st.session_state:
    st.session_state.p_show_download_dialog = False


@st.dialog("Add Case Studies")
def show_case_studies_dialog():
    """Dialog for adding case studies to prevent hallucination in emails."""
    st.write("### Important: Case Studies Required")
    st.write("To generate accurate and non-hallucinated emails, please provide real case studies of your product.")
    st.write("These will be used to create social proof emails that reference actual customer success stories.")
    
    case_studies = st.text_area(
        "Enter case studies (Include company names, industry, challenges, solutions, and specific metrics):",
        value="",
        height=200,
        key="case_studies_dialog_input"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save Case Studies", key="save_case_studies_btn", use_container_width=True):
            st.session_state.p_case_studies = case_studies
            st.session_state.p_dismissed_case_studies_dialog = True
            st.rerun()
    with col2:
        if st.button("Dismiss", key="dismiss_case_studies_btn", use_container_width=True):
            st.session_state.p_dismissed_case_studies_dialog = True
            st.rerun()


@st.dialog("Customize and Download Data", width="large")
def show_download_dialog():
    """Dialog for customizing columns and file name before download."""
    st.write("### Customize Your Export")
    st.write("Rename columns and set your file name before downloading.")
    
    # Get the dataframe to download
    df = st.session_state.p_spear_df.copy()
    
    # Create input for file name
    default_filename = "spear_data.csv"
    if "download_filename" in st.session_state:
        default_filename = st.session_state.download_filename
    
    filename = st.text_input(
        "File name:", 
        value=default_filename,
        key="download_filename"
    )
    
    # Add .csv extension if not present
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    
    # Create a section for column renaming
    st.write("### Column Names")
    st.write("Rename your columns below (leave blank to keep original name):")
    
    # Store column mapping in session state
    if "column_mapping" not in st.session_state:
        st.session_state.column_mapping = {col: col for col in df.columns}
    
    # Create a dictionary to store new column names
    col_mappings = {}
    
    # Create four columns for the inputs to save space
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    # Organize columns into four groups
    cols = list(df.columns)
    group_size = len(cols) // 4 + (1 if len(cols) % 4 > 0 else 0)
    
    # Distribute columns across the four column groups
    for i, col in enumerate(cols):
        group_idx = i // group_size  # Determine which group this column belongs to
        if group_idx >= 4:  # If we somehow have more groups than columns, cap at 3 (0-indexed)
            group_idx = 3
            
        with columns[group_idx]:
            default_value = st.session_state.column_mapping.get(col, col)
            new_name = st.text_input(
                f"'{col}':", 
                value=default_value,
                key=f"col_rename_{i}"
            )
            col_mappings[col] = new_name if new_name else col
    
    # Update session state with new mappings
    st.session_state.column_mapping = col_mappings
    
    # Preview section
    with st.expander("Preview Changes", expanded=False):
        preview_df = df.copy()
        # Rename columns based on the mapping
        preview_df.columns = [col_mappings.get(col, col) for col in preview_df.columns]
        st.dataframe(preview_df.head(3), use_container_width=True)
    
    # Download buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Download", key="confirm_download_btn", use_container_width=True):
            # Create a new dataframe with renamed columns
            download_df = df.copy()
            download_df.columns = [col_mappings.get(col, col) for col in download_df.columns]
            
            # Convert to CSV
            csv_data = download_df.to_csv(index=False)
            
            # Store in session state for the download button
            st.session_state.download_csv_data = csv_data
            st.session_state.download_csv_filename = filename
            
            # Close dialog and trigger download
            st.session_state.p_show_download_dialog = False
            st.rerun()
    
    with col2:
        if st.button("Cancel", key="cancel_download_btn", use_container_width=True):
            st.session_state.p_show_download_dialog = False
            st.rerun()


def show_spear_page():
    # Call our helper to keep permanent vars
    keep_permanent_session_vars()
    
    # Show the case studies dialog if needed
    if not st.session_state.p_case_studies and not st.session_state.p_dismissed_case_studies_dialog:
        show_case_studies_dialog()
    
    # Show the download dialog if triggered
    if st.session_state.p_show_download_dialog:
        show_download_dialog()
    
    # Ensure data persistence across page navigation
    # Check if we need to grab data from the main app
    if "p_df" in st.session_state and st.session_state.p_df is not None and not isinstance(st.session_state.p_df, bool):
        if st.session_state.p_spear_df.empty:
            # Only copy if we haven't already or if the spear dataframe is empty
            st.session_state.p_spear_df = st.session_state.p_df.copy()
            st.success("Successfully loaded data from HUNTER")
    
    st.title("SPEAR")
    
    # Add a refresh button in the header area
    refresh_col1, refresh_col2 = st.columns([3, 1])
    with refresh_col1:
        st.write("### EMAIL TOOL")
    with refresh_col2:
        if st.button("🔄 Refresh Data from HUNTER", key="spear_refresh_btn"):
            if "p_df" in st.session_state and st.session_state.p_df is not None and not isinstance(st.session_state.p_df, bool):
                st.session_state.p_spear_df = st.session_state.p_df.copy()
                st.success("Data refreshed from HUNTER")
                st.rerun()
            else:
                st.warning("No data available from HUNTER")
    
    # Display information about the dataframe
    if not st.session_state.p_spear_df.empty:
        st.write(f"### Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", st.session_state.p_spear_df.shape[0])
        col2.metric("Columns", st.session_state.p_spear_df.shape[1])
        col3.metric("Data Points", st.session_state.p_spear_df.size)
        
       
    else:
        st.info("No data available. Please refresh data from HUNTER or generate sample data.")
        
        # Simple sample data generator
        if st.button("Generate Sample Data", key="spear_generate_btn", use_container_width=True):
            # Create sample data with various column types for demonstration
            st.session_state.p_spear_df = pd.DataFrame({
                'Name': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
                'Email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'],
                'Department': ['Sales', 'Marketing', 'Engineering', 'HR', 'Sales'],
                'Performance': [4.2, 3.8, 4.5, 4.1, 3.9],
                'Salary': [75000, 82000, 95000, 67000, 78000],
            })
            # Make sure to store in both regular and permanent session state
            keep_permanent_session_vars()
            st.success("Sample data generated successfully!")
            st.rerun()
        
    # Interactive dataframe with selection capability
    if not st.session_state.p_spear_df.empty:
        # Email generation section
        st.write("### Email Generation")
        
        # Add case studies editor
        case_studies_expander = st.expander("Case Studies for Social Proof Emails", expanded=False)
        with case_studies_expander:
            st.write("Provide real case studies to avoid hallucinated customer examples in the generated emails.")
            
            case_studies = st.text_area(
                "Real customer success stories (include company names, metrics, and results):",
                value=st.session_state.p_case_studies,
                height=200,
                key="case_studies_input",
                on_change=lambda: setattr(st.session_state, 'p_case_studies', st.session_state.case_studies_input)
            )
            
            if st.button("Update Case Studies", key="update_case_studies_btn"):
                st.session_state.p_case_studies = case_studies
                st.success("Case studies updated successfully!")
        
        # Button to generate JMM SPEAR emails for selected rows
        if st.button("Generate SPEAR Emails", key="generate_emails_btn", use_container_width=True):
            # First check session state selection if dataframe selection isn't accessible
            selected_rows = []
            if 'spear_dataframe' in st.session_state and hasattr(st.session_state.spear_dataframe, 'selection'):
                selected_rows = st.session_state.spear_dataframe.selection.rows
            elif st.session_state.p_selected_rows:
                selected_rows = st.session_state.p_selected_rows
                
            if not selected_rows:
                st.warning("Please select at least one row from the table below to generate emails.")
            else:
                # Store selected rows in persistent session state
                st.session_state.p_selected_rows = list(selected_rows)
                # Set processing state
                st.session_state.p_generating_emails = True
                st.session_state.p_email_generation_complete = False
                st.rerun()
                    
        # Generate emails if button was clicked
        if st.session_state.p_generating_emails:
            with st.spinner("Generating JMM-style SPEAR emails in parallel..."):
                # Use either the dataframe selection or our persistent copy
                selected_indices = []
                if 'spear_dataframe' in st.session_state and hasattr(st.session_state.spear_dataframe, 'selection') and st.session_state.spear_dataframe.selection.rows:
                    selected_indices = st.session_state.spear_dataframe.selection.rows
                elif st.session_state.p_selected_rows:
                    selected_indices = st.session_state.p_selected_rows
                    
                if selected_indices:
                    # Process in the background
                    try:
                        # Add debug message
                        st.info(f"Processing {len(selected_indices)} selected contacts...")
                        
                        # Process the selected rows in true parallel, include case studies
                        updated_df = asyncio.run(generate_spear_emails(
                            st.session_state.p_spear_df, 
                            company_context={"case_studies": st.session_state.p_case_studies},  # Pass case studies to the email generator
                            selected_indices=selected_indices
                        ))
                        
                        # Check if emails were actually generated
                        has_content = False
                        error_messages = set()
                        
                        # Examine the generated emails for each selected row
                        for idx in selected_indices:
                            for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
                                if idx < len(updated_df) and col in updated_df.columns:
                                    value = updated_df.iloc[idx][col]
                                    if isinstance(value, str):
                                        if value.strip() and not value.startswith("Error:"):
                                            has_content = True
                                        elif value.startswith("Error:"):
                                            error_messages.add(value)
                        
                        # Update the dataframe regardless of content
                        st.session_state.p_spear_df = updated_df
                        
                        # Reset state and show appropriate message
                        st.session_state.p_generating_emails = False
                        st.session_state.p_email_generation_complete = True
                        
                        if has_content:
                            st.success(f"Successfully generated 4 SPEAR emails for {len(selected_indices)} contacts in parallel!")
                        else:
                            if error_messages:
                                error_list = "\n".join([f"• {msg}" for msg in error_messages])
                                st.error(f"Failed to generate emails. Errors encountered:\n{error_list}")
                            else:
                                st.warning("Email generation completed, but no content was generated. Check the logs for more details.")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating emails: {str(e)}")
                        # Add more debug information
                        st.error("Check the logs for detailed error information.")
                        st.session_state.p_generating_emails = False
                        st.session_state.p_email_generation_complete = False
                else:
                    # No selection exists, reset the state
                    st.warning("No rows selected. Please select rows from the table below.")
                    st.session_state.p_generating_emails = False
        
        # Show success message if complete
        if st.session_state.p_email_generation_complete:
            # Check if we actually have content in the generated emails
            has_content = False
            if st.session_state.p_selected_rows:
                for idx in st.session_state.p_selected_rows:
                    for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]:
                        if col in st.session_state.p_spear_df.columns and idx < len(st.session_state.p_spear_df):
                            value = st.session_state.p_spear_df.iloc[idx][col]
                            if isinstance(value, str) and value.strip() and not value.startswith("Error:"):
                                has_content = True
                                break
            
            if has_content:
                st.success("✅ All 4 SPEAR emails generated in single batch! Check the 'emailbody1', 'emailbody2', 'emailbody3', and 'emailbody4' columns.")
            else:
                st.warning("⚠️ The email generation process completed, but no content was generated. This may indicate an issue with the API response format.")
                # Add a button to view debug info
                if st.button("Show Technical Details", key="show_debug_info"):
                    st.info("Debug Information")
                    st.json({
                        "selected_rows": st.session_state.p_selected_rows,
                        "dataframe_columns": list(st.session_state.p_spear_df.columns),
                        "email_columns_exist": [col in st.session_state.p_spear_df.columns for col in ["emailbody1", "emailbody2", "emailbody3", "emailbody4"]]
                    })
        
        # Show the interactive dataframe
        st.write("### Data Explorer")
        
        # Create column config for the dataframe
        column_config = {}
        
        # Auto-detect and configure columns based on data types
        for col in st.session_state.p_spear_df.columns:
            # URLs as links
            if col.lower().endswith(('url', 'link', 'website')):
                column_config[col] = st.column_config.LinkColumn(col)
            # Emails as links
            elif col.lower().endswith('email'):
                column_config[col] = st.column_config.LinkColumn(col, validate="^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$")
            # Numeric columns with currency format
            elif col.lower() in ('salary', 'income', 'revenue', 'cost', 'price', 'budget'):
                column_config[col] = st.column_config.NumberColumn(col, format="$%d")
            # Date columns
            elif 'date' in col.lower():
                column_config[col] = st.column_config.DateColumn(col, format="MMM DD, YYYY")
            # Percentage columns
            elif col.lower() in ('percentage', 'rate', 'ratio'):
                column_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
            # Email body columns - show them as text areas
            elif col.lower().startswith('emailbody'):
                column_config[col] = st.column_config.TextColumn(
                    col, 
                    width="large",
                    help=f"JMM-style SPEAR email - {col[-1:]}"
                )
        
        # Add a callback function to handle dataframe selection
        def handle_selection_change():
            """Store selected rows in session state when selection changes"""
            if 'spear_dataframe' in st.session_state and hasattr(st.session_state.spear_dataframe, 'selection'):
                # Update our persistent copy of selected rows
                st.session_state.p_selected_rows = list(st.session_state.spear_dataframe.selection.rows)

        # Update the dataframe display to use the callback
        selection = st.dataframe(
            st.session_state.p_spear_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=False,
            on_select=handle_selection_change,
            selection_mode=["multi-row", "multi-column"],
            key="spear_dataframe"
        )
        
        # Download button for the current dataframe
        st.write("### Download Data")
        
        # Check if we have data to download after dialog
        if "download_csv_data" in st.session_state and "download_csv_filename" in st.session_state:
            st.download_button(
                label="Download Customized CSV",
                data=st.session_state.download_csv_data,
                file_name=st.session_state.download_csv_filename,
                mime="text/csv",
                key="download_customized_btn",
                use_container_width=True
            )
            # Clear the data after download to prevent re-downloading
            if st.button("Reset Download", key="reset_download_btn"):
                if "download_csv_data" in st.session_state:
                    del st.session_state.download_csv_data
                if "download_csv_filename" in st.session_state:
                    del st.session_state.download_csv_filename
                st.rerun()
        else:
            # Show the customize button
            if st.button("Customize and Download CSV", key="customize_download_btn", use_container_width=True):
                st.session_state.p_show_download_dialog = True
                st.rerun()


show_spear_page() 
"""
Centralized logging configuration for the application.

This module provides a consistent logging setup across all components
of the application, including file and console output.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define log file paths
app_log_path = os.path.join(logs_dir, "app.log")
debug_log_path = os.path.join(logs_dir, "debug.log")

# Configure root logger
def configure_logging(level=logging.INFO):
    """
    Configure the root logger with the specified log level.
    
    Args:
        level: The logging level to use (default: INFO)
    """
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set the root logger level
    root_logger.setLevel(level)
    
    # Create formatters
    standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(standard_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs (INFO and above)
    file_handler = RotatingFileHandler(app_log_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(standard_formatter)
    root_logger.addHandler(file_handler)
    
    # Debug file handler (DEBUG and above)
    debug_file_handler = RotatingFileHandler(debug_log_path, maxBytes=10*1024*1024, backupCount=5)
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_file_handler)
    
    # Log the configuration
    logging.info(f"Logging configured with level: {logging.getLevelName(level)}")
    logging.info(f"Standard log file: {app_log_path}")
    logging.info(f"Debug log file: {debug_log_path}")

# Configure LangSmith tracing
def configure_langsmith_tracing(project_name="email-gtmwiz"):
    """
    Configure LangSmith tracing for LangChain and LangGraph.
    
    Args:
        project_name: The project name to use in LangSmith
    """
    # Check if LANGCHAIN_API_KEY is set
    langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not langchain_api_key:
        logging.warning("LANGCHAIN_API_KEY not set. LangSmith tracing will not be enabled.")
        return False
    
    # Enable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    logging.info(f"LangSmith tracing enabled for project: {project_name}")
    return True

# Initialize logging with default settings
configure_logging()

# Get a logger for this module
logger = logging.getLogger(__name__) 
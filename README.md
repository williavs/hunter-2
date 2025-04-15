# Email GTM Wizard - AI-Powered Contact Data Enrichment Tool

A comprehensive Streamlit application for enriching B2B sales outreach with AI-powered personality analysis and targeted insights.

## Overview

Email GTM Wizard helps sales professionals create highly targeted, personalized outreach by providing in-depth analysis of prospects' personalities, communication preferences, and business challenges. The application leverages AI to structure analyses using the "Route-Ruin-Multiply" (RRM) framework, connecting prospect pain points to your specific solutions.

![Application Screenshot](https://place-holder-for-screenshot.png)

## Key Features

- **AI-Powered Personality Analysis**: Generate detailed personality profiles for prospects using Claude 3.5 Haiku via OpenRouter
- **Company Context Integration**: Enhance analyses with your company's specific value propositions and solutions
- **Website Content Scraping**: Automatically gather information from prospect websites
- **Intelligent Name Handling**: Process both combined and separate first/last name fields
- **Web Search Integration**: Use Tavily search API to gather additional prospect information
- **Data Editor**: View and modify contact data with inline editing capabilities
- **Progress Tracking**: Monitor analysis and scraping progress in real-time
- **Controlled Processing**: Choose how many contacts to analyze to manage costs and processing time
- **RRM Framework**: Structured analyses that connect prospect pain points to your solutions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-gtmwiz.git
cd email-gtmwiz
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your API keys:
```
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Configure your API keys in the sidebar (if not set in `.env` file)

3. Add your company context information:
   - Enter your company website URL and target geography
   - Click "Scrape & Analyze" to automatically generate company context
   - Review and approve the generated context

4. Upload your CSV file containing contact data with website URLs

5. Map the column fields in the dialog (website URL, name fields, etc.)

6. Click "Start Website Scraping" to gather website content

7. Use the slider to select how many contacts to analyze

8. Click "Analyze Personalities" to generate AI-powered insights

9. Download the enriched data as CSV when processing is complete

## API Requirements

- **OpenRouter API Key**: Required for accessing Claude 3.5 Haiku for personality analysis
  - Sign up at [OpenRouter](https://openrouter.ai/)
  
- **Tavily API Key**: Required for web search functionality
  - Sign up at [Tavily](https://tavily.com/)

## Dependencies

- streamlit
- pandas
- numpy
- langchain
- langgraph
- html2text
- requests
- python-dotenv
- langsmith (optional, for tracing)

## Architecture

The application follows a modular, agent-based architecture using LangGraph for workflow orchestration:
- Personality Analyzer Module: Processes contact information using AI models
- Company Context Module: Analyzes company websites to extract business insights
- Data Processing Pipeline: Manages data flow between components
- Search Integration: Gathers relevant information about contacts

## License

MIT 
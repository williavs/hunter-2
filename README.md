# HUNTER - Deep Sales Intelligence System

HUNTER is a sophisticated AI-powered sales intelligence platform designed for B2B sales teams to deeply analyze prospects and generate highly personalized outreach content.

## Project Purpose & Goals

HUNTER solves the critical challenge of creating truly personalized sales outreach at scale by:

1. **Analyzing Prospect Personalities**: Using AI to build detailed profiles based on available data
2. **Contextualizing Business Challenges**: Identifying specific pain points related to your solutions
3. **Generating Personalized Emails**: Creating highly targeted content using the RRM framework
4. **Streamlining Workflow**: Integrating with existing sales engagement platforms seamlessly

## Technical Architecture

The application follows a multi-agent, workflow-based architecture:

```
User Input → Data Preprocessing → Search Planning → Web Search → 
Analysis Orchestration → Personality Analysis → Evaluation → Results
```

Key components:
- **Streamlit UI**: Modern, intuitive interface with multi-page navigation
- **LangGraph Workflow**: Orchestrates the AI agents through a stateful process
- **Web Scraping**: Extracts content from prospect websites for analysis
- **OpenAI Web Search**: Uses OpenAI's Responses API with web search capabilities
- **Email Generation**: Creates multiple email variations using the RRM framework

## Key Technologies & Dependencies

- **Frontend**: [Streamlit](https://streamlit.io/) for UI components and data visualization
- **Language Models**: [OpenAI GPT-4.1](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) with web search capabilities
- **LLM Integration**: [LangChain](https://python.langchain.com/docs/integrations/llms/openai) for model integration
- **Workflow Orchestration**: [LangGraph](https://python.langchain.com/docs/langgraph) for structured agent interactions
- **Data Processing**: Pandas for data manipulation
- **Web Scraping**: BeautifulSoup, html2text, requests
- **Concurrent Processing**: asyncio for handling multiple analyses

## Core Business Rules

1. **RRM Framework**: All prospect analyses follow the Route-Ruin-Multiply framework:
   - **Route**: Identify the prospect's primary business objectives
   - **Ruin**: Pinpoint the challenges preventing those objectives
   - **Multiply**: Show how your solution addresses these challenges

2. **Company Context Integration**: Your company's services, products, and value propositions are integrated into all analyses

3. **Data Privacy**: All data remains local; scraped information is only used for analysis

4. **User-Controlled Processing**: Users decide how many contacts to analyze to manage costs

## Development Patterns & Conventions

- **Session State Management**: Prefix `p_` for permanent variables that persist across page navigation
- **Asynchronous Processing**: Concurrent operations for web scraping and AI analysis
- **Component Separation**: Modular design with specialized components for each function
- **Error Handling**: Robust error capture and reporting across all operations
- **Caching**: Search results are cached to reduce API costs and improve performance

## Known Challenges & Solutions

1. **Search Result Processing**: Special handling of search results ensures company mentions are preserved even in truncated content
2. **API Timeouts**: Extended timeouts (90 seconds) for reliable model responses
3. **Model Consistency**: Standardized on OpenAI GPT-4.1 across all components
4. **UI Navigation**: Custom navigation system with permanent session states
5. **Data Field Mapping**: Intelligent fuzzy matching for column identification

## Current Status & Next Steps

The system is fully functional with three main modules:
- **WHAT**: Introduction and setup instructions
- **HUNTER**: Core analysis module for prospect personality profiles
- **SPEAR**: Email generation module using the RRM framework

Future improvements could include:
- Expanded integrations with CRM systems
- Advanced analytics for email performance
- Multi-language support for international prospects
- Team collaboration features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/williavs/hunter-2.git
cd hunter-2
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your API key:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Configure your API key in the sidebar (if not set in `.env` file)

3. Add your company context information:
   - Enter your company website URL and target geography
   - Click "Scrape & Analyze" to automatically generate company context
   - Review and approve the generated context

4. Upload your CSV file containing contact data with website URLs

5. Map the column fields in the dialog (website URL, name fields, etc.)

6. Click "Start Website Scraping" to gather website content

7. Use the slider to select how many contacts to analyze

8. Click "Analyze Personalities" to generate AI-powered insights

9. Navigate to SPEAR to generate personalized emails

10. Download the enriched data as CSV when processing is complete

## Running with Docker

1.  **Build the Docker image manually (optional):**
    ```bash
    docker build -t hunter-app .
    ```

2.  **Easiest way (Recommended, especially for Mac users):**
    1. Make the script executable (first time only):
       ```bash
       chmod +x run_docker.sh
       ```
    2. Run the app:
       ```bash
       ./run_docker.sh
       ```
    3. Open [http://localhost:8501](http://localhost:8501) in your browser.
    4. Enter your API key(s) in the sidebar when prompted.

3.  **For Mac users:**
    - You can also use the included AppleScript to launch the app in Terminal:
      1. Double-click `RunDockerApp.applescript` or open it with Script Editor and run it.
      2. This will open Terminal, navigate to the project directory, and run the Docker script for you.
    - Make sure Docker Desktop is running before you start.

4.  **Note:**
    - API keys need to be entered via the sidebar UI when running via Docker, as the `.env` file is not used.
    - If you encounter permission issues, ensure you have rights to execute scripts and access Docker.

## Technical Documentation

- **OpenAI Web Search**: This project utilizes [OpenAI's Web Search capability](https://platform.openai.com/docs/api-reference/responses/web-search) through the Responses API, allowing models to search the web for the latest information before generating a response.

- **LangChain-OpenAI Integration**: The system leverages [LangChain's OpenAI integration](https://python.langchain.com/docs/integrations/llms/openai) to seamlessly work with OpenAI's models and tools.

- **LangGraph Workflow**: The multi-agent workflow is orchestrated using [LangGraph](https://python.langchain.com/docs/langgraph), which provides a powerful framework for creating stateful agent workflows.

## License

MIT 
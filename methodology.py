import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import os

def show_methodology_page():
    st.title("Technical Methodology")
    
    st.markdown("""
    # AI Sales Intelligence: Technical Architecture & Methodology
    
    This page provides an in-depth technical explanation of the AI systems powering GTM Wizards' sales intelligence platform.
    Designed for technical audiences, data scientists, and AI engineers, this documentation outlines our architecture,
    evaluation framework, and development methodology.
    """)
    
    # Create tabs for different sections
    tabs = st.tabs(["Architecture", "LangGraph Workflow", "Evaluation System", "Model Selection", "Technical Challenges"])
    
    with tabs[0]:
        st.header("System Architecture")
        
        st.markdown("""
        ### High-Level Architecture
        
        The GTM Wizards platform is built on a modular architecture centered around a multi-agent system orchestrated through LangGraph.
        The system processes sales prospect data through several specialized components:
        
        1. **Data Ingestion Layer**: Processes contact information from CSV uploads or API connections
        2. **Personality Analysis Pipeline**: Core AI engine for prospect analysis
        3. **Evaluation & Quality Assurance**: Automated scoring system for analysis quality
        4. **Presentation Layer**: Streamlit-based user interface
        
        ### Core Components
        """)
        
        # Display architecture diagram
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("""
            **Key Components:**
            
            - PersonalityAnalyzer
            - TavilySearchResults
            - ChatOpenRouter
            - LangGraph Workflow
            - Evaluation Framework
            """)
            
        with col2:
            # Create a simple architecture diagram using NetworkX
            G = nx.DiGraph()
            
            # Add nodes
            nodes = ["User Input", "Data Preprocessing", "Search Planning", 
                    "Tavily Search", "Analysis Orchestration", "Personality Analysis", 
                    "Evaluation", "Results Presentation"]
            
            for node in nodes:
                G.add_node(node)
            
            # Add edges
            edges = [
                ("User Input", "Data Preprocessing"),
                ("Data Preprocessing", "Search Planning"),
                ("Search Planning", "Tavily Search"),
                ("Tavily Search", "Analysis Orchestration"),
                ("Analysis Orchestration", "Personality Analysis"),
                ("Personality Analysis", "Evaluation"),
                ("Evaluation", "Results Presentation"),
                ("Results Presentation", "User Input")
            ]
            
            for edge in edges:
                G.add_edge(edge[0], edge[1])
            
            # Plot the graph
            plt.figure(figsize=(10, 6))
            pos = nx.shell_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=2000, edge_color='gray', 
                    font_size=10, font_weight='bold',
                    arrows=True, arrowsize=20)
            
            st.pyplot(plt.gcf())
            plt.close()
        
        st.markdown("""
        ### Technical Stack
        
        - **Frontend**: Streamlit for rapid UI development and data visualization
        - **Backend Processing**: Asynchronous Python with asyncio for concurrent operations
        - **LLM Integration**: LangChain for model integration and prompt management
        - **Workflow Orchestration**: LangGraph for agent workflows and state management
        - **Search Integration**: Tavily API for targeted web searches
        - **Language Models**: Primarily Claude 3.5 via OpenRouter API
        - **Evaluation**: Custom evaluation framework built on LangChain
        """)
        
    with tabs[1]:
        st.header("LangGraph Multi-Agent System")
        
        st.markdown("""
        ### Agent Workflow Design
        
        The core of our system is built on LangGraph, enabling a stateful, multi-step workflow for personality analysis.
        This approach allows for complex reasoning chains with specialized agents handling different aspects of the analysis process.
        """)
        
        st.code("""
# Build the LangGraph workflow for personality analysis
workflow = StateGraph(PersonalityState)
        
# Add nodes for each task
workflow.add_node("planning", self._planning_task)
workflow.add_node("search", self._search_task)
workflow.add_node("analysis_task", self._analysis_task)
        
# Define the edges to create a linear flow
workflow.add_edge("planning", "search")
workflow.add_edge("search", "analysis_task")
workflow.add_edge("analysis_task", END)
        
# Set the entry point
workflow.set_entry_point("planning")
""", language="python")
        
        st.markdown("""
        ### State Management
        
        The PersonalityState maintains context throughout the analysis workflow, capturing:
        
        - Contact information
        - Search queries and results
        - Analysis progress
        - Error logging
        
        This state-based approach allows the system to maintain context across multiple steps and facilitates:
        
        1. **Checkpointing**: The ability to pause/resume analysis
        2. **Debugging**: Transparent workflow visualization
        3. **Persistence**: Maintaining analysis state across sessions
        """)

        
        st.markdown("""
        ### Agent Components
        
        The system consists of three primary specialized components:
        
        1. **Planning Agent**: Generates optimized search queries to gather information about the prospect
        2. **Search Agent**: Executes concurrent searches and processes results
        3. **Analysis Agent**: Synthesizes search results into actionable personality insights
        
        Each component is structured with carefully engineered prompts that include:
        
        - XML-structured instructions for consistent parsing
        - Confidence level indicators for epistemic status
        - Industry-specific reference points
        - Validation mechanisms for assumptions
        """)
        
        # Simple diagram of the LangGraph flow
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create a simple flow diagram
            G = nx.DiGraph()
            nodes = ["Planning\nAgent", "Search\nAgent", "Analysis\nAgent"]
            G.add_nodes_from(nodes)
            edges = [("Planning\nAgent", "Search\nAgent"), 
                    ("Search\nAgent", "Analysis\nAgent")]
            G.add_edges_from(edges)
            
            plt.figure(figsize=(8, 4))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
                    node_size=3000, edge_color='gray', 
                    font_size=12, font_weight='bold',
                    arrows=True, arrowsize=20)
            
            # Add labels for what each agent does
            label_pos = {k: [v[0], v[1]-0.15] for k, v in pos.items()}
            labels = {
                "Planning\nAgent": "Query Generation",
                "Search\nAgent": "Information Retrieval",
                "Analysis\nAgent": "Insight Synthesis"
            }
            nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=10)
            
            st.pyplot(plt.gcf())
            plt.close()
        
        with col2:
            st.markdown("""
            **Key Benefits:**
            
            - Clear separation of concerns
            - Specialized optimization
            - Improved error isolation
            - Enhanced traceability
            - Modular system design
            """)
        
    with tabs[2]:
        st.header("Evaluation Framework")
        
        st.markdown("""
        ### Evaluation Methodology
        
        Our evaluation system employs a multi-dimensional framework to assess the quality and effectiveness of personality analyses. 
        Rather than focusing on theoretical accuracy, our evaluations emphasize practical utility for sales professionals.
        """)
        
        # Create evaluation dimensions visualization
        dimensions = [
            "Practical Relevance",
            "Conversation Guidance",
            "Pain Point Identification",
            "Personality Insight Depth",
            "Company Context Integration",
            "Actionability",
            "Overall Value"
        ]
        
        # Sample scores for visualization
        scores = [4, 4, 5, 4, 5, 4, 4]
        
        # Create radar chart
        categories = dimensions
        N = len(categories)
        
        # Create angle values for the radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Add the scores to the chart (and close the loop)
        scores_for_chart = scores + [scores[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Draw the polygon and the points
        ax.plot(angles, scores_for_chart, linewidth=2, linestyle='solid')
        ax.fill(angles, scores_for_chart, alpha=0.25)
        
        # Set the labels and styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], color='gray', size=10)
        ax.set_ylim(0, 5)
        
        # Add title
        plt.title('Personality Analysis Evaluation Dimensions', size=15, y=1.1)
        
        st.pyplot(fig)
        
        st.markdown("""
        ### Evaluation Dimensions
        
        Each personality analysis is evaluated across seven key dimensions on a scale of 1-5:
        
        1. **Practical Relevance**: How directly applicable the insights are to sales outreach
        2. **Conversation Guidance**: Quality of specific conversational recommendations
        3. **Pain Point Identification**: Accuracy and specificity of identified challenges
        4. **Personality Insight Depth**: Depth and nuance of personality understanding
        5. **Company Context Integration**: Integration of company-specific context 
        6. **Actionability**: Presence of concrete, implementable recommendations
        7. **Overall Value**: Holistic assessment of the analysis's utility
        
        ### Evaluation Implementation
        
        The evaluation system uses a specialized LLM evaluator with:
        
        - Rigid scoring criteria
        - Comparative benchmarking
        - Qualitative feedback sections
        - Specific enhancement recommendations
        
        This system provides continuous feedback that directly improves our analysis prompts and methodology.
        """)
        
        # Show sample evaluation results
        st.subheader("Sample Evaluation Output")
        
        eval_data = {
            "Dimension": dimensions,
            "Score": scores,
            "Comments": [
                "Excellent industry-specific insights",
                "Strong conversational phrases and subject lines",
                "Precise pain point identification with evidence",
                "Good depth with clear confidence levels",
                "Excellent integration with company value proposition",
                "Highly actionable with specific next steps",
                "High overall value for sales engagement"
            ]
        }
        
        st.dataframe(pd.DataFrame(eval_data), use_container_width=True)
        
        st.markdown("""
        ### Continuous Improvement Cycle
        
        Evaluation results directly inform prompt engineering improvements through:
        
        1. **Pattern Identification**: Aggregating common weaknesses
        2. **Prompt Refinement**: Targeted improvements to prompt structure
        3. **Confidence Calibration**: Better handling of epistemic uncertainty
        4. **Evidence Requirements**: Enhanced evidence gathering in search queries
        """)
        
    with tabs[3]:
        st.header("Model Selection & Prompt Engineering")
        
        st.markdown("""
        ### Model Selection Criteria
        
        Our production system primarily uses Claude 3.5 Haiku via the OpenRouter API, selected based on:
        
        1. **Reasoning Quality**: Ability to synthesize complex information
        2. **Emotional Intelligence**: Nuanced understanding of human psychology
        3. **Performance Consistency**: Reliable outputs across different inputs
        4. **Cost Efficiency**: Optimal balance of quality and computation cost
        5. **Prompt Adherence**: Follows structured output requirements
        
        We've tested across multiple models, including GPT-4o, Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 3.7 Sonnet, finding that Claude models generally excel at:
        
        - Following XML-structured prompts
        - Providing consistent confidence assessments
        - Generating actionable sales insights
        """)
        
        
        
        st.markdown("""
        ### Prompt Engineering Methodology
        
        Our prompts are engineered with several key principles:
        
        1. **XML Structure**: Clear delineation of context, objectives, and output requirements
        2. **Confidence Levels**: Explicit tracking of evidence quality for all claims
        3. **Validation Mechanisms**: Built-in methods to verify speculative insights
        4. **Alternative Approaches**: Multiple strategy options for different scenarios
        5. **Industry-Specific Metrics**: Inclusion of relevant benchmarks and measurements
        
        Sample prompt structure:
        """)
        
        st.code("""
<personality_analysis_request>
    <context>
        [Contact information and search results]
    </context>
    
    <analysis_framework>
        <personality_analysis>
            - For each insight, clearly indicate CONFIDENCE LEVEL (High/Medium/Low) and evidence source
            - Explore both analytical AND emotional aspects of their personality
            - For each inferred trait, provide ONE specific validation question a salesperson could ask
        </personality_analysis>

        <conversation_style>
            - Provide 2-3 exact conversational phrases or questions that would resonate
            - Provide ALTERNATIVE approaches for different possible communication preferences
        </conversation_style>
        
        [Additional sections with similar structure]
    </analysis_framework>
</personality_analysis_request>
""", language="xml")
        
        st.markdown("""
        ### Prompt Optimization Process
        
        We follow a systematic approach to prompt optimization:
        
        1. **Baseline Testing**: Initial prompt design based on expert knowledge
        2. **Structured Evaluation**: Assessment across the 7 core dimensions
        3. **Pattern Analysis**: Identification of common weaknesses
        4. **Iterative Refinement**: Targeted improvements to address weakness patterns
        5. **A/B Testing**: Comparative testing of prompt variations
        6. **Production Deployment**: Implementation of optimized prompts
        
        This process has led to significant improvements in practical relevance, conversation guidance, and actionability over multiple iterations.
        """)
        
    with tabs[4]:
        st.header("Technical Challenges & Solutions")
        
        st.markdown("""
        ### Key Technical Challenges
        
        Throughout development, we've addressed several significant technical challenges:
        """)
        
        challenges = [
            {
                "challenge": "Query Generation Quality",
                "description": "Initial search queries were too generic, leading to irrelevant results.",
                "solution": "Implemented structured XML prompts with specific instructions to find evidence of actual behavior and communication styles."
            },
            {
                "challenge": "Evidence Attribution",
                "description": "Early analyses failed to distinguish between evidence-based insights and speculative inferences.",
                "solution": "Added confidence level indicators (High/Medium/Low) with explicit source attribution for all claims."
            },
            {
                "challenge": "Concurrency Management",
                "description": "Processing multiple contacts simultaneously caused API rate limiting and resource exhaustion.",
                "solution": "Implemented semaphore-based concurrency control with configurable limits and intelligent batching."
            },
            {
                "challenge": "Result Consistency",
                "description": "Analysis quality varied significantly between similar inputs.",
                "solution": "Standardized XML-structured prompts with explicit validation requirements for key insights."
            },
            {
                "challenge": "Output Specificity",
                "description": "Generated insights were often too generic to be actionable.",
                "solution": "Added requirements for industry-specific metrics, benchmarks, and concrete validation methods."
            }
        ]
        
        for i, item in enumerate(challenges):
            with st.expander(f"{i+1}. {item['challenge']}"):
                st.markdown(f"**Problem:** {item['description']}")
                st.markdown(f"**Solution:** {item['solution']}")
        
        st.markdown("""
        ### LangGraph Implementation Insights
        
        LangGraph provided significant advantages for our multi-agent system:
        
        1. **State Management**: Maintaining context across agent interactions
        2. **Workflow Visualization**: Clear understanding of processing flow
        3. **Error Isolation**: Containing failures within specific processing steps
        4. **Retry Capabilities**: Graceful handling of transient failures
        5. **Traceability**: Comprehensive logging for debugging and optimization
        
        The key to our implementation was designing clear interfaces between agents and a structured state object that maintains all necessary context.
        """)
        
        st.markdown("""
        ### Future Technical Directions
        
        Based on our learnings, we're exploring several technical enhancements:
        
        1. **Human-in-the-Loop Integration**: Incorporating selective human review in the analysis workflow
        2. **Custom RAG Systems**: Building retrieval-augmented generation for industry-specific knowledge
        3. **Adaptive Prompting**: Dynamically adjusting prompts based on input characteristics
        4. **Multi-Modal Analysis**: Incorporating image analysis for website and social media content
        5. **Feedback Integration**: Closing the loop with salesperson feedback on analysis quality
        """)

# This ensures the page content is displayed when loaded through navigation
if __name__ != "__main__":
    show_methodology_page() 
import streamlit as st

# Only set page config when running this file directly (not through navigation)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Willy VanSickle | HUNTER",
        page_icon="ℹ️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Profile header with enhanced styling
st.markdown("""
<div style="text-align: center; padding: 20px 0; background-color: rgba(255, 87, 34, 0.1); border-radius: 10px; margin-bottom: 25px;">
    <h1 style="color: #FF5722; margin-bottom: 0;">Willy VanSickle</h1>
    <h3 style="color: #9E9E9E; font-weight: 300; margin-top: 5px;">Strategic AI Consulting for Growth</h3>
</div>
""", unsafe_allow_html=True)

# Introduction section with custom quote styling
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ## Turning AI Possibilities into Business Performance
    
    I bridge the gap between AI technology and business outcomes, helping companies implement 
    practical AI solutions that drive measurable results. As a sales expert turned AI engineer, 
    I bring a unique perspective that focuses on revenue-generating applications of artificial 
    intelligence.
    
    My approach combines deep technical expertise with business acumen, ensuring that AI 
    implementations deliver real value rather than just technological novelty.
    """)
    
    st.markdown("""
    ### Expertise Areas
    
    - **AI Strategy Development**
    - **Sales & GTM Optimization**
    - **Custom AI Application Development**
    - **LLM Systems & Agent Architecture**
    - **AI-Powered Growth Hacking**
    """)

with col2:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-left: 3px solid #FF5722; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <p style="font-style: italic; color: #E0E0E0;">
        "After talking to many firms, I have come to believe a key factor in successful AI adoption is whether the executive team actually experiments with AI to try to get work done themselves. Those who do tend to feel urgency and push for transformation."
        </p>
        <p style="text-align: right; color: #9E9E9E;">— Ethan Mollick, Professor at The Wharton School</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: rgba(255, 87, 34, 0.1); border-radius: 5px; padding: 15px; margin-top: 20px; text-align: center;">
        <p style="margin-bottom: 10px;"><strong>Contact Me</strong></p>
        <p style="margin: 5px 0;"><a href="mailto:willyv3@v3-ai.com" style="color: #FF5722; text-decoration: none;">willyv3@v3-ai.com</a></p>
    </div>
    """, unsafe_allow_html=True)

# Professional background section
st.markdown("## Professional Background", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Sales & GTM Expert
    **10+ years crushing sales targets**
    
    - Pipeline mastery through AI-driven lead scoring
    - Data-driven GTM strategy development
    - Automated sales systems and workflows
    - Growth hacking with AI tools
    """)

with col2:
    st.markdown("""
    ### AI Engineer
    **Revenue-driving AI solutions**
    
    - Custom AI application development
    - LLM integration and agent systems
    - AI workflow automation
    - Technical implementation expertise
    """)

with col3:
    st.markdown("""
    ### Community Leader
    **Human-first tech solutions**
    
    - Knowledge sharing and education
    - Translating technical concepts
    - Building AI literacy
    - Ethical AI implementation
    """)

# Featured projects section
st.markdown("## App Playground", unsafe_allow_html=True)
st.markdown("""
Explore my suite of AI-powered tools designed to help you work smarter. These applications demonstrate 
practical AI implementations that deliver immediate value.
""")

proj1, proj2 = st.columns(2)

with proj1:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; height: 100%;">
        <h3 style="color: #FF5722;">V3 Discourse Engine</h3>
        <p style="color: #9E9E9E; font-size: 14px;">AI-POWERED DEBATE & ANALYSIS PLATFORM</p>
        <p>Analyze topics from different angles through structured debate with research-backed arguments and balanced perspectives. Perfect for decision-making, research, and understanding complex topics.</p>
        <p><a href="https://willyv3.com/app-playground" style="color: #FF5722; text-decoration: none;">Try It Now →</a></p>
    </div>
    """, unsafe_allow_html=True)

with proj2:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; height: 100%;">
        <h3 style="color: #FF5722;">LinkedIn Carousel Maker</h3>
        <p style="color: #9E9E9E; font-size: 14px;">FREE CONTENT CREATION TOOL</p>
        <p>Create professional LinkedIn carousels in minutes with this free, easy-to-use tool. No design skills required. Perfect for building your personal brand or company presence.</p>
        <p><a href="https://willy-carousel.netlify.app/" style="color: #FF5722; text-decoration: none;">Create Carousels Now →</a></p>
    </div>
    """, unsafe_allow_html=True)

proj3, proj4 = st.columns(2)

with proj3:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; height: 100%;">
        <h3 style="color: #FF5722;">V3 Biz Finder</h3>
        <p style="color: #9E9E9E; font-size: 14px;">BUSINESS SEARCH & LEAD GENERATION</p>
        <p>Powerful local business search tool that helps you find and aggregate business information for lead generation and market research.</p>
        <p><a href="https://willyv3.com/app-playground" style="color: #FF5722; text-decoration: none;">Try It Now →</a></p>
    </div>
    """, unsafe_allow_html=True)

with proj4:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; height: 100%;">
        <h3 style="color: #FF5722;">GTM Context Builder</h3>
        <p style="color: #9E9E9E; font-size: 14px;">SALES INTELLIGENCE PLATFORM</p>
        <p>Enhance your go-to-market approach with AI-powered context building and personalization tools that improve targeted outreach.</p>
        <p><a href="https://willyv3.com/app-playground" style="color: #FF5722; text-decoration: none;">Try It Now →</a></p>
    </div>
    """, unsafe_allow_html=True)

# Featured workshops section
st.markdown("## AI Workshops", unsafe_allow_html=True)
st.markdown("""
Hands-on learning experiences to master AI development and implementation. Join me for practical
workshops that bridge the gap between AI potential and business applications.
""")

# Workshops section with actual data
workshops1, workshops2 = st.columns(2)

with workshops1:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; height: 100%;">
        <h3 style="color: #FF5722;">Building an AI Knowledge Assistant for Slack with RAG</h3>
        <p style="color: #9E9E9E; font-size: 14px;">INTERMEDIATE WORKSHOP | 3.5 HOURS</p>
        <p>Create a production-ready Slack bot that leverages RAG for intelligent document search and responses. Learn practical RAG architecture implementation that delivers immediate business value.</p>
        <p><strong>Key Topics:</strong> RAG Architecture, Document Processing, Vector Embeddings, Slack API Integration</p>
        <p><a href="https://willyv3.com/workshops" style="color: #FF5722; text-decoration: none;">Apply Now →</a></p>
    </div>
    """, unsafe_allow_html=True)

with workshops2:
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; height: 100%;">
        <h3 style="color: #FF5722;">Setting Up Your Mac for AI-Powered Development</h3>
        <p style="color: #9E9E9E; font-size: 14px;">BEGINNER WORKSHOP | 2 HOURS</p>
        <p>Master Cursor and Roo Code for enhanced coding productivity. Learn to configure these powerful AI coding tools and implement effective prompting techniques.</p>
        <p><strong>Key Topics:</strong> AI-assisted development workflows, Cursor configuration, Effective prompting for code generation</p>
        <p><a href="https://willyv3.com/workshops" style="color: #FF5722; text-decoration: none;">Apply Now →</a></p>
    </div>
    """, unsafe_allow_html=True)

# My approach section
st.markdown("## My Approach", unsafe_allow_html=True)
st.markdown("""
Transforming AI possibilities into business solutions with a proven three-step process:
""")

approach1, approach2, approach3 = st.columns(3)

with approach1:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: rgba(255, 87, 34, 0.1); border-radius: 10px;">
        <h3 style="color: #FF5722;">1. Listen & Learn</h3>
        <p>Deep dive into your business challenges through focused discovery sessions</p>
        <ul style="text-align: left;">
            <li>In-depth business analysis</li>
            <li>Challenge identification</li>
            <li>Opportunity mapping</li>
            <li>Success metrics definition</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with approach2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: rgba(255, 87, 34, 0.1); border-radius: 10px;">
        <h3 style="color: #FF5722;">2. Prototype & Prove</h3>
        <p>Rapid development of AI solutions with measurable impact</p>
        <ul style="text-align: left;">
            <li>Quick proof of concepts</li>
            <li>Iterative refinement</li>
            <li>Performance metrics</li>
            <li>ROI validation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with approach3:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: rgba(255, 87, 34, 0.1); border-radius: 10px;">
        <h3 style="color: #FF5722;">3. Scale & Strategize</h3>
        <p>Expand successful solutions across your organization with proven frameworks</p>
        <ul style="text-align: left;">
            <li>Enterprise integration</li>
            <li>Team enablement</li>
            <li>Process optimization</li>
            <li>Growth acceleration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Call to action section with real links
st.markdown("""
<div style="text-align: center; margin: 40px 0; padding: 30px; background-color: rgba(255, 87, 34, 0.1); border-radius: 10px;">
    <h2 style="color: #FF5722; margin-bottom: 20px;">Ready to Transform Your AI Vision?</h2>
    <p style="font-size: 18px; margin-bottom: 20px;">Let's work together to implement AI solutions that drive real business results.</p>
    <p>
        <a href="mailto:willyv3@v3-ai.com" style="display: inline-block; margin: 10px; padding: 10px 20px; background-color: #FF5722; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">Contact Me</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Footer section with real links
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <p style="color: #9E9E9E; font-size: 14px;">© 2025 V3Consult. All rights reserved.</p>
    <p style="color: #9E9E9E; font-size: 14px;">
        <a href="https://willyv3.com" style="color: #FF5722; text-decoration: none; margin: 0 10px;">Website</a> |
        <a href="https://linkedin.com" style="color: #FF5722; text-decoration: none; margin: 0 10px;">LinkedIn</a> |
        <a href="https://willyv3.com/workshops" style="color: #FF5722; text-decoration: none; margin: 0 10px;">Workshops</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Run the page if this file is run directly  
if __name__ == "__main__":
    pass 
"""
Advanced RAG Techniques MVP - Home Page
A comprehensive demonstration of Advanced RAG techniques from the Weaviate ebook
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers_basic import create_overview_metrics
from data.rag_techniques import RAG_TECHNIQUES

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Techniques MVP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .feature-highlight {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Advanced RAG Techniques MVP</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        A comprehensive demonstration of Advanced RAG techniques featuring multiple vector databases,
        interactive demos, and downloadable insights.
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>13</h3>
            <p>Total Techniques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>3</h3>
            <p>Vector Databases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>4</h3>
            <p>Main Categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>12+</h3>
            <p>Interactive Demos</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Guide
    st.markdown("## 🧭 Navigation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📊 Indexing Optimization</h3>
            <p>Explore data preprocessing and chunking strategies with interactive demos for text cleaning and document segmentation.</p>
            <div class="feature-highlight">
                <strong>Features:</strong> Text cleaning, chunking strategies, interactive demos
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <h3>🔄 Retrieval Optimization</h3>
            <p>Discover metadata filtering, hybrid search, and vector search optimization techniques.</p>
            <div class="feature-highlight">
                <strong>Features:</strong> Hybrid search, metadata filtering, performance comparison
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 Pre-retrieval Optimization</h3>
            <p>Learn query decomposition, routing, and transformation techniques with visual demonstrations.</p>
            <div class="feature-highlight">
                <strong>Features:</strong> Query decomposition, routing visualization, transformation
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <h3>⚡ Post-retrieval Optimization</h3>
            <p>Master re-ranking, context processing, prompt engineering, and LLM fine-tuning.</p>
            <div class="feature-highlight">
                <strong>Features:</strong> Re-ranking, context processing, prompt engineering
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Vector Database Status
    st.markdown("## 🗄️ Vector Database Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h4>Faiss</h4>
            <p><span style="color: green;">●</span> Ready</p>
            <p>CPU-based similarity search library by Facebook AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h4>ChromaDB</h4>
            <p><span style="color: green;">●</span> Ready</p>
            <p>Open-source embedding database</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="section-card">
            <h4>Weaviate</h4>
            <p><span style="color: orange;">●</span> Needs Setup</p>
            <p>Cloud-native vector database (requires credentials)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Techniques Overview
    st.markdown("## 📋 Techniques Overview")
    
    # Create overview chart
    categories = ["Indexing", "Pre-retrieval", "Retrieval", "Post-retrieval"]
    technique_counts = [3, 3, 3, 4]  # Based on our implementation
    
    fig = px.bar(
        x=categories,
        y=technique_counts,
        title="RAG Techniques by Category",
        labels={"x": "Category", "y": "Number of Techniques"},
        color=technique_counts,
        color_continuous_scale="viridis"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Getting Started
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    <div class="section-card">
        <h4>How to Use This Application</h4>
        <ol>
            <li><strong>Navigate</strong> using the sidebar to explore different RAG technique categories</li>
            <li><strong>Interact</strong> with demos to see techniques in action</li>
            <li><strong>Download</strong> data tables and results for further analysis</li>
            <li><strong>Compare</strong> vector database performance and capabilities</li>
            <li><strong>Experiment</strong> with different parameters and configurations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        Built with ❤️ using Streamlit | Based on Weaviate's Advanced RAG Techniques ebook
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


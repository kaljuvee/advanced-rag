"""
Indexing Optimization Techniques Page
Demonstrates data preprocessing and chunking strategies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers_basic import (
    clean_text, apply_chunking, create_chunking_visualization,
    create_chunk_size_distribution
)
from data.rag_techniques import RAG_TECHNIQUES

st.set_page_config(page_title="Indexing Optimization", page_icon="📊", layout="wide")

st.title("📊 Indexing Optimization Techniques")

st.markdown("""
Indexing optimization techniques focus on preparing and structuring data before it enters the RAG system. 
These techniques ensure that documents are properly processed, cleaned, and segmented for optimal retrieval performance.
""")

# Techniques Overview
st.markdown("## 📋 Indexing Techniques Overview")

indexing_data = RAG_TECHNIQUES.get("indexing_optimization", {})
if indexing_data and "techniques" in indexing_data:
    techniques_data = []
    for technique_id, technique in indexing_data["techniques"].items():
        techniques_data.append({
            "Technique": technique.get("title", technique_id),
            "Description": technique.get("description", ""),
            "Category": "Indexing Optimization",
            "Complexity": "Medium"
        })
    
    df = pd.DataFrame(techniques_data)
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Indexing Techniques",
        data=csv,
        file_name="indexing_techniques.csv",
        mime="text/csv"
    )

st.markdown("---")

# Interactive Demos
st.markdown("## 🛠️ Interactive Demos")

tab1, tab2 = st.tabs(["📝 Text Cleaning", "✂️ Chunking Strategies"])

with tab1:
    st.markdown("### Text Cleaning Demo")
    st.markdown("Clean and preprocess text data for better indexing performance.")
    
    # Sample text
    sample_text = st.text_area(
        "Input Text:",
        value="""This is a SAMPLE text with    extra spaces, special characters @#$%, 
        and inconsistent formatting.
        It also contains URLs like https://example.com and email@domain.com.
        
        There are multiple    spaces and 
        line breaks that need cleaning.""",
        height=150
    )
    
    # Cleaning options
    col1, col2, col3 = st.columns(3)
    with col1:
        remove_special = st.checkbox("Remove special characters", value=True)
    with col2:
        normalize_spaces = st.checkbox("Normalize whitespace", value=True)
    with col3:
        remove_urls = st.checkbox("Remove URLs/emails", value=True)
    
    if st.button("Clean Text"):
        cleaned = clean_text(sample_text, remove_special, normalize_spaces, remove_urls)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Text:**")
            st.text_area("", value=sample_text, height=200, disabled=True)
        
        with col2:
            st.markdown("**Cleaned Text:**")
            st.text_area("", value=cleaned, height=200, disabled=True)
        
        # Statistics
        st.markdown("### Cleaning Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Length", len(sample_text))
        with col2:
            st.metric("Cleaned Length", len(cleaned))
        with col3:
            reduction = ((len(sample_text) - len(cleaned)) / len(sample_text)) * 100
            st.metric("Reduction", f"{reduction:.1f}%")

with tab2:
    st.markdown("### Chunking Strategies Demo")
    st.markdown("Compare different document chunking approaches.")
    
    # Sample document
    sample_doc = st.text_area(
        "Document Text:",
        value="""Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so.

Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.""",
        height=200
    )
    
    # Chunking parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.slider("Chunk Size (characters)", 50, 500, 200)
    with col2:
        overlap = st.slider("Overlap (characters)", 0, 100, 20)
    with col3:
        method = st.selectbox("Chunking Method", ["Fixed Size", "Recursive", "Sentence-based"])
    
    if st.button("Apply Chunking"):
        chunks = apply_chunking(sample_doc, chunk_size, overlap, method)
        
        # Display chunks
        st.markdown(f"### Generated Chunks ({len(chunks)} total)")
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1} ({len(chunk)} characters)"):
                st.text(chunk)
        
        # Visualization
        fig = create_chunking_visualization(chunks)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download chunks
        chunks_df = pd.DataFrame({
            "Chunk_ID": range(1, len(chunks) + 1),
            "Content": chunks,
            "Length": [len(chunk) for chunk in chunks]
        })
        
        csv = chunks_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Chunks",
            data=csv,
            file_name="document_chunks.csv",
            mime="text/csv"
        )

st.markdown("---")

# Best Practices
st.markdown("## 💡 Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Text Cleaning
    - Remove irrelevant characters and formatting
    - Normalize whitespace and encoding
    - Handle special cases (URLs, emails, etc.)
    - Preserve semantic meaning
    """)

with col2:
    st.markdown("""
    ### Chunking Strategy
    - Balance chunk size with context preservation
    - Use appropriate overlap for continuity
    - Consider document structure and semantics
    - Test different methods for your use case
    """)

# Footer
st.markdown("---")
st.markdown("*Navigate to other sections using the sidebar to explore more RAG techniques.*")


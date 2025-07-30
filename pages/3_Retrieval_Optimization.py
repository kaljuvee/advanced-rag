"""
Retrieval Optimization Strategies Page
Demonstrates metadata filtering, hybrid search, and vector optimization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers_basic import (
    apply_metadata_filter, perform_hybrid_search, 
    create_search_comparison_viz, create_filter_effectiveness_viz
)
from data.rag_techniques import RAG_TECHNIQUES

st.set_page_config(page_title="Retrieval Optimization", page_icon="🔄", layout="wide")

st.title("🔄 Retrieval Optimization Strategies")

st.markdown("""
Retrieval optimization strategies focus on improving the actual document retrieval process. 
These techniques enhance search accuracy, combine multiple search methods, and filter results effectively.
""")

# Techniques Overview
st.markdown("## 📋 Retrieval Optimization Techniques")

retrieval_techniques = RAG_TECHNIQUES.get("retrieval_optimization", {})
if retrieval_techniques:
    techniques_data = []
    for technique_id, technique in retrieval_techniques.items():
        techniques_data.append({
            "Technique": technique["name"],
            "Description": technique["description"],
            "Use Case": technique["use_case"],
            "Complexity": technique.get("complexity", "Medium")
        })
    
    df = pd.DataFrame(techniques_data)
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Retrieval Techniques",
        data=csv,
        file_name="retrieval_techniques.csv",
        mime="text/csv"
    )

st.markdown("---")

# Interactive Demos
st.markdown("## 🛠️ Interactive Demos")

tab1, tab2, tab3 = st.tabs(["🔍 Metadata Filtering", "🔄 Hybrid Search", "📊 Performance Comparison"])

with tab1:
    st.markdown("### Metadata Filtering Demo")
    st.markdown("Filter documents based on metadata criteria for more targeted retrieval.")
    
    # Sample documents with metadata
    sample_docs = [
        {"title": "Introduction to Machine Learning", "category": "AI/ML", "year": 2023, "author": "Dr. Smith", "difficulty": "Beginner"},
        {"title": "Deep Learning Fundamentals", "category": "Deep Learning", "year": 2022, "author": "Prof. Johnson", "difficulty": "Intermediate"},
        {"title": "Natural Language Processing", "category": "NLP", "year": 2023, "author": "Dr. Brown", "difficulty": "Advanced"},
        {"title": "Computer Vision Applications", "category": "Computer Vision", "year": 2021, "author": "Dr. Davis", "difficulty": "Intermediate"},
        {"title": "Reinforcement Learning Basics", "category": "Reinforcement Learning", "year": 2023, "author": "Prof. Wilson", "difficulty": "Beginner"}
    ]
    
    # Display all documents
    st.markdown("**Available Documents:**")
    docs_df = pd.DataFrame(sample_docs)
    st.dataframe(docs_df, use_container_width=True)
    
    # Filter controls
    st.markdown("**Apply Filters:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.multiselect(
            "Category:",
            options=["AI/ML", "Deep Learning", "NLP", "Computer Vision", "Reinforcement Learning"],
            default=[]
        )
    
    with col2:
        year_filter = st.multiselect(
            "Year:",
            options=[2021, 2022, 2023],
            default=[]
        )
    
    with col3:
        difficulty_filter = st.multiselect(
            "Difficulty:",
            options=["Beginner", "Intermediate", "Advanced"],
            default=[]
        )
    
    if st.button("Apply Filters"):
        filtered_docs = apply_metadata_filter(sample_docs, category_filter, year_filter, difficulty_filter)
        
        st.markdown(f"### Filtered Results ({len(filtered_docs)} documents)")
        if filtered_docs:
            filtered_df = pd.DataFrame(filtered_docs)
            st.dataframe(filtered_df, use_container_width=True)
            
            # Visualization
            fig = create_filter_effectiveness_viz(sample_docs, filtered_docs)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download filtered results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Results",
                data=csv,
                file_name="filtered_documents.csv",
                mime="text/csv"
            )
        else:
            st.warning("No documents match the selected filters.")

with tab2:
    st.markdown("### Hybrid Search Demo")
    st.markdown("Combine vector search with keyword search for improved retrieval accuracy.")
    
    # Search query
    search_query = st.text_input("Search Query:", "machine learning neural networks")
    
    # Search method weights
    col1, col2 = st.columns(2)
    with col1:
        vector_weight = st.slider("Vector Search Weight", 0.0, 1.0, 0.7, 0.1)
    with col2:
        keyword_weight = st.slider("Keyword Search Weight", 0.0, 1.0, 0.3, 0.1)
    
    # Normalize weights
    total_weight = vector_weight + keyword_weight
    if total_weight > 0:
        vector_weight = vector_weight / total_weight
        keyword_weight = keyword_weight / total_weight
    
    if st.button("Search Documents"):
        search_results = perform_hybrid_search(search_query, vector_weight, keyword_weight)
        
        st.markdown("### Search Results")
        
        # Display results
        for i, result in enumerate(search_results, 1):
            with st.expander(f"Result {i}: {result['title']} (Score: {result['score']:.3f})"):
                st.write(f"**Category:** {result['category']}")
                st.write(f"**Content:** {result['content']}")
                st.write(f"**Vector Score:** {result['vector_score']:.3f}")
                st.write(f"**Keyword Score:** {result['keyword_score']:.3f}")
        
        # Visualization
        fig = create_search_comparison_viz(search_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        results_df = pd.DataFrame(search_results)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Search Results",
            data=csv,
            file_name="hybrid_search_results.csv",
            mime="text/csv"
        )

with tab3:
    st.markdown("### Performance Comparison")
    st.markdown("Compare different retrieval methods and their effectiveness.")
    
    # Performance metrics
    methods = ["Vector Search", "Keyword Search", "Hybrid Search", "Filtered Search"]
    precision = [0.85, 0.72, 0.91, 0.88]
    recall = [0.78, 0.85, 0.89, 0.82]
    f1_score = [0.81, 0.78, 0.90, 0.85]
    
    performance_df = pd.DataFrame({
        "Method": methods,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    })
    
    st.dataframe(performance_df, use_container_width=True)
    
    # Visualization
    fig = px.bar(
        performance_df.melt(id_vars=["Method"], var_name="Metric", value_name="Score"),
        x="Method",
        y="Score",
        color="Metric",
        title="Retrieval Method Performance Comparison",
        barmode="group"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download performance data
    csv = performance_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Performance Data",
        data=csv,
        file_name="retrieval_performance.csv",
        mime="text/csv"
    )

st.markdown("---")

# Best Practices
st.markdown("## 💡 Best Practices")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Metadata Filtering
    - Design comprehensive metadata schemas
    - Index metadata efficiently
    - Use appropriate filter combinations
    - Consider filter selectivity
    """)

with col2:
    st.markdown("""
    ### Hybrid Search
    - Balance vector and keyword weights
    - Normalize scores appropriately
    - Consider query characteristics
    - Test different combination strategies
    """)

with col3:
    st.markdown("""
    ### Performance Optimization
    - Monitor retrieval metrics regularly
    - A/B test different approaches
    - Optimize for your specific use case
    - Consider computational costs
    """)

# Footer
st.markdown("---")
st.markdown("*Navigate to other sections using the sidebar to explore more RAG techniques.*")


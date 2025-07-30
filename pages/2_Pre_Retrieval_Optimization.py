"""
Pre-retrieval Optimization Techniques Page
Demonstrates query transformation and routing strategies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers_basic import (
    decompose_query, route_query, create_query_decomposition_viz,
    create_query_routing_viz
)
from data.rag_techniques import RAG_TECHNIQUES

st.set_page_config(page_title="Pre-retrieval Optimization", page_icon="🎯", layout="wide")

st.title("🎯 Pre-retrieval Optimization Techniques")

st.markdown("""
Pre-retrieval optimization techniques enhance queries before they are used for document retrieval. 
These techniques improve query quality, decompose complex queries, and route queries to appropriate systems.
""")

# Techniques Overview
st.markdown("## 📋 Pre-retrieval Techniques Overview")

preretrieval_data = RAG_TECHNIQUES.get("pre_retrieval_optimization", {})
if preretrieval_data and "techniques" in preretrieval_data:
    techniques_data = []
    for technique_id, technique in preretrieval_data["techniques"].items():
        techniques_data.append({
            "Technique": technique.get("title", technique_id),
            "Description": technique.get("description", ""),
            "Category": "Pre-retrieval Optimization",
            "Complexity": "Medium"
        })
    
    df = pd.DataFrame(techniques_data)
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Pre-retrieval Techniques",
        data=csv,
        file_name="preretrieval_techniques.csv",
        mime="text/csv"
    )

st.markdown("---")

# Interactive Demos
st.markdown("## 🛠️ Interactive Demos")

tab1, tab2, tab3 = st.tabs(["🔍 Query Decomposition", "🎯 Query Routing", "✨ Query Enhancement"])

with tab1:
    st.markdown("### Query Decomposition Demo")
    st.markdown("Break down complex queries into simpler sub-queries for better retrieval.")
    
    # Sample queries
    sample_queries = [
        "What are the differences between supervised and unsupervised learning in machine learning?",
        "How do neural networks work and what are their applications in computer vision?",
        "Compare the performance of different database systems for large-scale applications"
    ]
    
    selected_query = st.selectbox("Select a sample query:", sample_queries)
    custom_query = st.text_input("Or enter your own query:", "")
    
    query_to_process = custom_query if custom_query else selected_query
    
    if st.button("Decompose Query"):
        decomposition = decompose_query(query_to_process)
        
        st.markdown("### Decomposition Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Query:**")
            st.info(query_to_process)
        
        with col2:
            st.markdown("**Sub-queries:**")
            for i, sub_query in enumerate(decomposition["sub_queries"], 1):
                st.success(f"{i}. {sub_query}")
        
        # Visualization
        fig = create_query_decomposition_viz(decomposition)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        decomp_df = pd.DataFrame({
            "Original_Query": [query_to_process],
            "Sub_Query_1": [decomposition["sub_queries"][0] if len(decomposition["sub_queries"]) > 0 else ""],
            "Sub_Query_2": [decomposition["sub_queries"][1] if len(decomposition["sub_queries"]) > 1 else ""],
            "Sub_Query_3": [decomposition["sub_queries"][2] if len(decomposition["sub_queries"]) > 2 else ""],
            "Complexity_Score": [decomposition["complexity"]]
        })
        
        csv = decomp_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Decomposition Results",
            data=csv,
            file_name="query_decomposition.csv",
            mime="text/csv"
        )

with tab2:
    st.markdown("### Query Routing Demo")
    st.markdown("Route queries to the most appropriate retrieval system or database.")
    
    # Sample queries for routing
    routing_queries = [
        "What is the latest stock price of Apple?",
        "Explain the concept of machine learning",
        "Show me recent news about artificial intelligence",
        "How do I implement a neural network in Python?"
    ]
    
    selected_routing_query = st.selectbox("Select a query for routing:", routing_queries)
    custom_routing_query = st.text_input("Or enter your own routing query:", "")
    
    routing_query = custom_routing_query if custom_routing_query else selected_routing_query
    
    if st.button("Route Query"):
        routing_result = route_query(routing_query)
        
        st.markdown("### Routing Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Query:**")
            st.info(routing_query)
            st.markdown("**Recommended System:**")
            st.success(routing_result["recommended_system"])
        
        with col2:
            st.markdown("**Routing Scores:**")
            for system, score in routing_result["scores"].items():
                st.metric(system, f"{score:.2f}")
        
        # Visualization
        fig = create_query_routing_viz(routing_result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        routing_df = pd.DataFrame({
            "Query": [routing_query],
            "Recommended_System": [routing_result["recommended_system"]],
            "Vector_DB_Score": [routing_result["scores"]["Vector DB"]],
            "Knowledge_Graph_Score": [routing_result["scores"]["Knowledge Graph"]],
            "SQL_Database_Score": [routing_result["scores"]["SQL Database"]],
            "Web_Search_Score": [routing_result["scores"]["Web Search"]]
        })
        
        csv = routing_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Routing Results",
            data=csv,
            file_name="query_routing.csv",
            mime="text/csv"
        )

with tab3:
    st.markdown("### Query Enhancement Demo")
    st.markdown("Enhance queries with synonyms, context, and semantic expansion.")
    
    enhancement_query = st.text_input("Enter query to enhance:", "machine learning algorithms")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        add_synonyms = st.checkbox("Add synonyms", value=True)
    with col2:
        add_context = st.checkbox("Add context", value=True)
    with col3:
        expand_semantically = st.checkbox("Semantic expansion", value=True)
    
    if st.button("Enhance Query"):
        enhanced_query = enhancement_query
        
        if add_synonyms:
            enhanced_query += " (AI, artificial intelligence, ML)"
        if add_context:
            enhanced_query += " in computer science and data analysis"
        if expand_semantically:
            enhanced_query += " including supervised learning, unsupervised learning, neural networks"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Query:**")
            st.info(enhancement_query)
        
        with col2:
            st.markdown("**Enhanced Query:**")
            st.success(enhanced_query)
        
        # Enhancement metrics
        st.markdown("### Enhancement Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Length", len(enhancement_query.split()))
        with col2:
            st.metric("Enhanced Length", len(enhanced_query.split()))
        with col3:
            expansion_ratio = len(enhanced_query.split()) / len(enhancement_query.split())
            st.metric("Expansion Ratio", f"{expansion_ratio:.1f}x")

st.markdown("---")

# Best Practices
st.markdown("## 💡 Best Practices")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Query Decomposition
    - Identify key concepts and relationships
    - Break complex queries into atomic parts
    - Maintain logical dependencies
    - Consider query complexity metrics
    """)

with col2:
    st.markdown("""
    ### Query Routing
    - Analyze query intent and type
    - Consider data source characteristics
    - Use confidence scoring
    - Implement fallback mechanisms
    """)

with col3:
    st.markdown("""
    ### Query Enhancement
    - Add relevant synonyms and variants
    - Include domain-specific context
    - Balance expansion with precision
    - Test enhancement effectiveness
    """)

# Footer
st.markdown("---")
st.markdown("*Navigate to other sections using the sidebar to explore more RAG techniques.*")


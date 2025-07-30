"""
Vector Database Comparison Page
Demonstrates and compares Faiss, ChromaDB, and Weaviate
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.vector_db_utils import (
    get_database_status, benchmark_databases, 
    create_performance_comparison, create_similarity_heatmap
)

st.set_page_config(page_title="Vector Database Comparison", page_icon="🗄️", layout="wide")

st.title("🗄️ Vector Database Comparison")

st.markdown("""
Compare popular vector databases for RAG applications. This section demonstrates the capabilities, 
performance, and use cases of Faiss, ChromaDB, and Weaviate.
""")

# Database Status
st.markdown("## 📊 Database Status")

status = get_database_status()
status_data = []
for db_name, info in status.items():
    status_data.append({
        "Database": db_name,
        "Status": info["status"],
        "Available": "✅" if info["available"] else "❌",
        "Description": info["description"]
    })

status_df = pd.DataFrame(status_data)
st.dataframe(status_df, use_container_width=True)

# Download status
csv = status_df.to_csv(index=False)
st.download_button(
    label="📥 Download Database Status",
    data=csv,
    file_name="vector_db_status.csv",
    mime="text/csv"
)

st.markdown("---")

# Database Features Comparison
st.markdown("## 🔍 Feature Comparison")

features_data = [
    {
        "Feature": "Deployment",
        "Faiss": "Local/Embedded",
        "ChromaDB": "Local/Cloud",
        "Weaviate": "Cloud-native"
    },
    {
        "Feature": "Scalability",
        "Faiss": "High (CPU)",
        "ChromaDB": "Medium",
        "Weaviate": "Very High"
    },
    {
        "Feature": "Ease of Use",
        "Faiss": "Medium",
        "ChromaDB": "High",
        "Weaviate": "High"
    },
    {
        "Feature": "Metadata Support",
        "Faiss": "Limited",
        "ChromaDB": "Full",
        "Weaviate": "Full"
    },
    {
        "Feature": "Real-time Updates",
        "Faiss": "Manual",
        "ChromaDB": "Automatic",
        "Weaviate": "Automatic"
    },
    {
        "Feature": "Cost",
        "Faiss": "Free",
        "ChromaDB": "Free/Paid",
        "Weaviate": "Paid"
    }
]

features_df = pd.DataFrame(features_data)
st.dataframe(features_df, use_container_width=True)

# Download features comparison
csv = features_df.to_csv(index=False)
st.download_button(
    label="📥 Download Feature Comparison",
    data=csv,
    file_name="vector_db_features.csv",
    mime="text/csv"
)

st.markdown("---")

# Interactive Benchmark
st.markdown("## 🚀 Performance Benchmark")

st.markdown("""
Run a benchmark comparison across all available vector databases using sample documents.
""")

# Benchmark controls
col1, col2 = st.columns(2)
with col1:
    benchmark_query = st.text_input("Benchmark Query:", "machine learning algorithms")
with col2:
    num_results = st.slider("Number of Results:", 1, 5, 3)

if st.button("Run Benchmark"):
    with st.spinner("Running benchmark across all databases..."):
        benchmark_results = benchmark_databases(benchmark_query)
    
    st.markdown("### Benchmark Results")
    
    # Results summary
    summary_data = []
    for db_name, result in benchmark_results.items():
        summary_data.append({
            "Database": db_name,
            "Status": result["status"],
            "Execution Time (s)": f"{result['execution_time']:.3f}",
            "Results Found": len(result["results"])
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Performance visualization
    successful_results = {k: v["results"] for k, v in benchmark_results.items() if v["status"] == "Success"}
    
    if successful_results:
        fig = create_performance_comparison(successful_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results for each database
        for db_name, results in successful_results.items():
            if results:
                st.markdown(f"#### {db_name} Results")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Similarity heatmap
                fig_heatmap = create_similarity_heatmap(results)
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Download benchmark results
    all_results = []
    for db_name, result in benchmark_results.items():
        for doc in result["results"]:
            doc_result = doc.copy()
            doc_result["database"] = db_name
            doc_result["execution_time"] = result["execution_time"]
            all_results.append(doc_result)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Benchmark Results",
            data=csv,
            file_name="vector_db_benchmark.csv",
            mime="text/csv"
        )

st.markdown("---")

# Use Case Recommendations
st.markdown("## 💡 Use Case Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Faiss
    **Best for:**
    - High-performance similarity search
    - CPU-based deployments
    - Research and prototyping
    - Large-scale batch processing
    
    **Considerations:**
    - Requires manual index management
    - Limited metadata support
    - No built-in persistence
    """)

with col2:
    st.markdown("""
    ### ChromaDB
    **Best for:**
    - Rapid prototyping
    - Small to medium datasets
    - Local development
    - Simple deployment requirements
    
    **Considerations:**
    - Good balance of features and simplicity
    - Active development community
    - Suitable for most RAG applications
    """)

with col3:
    st.markdown("""
    ### Weaviate
    **Best for:**
    - Production deployments
    - Large-scale applications
    - Complex metadata requirements
    - Enterprise features needed
    
    **Considerations:**
    - Requires cloud setup or self-hosting
    - More complex configuration
    - Commercial licensing for some features
    """)

st.markdown("---")

# Setup Instructions
st.markdown("## ⚙️ Setup Instructions")

tab1, tab2, tab3 = st.tabs(["Faiss Setup", "ChromaDB Setup", "Weaviate Setup"])

with tab1:
    st.markdown("""
    ### Faiss Installation
    ```bash
    # CPU version
    pip install faiss-cpu
    
    # GPU version (if CUDA available)
    pip install faiss-gpu
    ```
    
    ### Basic Usage
    ```python
    import faiss
    import numpy as np
    
    # Create index
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors
    vectors = np.random.random((1000, dimension)).astype('float32')
    index.add(vectors)
    
    # Search
    query = np.random.random((1, dimension)).astype('float32')
    distances, indices = index.search(query, k=5)
    ```
    """)

with tab2:
    st.markdown("""
    ### ChromaDB Installation
    ```bash
    pip install chromadb
    ```
    
    ### Basic Usage
    ```python
    import chromadb
    
    # Create client
    client = chromadb.Client()
    
    # Create collection
    collection = client.create_collection("my_collection")
    
    # Add documents
    collection.add(
        documents=["Document 1", "Document 2"],
        metadatas=[{"source": "web"}, {"source": "book"}],
        ids=["id1", "id2"]
    )
    
    # Query
    results = collection.query(
        query_texts=["search query"],
        n_results=5
    )
    ```
    """)

with tab3:
    st.markdown("""
    ### Weaviate Setup
    ```bash
    pip install weaviate-client
    ```
    
    ### Basic Usage
    ```python
    import weaviate
    
    # Connect to Weaviate instance
    client = weaviate.Client("http://localhost:8080")
    
    # Create schema
    schema = {
        "class": "Document",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "title", "dataType": ["string"]}
        ]
    }
    client.schema.create_class(schema)
    
    # Add data
    client.data_object.create({
        "content": "Document content",
        "title": "Document title"
    }, "Document")
    
    # Query
    result = client.query.get("Document", ["content", "title"]).do()
    ```
    
    **Note:** Requires a running Weaviate instance or cloud setup.
    """)

# Footer
st.markdown("---")
st.markdown("*Navigate to other sections using the sidebar to explore more RAG techniques.*")


"""
Post-retrieval Optimization Techniques Page
Demonstrates re-ranking, context processing, prompt engineering, and LLM fine-tuning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers_basic import (
    rerank_documents, process_context, generate_optimized_prompt,
    compare_finetuning_vs_prompting, create_reranking_viz, create_context_processing_viz
)
from data.rag_techniques import RAG_TECHNIQUES

st.set_page_config(page_title="Post-retrieval Optimization", page_icon="⚡", layout="wide")

st.title("⚡ Post-retrieval Optimization Techniques")

st.markdown("""
Post-retrieval optimization techniques enhance the quality and relevance of retrieved information after the initial retrieval step. 
These techniques refine, re-rank, and optimize the context before it's passed to the generation stage.
""")

# Techniques Overview
st.markdown("## 📋 Post-retrieval Techniques Overview")

postretrieval_techniques = RAG_TECHNIQUES.get("post_retrieval_optimization", {})
if postretrieval_techniques:
    techniques_data = []
    for technique_id, technique in postretrieval_techniques.items():
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
        label="📥 Download Post-retrieval Techniques",
        data=csv,
        file_name="postretrieval_techniques.csv",
        mime="text/csv"
    )

st.markdown("---")

# Interactive Demos
st.markdown("## 🛠️ Interactive Demos")

tab1, tab2, tab3, tab4 = st.tabs(["🔄 Re-ranking", "📝 Context Processing", "🎯 Prompt Engineering", "🧠 LLM Fine-tuning"])

with tab1:
    st.markdown("### Document Re-ranking Demo")
    st.markdown("Reorder retrieved documents based on additional relevance signals and query-document alignment.")
    
    # Sample retrieved documents
    sample_docs = [
        {"title": "Introduction to Machine Learning", "content": "Machine learning is a subset of AI...", "initial_score": 0.85},
        {"title": "Deep Learning Fundamentals", "content": "Deep learning uses neural networks...", "initial_score": 0.78},
        {"title": "Natural Language Processing", "content": "NLP combines computational linguistics...", "initial_score": 0.72},
        {"title": "Computer Vision Applications", "content": "Computer vision enables machines...", "initial_score": 0.69},
        {"title": "Reinforcement Learning Basics", "content": "Reinforcement learning is about training agents...", "initial_score": 0.65}
    ]
    
    # Query for re-ranking
    rerank_query = st.text_input("Query for re-ranking:", "deep learning neural networks")
    
    # Re-ranking method
    rerank_method = st.selectbox("Re-ranking Method:", ["Cross-encoder", "Semantic Similarity", "Query-Document Alignment"])
    
    if st.button("Apply Re-ranking"):
        reranked_docs = rerank_documents(sample_docs, rerank_query, rerank_method)
        
        st.markdown("### Re-ranking Results")
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Ranking:**")
            for i, doc in enumerate(sample_docs, 1):
                st.write(f"{i}. {doc['title']} (Score: {doc['initial_score']:.3f})")
        
        with col2:
            st.markdown("**Re-ranked Results:**")
            for i, doc in enumerate(reranked_docs, 1):
                st.write(f"{i}. {doc['title']} (Score: {doc['rerank_score']:.3f})")
        
        # Visualization
        fig = create_reranking_viz(sample_docs, reranked_docs)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        rerank_df = pd.DataFrame(reranked_docs)
        csv = rerank_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Re-ranking Results",
            data=csv,
            file_name="reranking_results.csv",
            mime="text/csv"
        )

with tab2:
    st.markdown("### Context Post-processing Demo")
    st.markdown("Refine and optimize retrieved context before passing to the generation model.")
    
    # Sample context
    sample_context = st.text_area(
        "Original Context:",
        value="""Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.""",
        height=150
    )
    
    # Processing options
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        remove_redundancy = st.checkbox("Remove redundancy", value=True)
    with col2:
        summarize_content = st.checkbox("Summarize content", value=False)
    with col3:
        extract_key_points = st.checkbox("Extract key points", value=True)
    with col4:
        reorder_by_relevance = st.checkbox("Reorder by relevance", value=False)
    
    if st.button("Process Context"):
        processed_context = process_context(
            sample_context, remove_redundancy, summarize_content, 
            extract_key_points, reorder_by_relevance
        )
        
        st.markdown("### Processing Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Context:**")
            st.text_area("", value=sample_context, height=200, disabled=True)
        
        with col2:
            st.markdown("**Processed Context:**")
            st.text_area("", value=processed_context["text"], height=200, disabled=True)
        
        # Processing metrics
        st.markdown("### Processing Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Words", processed_context["original_words"])
        with col2:
            st.metric("Processed Words", processed_context["processed_words"])
        with col3:
            st.metric("Compression", f"{processed_context['compression']:.1f}%")
        
        # Visualization
        fig = create_context_processing_viz(processed_context)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Prompt Engineering Demo")
    st.markdown("Optimize prompts to improve the quality of generated responses using retrieved context.")
    
    # Sample query and context
    prompt_query = st.text_area(
        "Query:",
        value="What are the main differences between supervised and unsupervised learning?",
        height=100
    )
    
    prompt_context = st.text_area(
        "Context:",
        value="Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Unsupervised learning finds patterns in data without labeled examples. Semi-supervised learning combines both approaches.",
        height=100
    )
    
    # Prompt strategy
    col1, col2 = st.columns(2)
    with col1:
        prompt_strategy = st.selectbox("Prompt Strategy:", ["Chain-of-thought", "Few-shot", "Zero-shot", "Role-based"])
    with col2:
        response_format = st.selectbox("Response Format:", ["Detailed explanation", "Bullet points", "Comparison table", "Step-by-step"])
    
    if st.button("Generate Prompt"):
        optimized_prompt = generate_optimized_prompt(prompt_query, prompt_context, prompt_strategy, response_format)
        
        st.markdown("### Generated Prompt")
        st.text_area("Optimized Prompt:", value=optimized_prompt["prompt"], height=200, disabled=True)
        
        # Prompt metrics
        st.markdown("### Prompt Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prompt Length (words)", optimized_prompt["word_count"])
        with col2:
            st.metric("Strategy Applied", prompt_strategy)
        with col3:
            st.metric("Format", response_format)

with tab4:
    st.markdown("### LLM Fine-tuning vs. Prompt Engineering")
    st.markdown("Compare fine-tuning and prompt engineering approaches for different scenarios.")
    
    # Scenario selection
    scenarios = [
        "Domain-specific terminology",
        "Complex reasoning tasks", 
        "Consistent output format",
        "Multilingual support",
        "Real-time applications"
    ]
    
    selected_scenario = st.selectbox("Select Scenario:", scenarios)
    
    if st.button("Compare Approaches"):
        comparison = compare_finetuning_vs_prompting(selected_scenario)
        
        st.markdown("### Comparison Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Prompt Engineering Approach")
            st.info(comparison["prompt_engineering"]["description"])
            st.write(f"**Effort:** {comparison['prompt_engineering']['effort']}")
            st.write(f"**Effectiveness:** {comparison['prompt_engineering']['effectiveness']}")
        
        with col2:
            st.markdown("#### Fine-tuning Approach")
            st.info(comparison["fine_tuning"]["description"])
            st.write(f"**Effort:** {comparison['fine_tuning']['effort']}")
            st.write(f"**Effectiveness:** {comparison['fine_tuning']['effectiveness']}")
        
        # Recommendation
        st.markdown("### 💡 Recommendation")
        st.success(comparison["recommendation"])
        
        # Comparison chart
        methods = ["Prompt Engineering", "Fine-tuning"]
        effort_scores = [comparison["prompt_engineering"]["effort_score"], comparison["fine_tuning"]["effort_score"]]
        effectiveness_scores = [comparison["prompt_engineering"]["effectiveness_score"], comparison["fine_tuning"]["effectiveness_score"]]
        
        comparison_df = pd.DataFrame({
            "Method": methods,
            "Effort (Lower is Better)": effort_scores,
            "Effectiveness (Higher is Better)": effectiveness_scores
        })
        
        fig = px.scatter(
            comparison_df,
            x="Effort (Lower is Better)",
            y="Effectiveness (Higher is Better)",
            text="Method",
            title=f"Approach Comparison for: {selected_scenario}",
            size_max=20
        )
        fig.update_traces(textposition="top center", marker_size=15)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Best Practices
st.markdown("## 💡 Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Re-ranking & Context Processing
    - Use multiple relevance signals for re-ranking
    - Balance context length with information density
    - Remove redundant or contradictory information
    - Preserve essential context for generation
    """)

with col2:
    st.markdown("""
    ### Prompt Engineering & Fine-tuning
    - Choose strategy based on use case requirements
    - Test different prompt formats and structures
    - Consider computational costs and latency
    - Monitor output quality and consistency
    """)

# Footer
st.markdown("---")
st.markdown("*Navigate to other sections using the sidebar to explore more RAG techniques.*")


"""
Basic utility functions for the Advanced RAG Techniques MVP
Comprehensive helper functions for all RAG technique demonstrations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import re
import base64
import numpy as np
import random

# ============================================================================
# INDEXING OPTIMIZATION FUNCTIONS
# ============================================================================

def clean_text(text: str, remove_special: bool = True, normalize_spaces: bool = True, remove_urls: bool = True) -> str:
    """Clean and preprocess text data"""
    cleaned = text
    
    if remove_urls:
        # Remove URLs and email addresses
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
    
    if remove_special:
        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-]', '', cleaned)
    
    if normalize_spaces:
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
    
    return cleaned

def apply_chunking(text: str, chunk_size: int = 200, overlap: int = 20, method: str = "Fixed Size") -> List[str]:
    """Apply different chunking strategies to text"""
    chunks = []
    
    if method == "Fixed Size":
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
    
    elif method == "Recursive":
        # Split by sentences first, then by size if needed
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    elif method == "Sentence-based":
        # Split by sentences and group them
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if sentence.strip():
                if len(current_chunk + sentence) <= chunk_size:
                    current_chunk += sentence.strip() + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence.strip() + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def create_chunking_visualization(chunks: List[str]) -> go.Figure:
    """Create visualization for chunking results"""
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(1, len(chunks) + 1)),
        y=chunk_lengths,
        name="Chunk Length",
        marker_color="lightblue"
    ))
    
    fig.update_layout(
        title="Chunk Size Distribution",
        xaxis_title="Chunk Number",
        yaxis_title="Character Count",
        height=400
    )
    
    return fig

def create_chunk_size_distribution(chunks: List[str]) -> go.Figure:
    """Create chunk size distribution histogram"""
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    fig = px.histogram(
        x=chunk_lengths,
        nbins=10,
        title="Chunk Size Distribution",
        labels={"x": "Chunk Length (characters)", "y": "Frequency"}
    )
    
    return fig

# ============================================================================
# PRE-RETRIEVAL OPTIMIZATION FUNCTIONS
# ============================================================================

def decompose_query(query: str) -> Dict[str, Any]:
    """Decompose a complex query into sub-queries"""
    # Simple decomposition based on keywords and structure
    sub_queries = []
    
    # Look for comparison words
    if any(word in query.lower() for word in ["difference", "compare", "vs", "versus"]):
        parts = re.split(r'\band\b|\bvs\b|\bversus\b', query, flags=re.IGNORECASE)
        for part in parts:
            if part.strip():
                sub_queries.append(f"What is {part.strip()}?")
    
    # Look for multiple concepts
    elif " and " in query.lower():
        parts = query.split(" and ")
        for part in parts:
            if part.strip():
                sub_queries.append(part.strip() + "?")
    
    # Look for how/what questions
    elif query.lower().startswith(("how", "what", "why", "when", "where")):
        # Extract main concepts
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        if len(concepts) > 1:
            for concept in concepts[:3]:  # Limit to 3 sub-queries
                sub_queries.append(f"What is {concept}?")
        else:
            sub_queries = [query]
    
    else:
        # Default: split by major concepts
        words = query.split()
        if len(words) > 5:
            mid = len(words) // 2
            sub_queries = [
                " ".join(words[:mid]) + "?",
                " ".join(words[mid:]) + "?"
            ]
        else:
            sub_queries = [query]
    
    # Ensure we have at least the original query
    if not sub_queries:
        sub_queries = [query]
    
    complexity = min(len(sub_queries) * 0.3 + len(query.split()) * 0.1, 1.0)
    
    return {
        "original_query": query,
        "sub_queries": sub_queries[:3],  # Limit to 3 for display
        "complexity": complexity
    }

def route_query(query: str) -> Dict[str, Any]:
    """Route query to appropriate system based on content analysis"""
    query_lower = query.lower()
    
    # Define routing scores
    scores = {
        "Vector DB": 0.0,
        "Knowledge Graph": 0.0,
        "SQL Database": 0.0,
        "Web Search": 0.0
    }
    
    # Vector DB indicators
    if any(word in query_lower for word in ["similar", "semantic", "meaning", "concept", "related"]):
        scores["Vector DB"] += 0.4
    if any(word in query_lower for word in ["machine learning", "ai", "neural", "algorithm"]):
        scores["Vector DB"] += 0.3
    
    # Knowledge Graph indicators
    if any(word in query_lower for word in ["relationship", "connection", "related to", "linked"]):
        scores["Knowledge Graph"] += 0.4
    if any(word in query_lower for word in ["entity", "person", "organization", "location"]):
        scores["Knowledge Graph"] += 0.3
    
    # SQL Database indicators
    if any(word in query_lower for word in ["count", "sum", "average", "statistics", "data"]):
        scores["SQL Database"] += 0.4
    if any(word in query_lower for word in ["table", "record", "database", "structured"]):
        scores["SQL Database"] += 0.3
    
    # Web Search indicators
    if any(word in query_lower for word in ["latest", "recent", "news", "current", "today"]):
        scores["Web Search"] += 0.4
    if any(word in query_lower for word in ["price", "stock", "weather", "real-time"]):
        scores["Web Search"] += 0.3
    
    # Add base scores
    scores["Vector DB"] += 0.2
    scores["Knowledge Graph"] += 0.1
    scores["SQL Database"] += 0.1
    scores["Web Search"] += 0.2
    
    # Normalize scores
    max_score = max(scores.values())
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}
    
    recommended_system = max(scores, key=scores.get)
    
    return {
        "query": query,
        "scores": scores,
        "recommended_system": recommended_system
    }

def create_query_decomposition_viz(decomposition: Dict[str, Any]) -> go.Figure:
    """Create visualization for query decomposition"""
    sub_queries = decomposition["sub_queries"]
    
    fig = go.Figure()
    
    # Create a simple tree-like visualization
    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[1],
        mode='markers+text',
        marker=dict(size=20, color='lightblue'),
        text=["Original Query"],
        textposition="middle center",
        name="Original"
    ))
    
    # Add sub-queries
    x_positions = np.linspace(0.1, 0.9, len(sub_queries))
    for i, (x_pos, sub_query) in enumerate(zip(x_positions, sub_queries)):
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[0.3],
            mode='markers+text',
            marker=dict(size=15, color='lightgreen'),
            text=[f"Sub-query {i+1}"],
            textposition="middle center",
            name=f"Sub-query {i+1}"
        ))
        
        # Add connecting line
        fig.add_trace(go.Scatter(
            x=[0.5, x_pos],
            y=[1, 0.3],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Query Decomposition Structure",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=400,
        showlegend=False
    )
    
    return fig

def create_query_routing_viz(routing_result: Dict[str, Any]) -> go.Figure:
    """Create visualization for query routing"""
    systems = list(routing_result["scores"].keys())
    scores = list(routing_result["scores"].values())
    
    colors = ['red' if system == routing_result["recommended_system"] else 'lightblue' for system in systems]
    
    fig = go.Figure(data=[
        go.Bar(x=systems, y=scores, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Query Routing Scores",
        xaxis_title="System",
        yaxis_title="Relevance Score",
        height=400
    )
    
    return fig

# ============================================================================
# RETRIEVAL OPTIMIZATION FUNCTIONS
# ============================================================================

def apply_metadata_filter(documents: List[Dict], category_filter: List[str], 
                         year_filter: List[int], difficulty_filter: List[str]) -> List[Dict]:
    """Apply metadata filters to documents"""
    filtered_docs = documents.copy()
    
    if category_filter:
        filtered_docs = [doc for doc in filtered_docs if doc.get("category") in category_filter]
    
    if year_filter:
        filtered_docs = [doc for doc in filtered_docs if doc.get("year") in year_filter]
    
    if difficulty_filter:
        filtered_docs = [doc for doc in filtered_docs if doc.get("difficulty") in difficulty_filter]
    
    return filtered_docs

def perform_hybrid_search(query: str, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict]:
    """Perform hybrid search combining vector and keyword search"""
    # Sample documents for demonstration
    sample_docs = [
        {"title": "Introduction to Machine Learning", "content": "Machine learning is a subset of AI...", "category": "AI/ML"},
        {"title": "Deep Learning Fundamentals", "content": "Deep learning uses neural networks...", "category": "Deep Learning"},
        {"title": "Natural Language Processing", "content": "NLP combines computational linguistics...", "category": "NLP"},
        {"title": "Computer Vision Applications", "content": "Computer vision enables machines...", "category": "Computer Vision"},
        {"title": "Reinforcement Learning Basics", "content": "Reinforcement learning is about training agents...", "category": "Reinforcement Learning"}
    ]
    
    results = []
    query_words = set(query.lower().split())
    
    for doc in sample_docs:
        # Simulate vector search score
        vector_score = random.uniform(0.5, 0.95)
        
        # Calculate keyword search score
        doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
        keyword_matches = len(query_words.intersection(doc_words))
        keyword_score = min(keyword_matches / len(query_words), 1.0) if query_words else 0
        
        # Combine scores
        combined_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
        
        result = doc.copy()
        result["vector_score"] = vector_score
        result["keyword_score"] = keyword_score
        result["score"] = combined_score
        results.append(result)
    
    # Sort by combined score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

def create_search_comparison_viz(search_results: List[Dict]) -> go.Figure:
    """Create visualization comparing search methods"""
    titles = [result["title"] for result in search_results]
    vector_scores = [result["vector_score"] for result in search_results]
    keyword_scores = [result["keyword_score"] for result in search_results]
    combined_scores = [result["score"] for result in search_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Vector Score',
        x=titles,
        y=vector_scores,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Keyword Score',
        x=titles,
        y=keyword_scores,
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        name='Combined Score',
        x=titles,
        y=combined_scores,
        marker_color='orange'
    ))
    
    fig.update_layout(
        title='Search Method Comparison',
        xaxis_title='Documents',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    return fig

def create_filter_effectiveness_viz(original_docs: List[Dict], filtered_docs: List[Dict]) -> go.Figure:
    """Create visualization showing filter effectiveness"""
    categories = ["Original", "Filtered"]
    counts = [len(original_docs), len(filtered_docs)]
    
    fig = px.bar(
        x=categories,
        y=counts,
        title="Filter Effectiveness",
        labels={"x": "Document Set", "y": "Number of Documents"},
        color=counts,
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(showlegend=False, height=400)
    
    return fig

# ============================================================================
# POST-RETRIEVAL OPTIMIZATION FUNCTIONS
# ============================================================================

def rerank_documents(documents: List[Dict], query: str, method: str = "Cross-encoder") -> List[Dict]:
    """Re-rank documents based on query relevance"""
    reranked_docs = []
    
    for doc in documents:
        # Simulate re-ranking based on method
        if method == "Cross-encoder":
            # Simulate cross-encoder scoring
            rerank_score = doc["initial_score"] * random.uniform(0.8, 1.2)
        elif method == "Semantic Similarity":
            # Simulate semantic similarity
            rerank_score = doc["initial_score"] * random.uniform(0.7, 1.1)
        else:  # Query-Document Alignment
            # Simulate alignment scoring
            rerank_score = doc["initial_score"] * random.uniform(0.9, 1.15)
        
        rerank_score = min(rerank_score, 1.0)  # Cap at 1.0
        
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = rerank_score
        reranked_docs.append(doc_copy)
    
    # Sort by re-ranking score
    reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_docs

def process_context(context: str, remove_redundancy: bool = True, summarize_content: bool = False,
                   extract_key_points: bool = True, reorder_by_relevance: bool = False) -> Dict[str, Any]:
    """Process and optimize context for generation"""
    processed_text = context
    original_words = len(context.split())
    
    if remove_redundancy:
        # Simple redundancy removal (remove repeated sentences)
        sentences = context.split('. ')
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            if sentence.lower() not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence.lower())
        processed_text = '. '.join(unique_sentences)
    
    if summarize_content:
        # Simple summarization (take first and last sentences of each paragraph)
        paragraphs = processed_text.split('\n\n')
        summarized = []
        for para in paragraphs:
            sentences = para.split('. ')
            if len(sentences) > 2:
                summarized.append(sentences[0] + '. ' + sentences[-1])
            else:
                summarized.append(para)
        processed_text = '\n\n'.join(summarized)
    
    if extract_key_points:
        # Add key point markers
        processed_text = "Key Points:\n" + processed_text
    
    processed_words = len(processed_text.split())
    compression = ((original_words - processed_words) / original_words) * 100 if original_words > 0 else 0
    
    return {
        "text": processed_text,
        "original_words": original_words,
        "processed_words": processed_words,
        "compression": compression
    }

def generate_optimized_prompt(query: str, context: str, strategy: str = "Chain-of-thought", 
                            response_format: str = "Detailed explanation") -> Dict[str, Any]:
    """Generate optimized prompt based on strategy and format"""
    
    prompt_parts = []
    
    # Add strategy-specific instructions
    if strategy == "Chain-of-thought":
        prompt_parts.append("Think step by step and explain your reasoning.")
    elif strategy == "Few-shot":
        prompt_parts.append("Here are some examples of similar questions and answers:")
        prompt_parts.append("Example: Q: What is AI? A: AI is artificial intelligence...")
    elif strategy == "Zero-shot":
        prompt_parts.append("Answer the following question directly.")
    elif strategy == "Role-based":
        prompt_parts.append("You are an expert in this field. Provide a professional response.")
    
    # Add format-specific instructions
    if response_format == "Bullet points":
        prompt_parts.append("Format your response as bullet points.")
    elif response_format == "Comparison table":
        prompt_parts.append("Present the information in a comparison table format.")
    elif response_format == "Step-by-step":
        prompt_parts.append("Break down your response into clear steps.")
    
    # Add context and query
    prompt_parts.append(f"Context: {context}")
    prompt_parts.append(f"Question: {query}")
    prompt_parts.append("Answer:")
    
    optimized_prompt = "\n\n".join(prompt_parts)
    
    return {
        "prompt": optimized_prompt,
        "word_count": len(optimized_prompt.split()),
        "strategy": strategy,
        "format": response_format
    }

def compare_finetuning_vs_prompting(scenario: str) -> Dict[str, Any]:
    """Compare fine-tuning vs prompt engineering for different scenarios"""
    
    scenarios_data = {
        "Domain-specific terminology": {
            "prompt_engineering": {
                "description": "Use domain-specific prompts with examples and terminology definitions",
                "effort": "Low - requires prompt design and testing",
                "effectiveness": "Medium - may struggle with very specialized terms",
                "effort_score": 2,
                "effectiveness_score": 6
            },
            "fine_tuning": {
                "description": "Train model on domain-specific corpus with specialized vocabulary",
                "effort": "High - requires dataset preparation and training",
                "effectiveness": "High - model learns domain language patterns",
                "effort_score": 8,
                "effectiveness_score": 9
            },
            "recommendation": "Fine-tuning is better for domains with extensive specialized vocabulary"
        },
        "Complex reasoning tasks": {
            "prompt_engineering": {
                "description": "Use chain-of-thought prompting and structured reasoning templates",
                "effort": "Medium - requires careful prompt design",
                "effectiveness": "High - can guide reasoning effectively",
                "effort_score": 4,
                "effectiveness_score": 8
            },
            "fine_tuning": {
                "description": "Train on reasoning datasets and problem-solving examples",
                "effort": "Very High - requires large reasoning datasets",
                "effectiveness": "High - but may not generalize well",
                "effort_score": 9,
                "effectiveness_score": 7
            },
            "recommendation": "Prompt engineering is more cost-effective for reasoning tasks"
        },
        "Consistent output format": {
            "prompt_engineering": {
                "description": "Use structured prompts with format examples and constraints",
                "effort": "Low - template-based approach",
                "effectiveness": "High - very effective for format control",
                "effort_score": 2,
                "effectiveness_score": 9
            },
            "fine_tuning": {
                "description": "Train model to always output in specific format",
                "effort": "Medium - requires format-specific training data",
                "effectiveness": "Very High - consistent formatting",
                "effort_score": 6,
                "effectiveness_score": 10
            },
            "recommendation": "Both approaches work well; choose based on complexity needs"
        },
        "Multilingual support": {
            "prompt_engineering": {
                "description": "Use language-specific prompts and translation instructions",
                "effort": "Medium - requires prompts for each language",
                "effectiveness": "Medium - depends on base model capabilities",
                "effort_score": 5,
                "effectiveness_score": 6
            },
            "fine_tuning": {
                "description": "Train on multilingual datasets for target languages",
                "effort": "Very High - requires multilingual training data",
                "effectiveness": "Very High - native language understanding",
                "effort_score": 10,
                "effectiveness_score": 10
            },
            "recommendation": "Fine-tuning is essential for high-quality multilingual support"
        },
        "Real-time applications": {
            "prompt_engineering": {
                "description": "Optimize prompts for speed and efficiency",
                "effort": "Low - focus on prompt optimization",
                "effectiveness": "High - fast inference with good results",
                "effort_score": 2,
                "effectiveness_score": 8
            },
            "fine_tuning": {
                "description": "Create smaller, specialized models for faster inference",
                "effort": "High - model compression and optimization needed",
                "effectiveness": "Very High - optimized for specific use case",
                "effort_score": 8,
                "effectiveness_score": 9
            },
            "recommendation": "Prompt engineering for quick deployment, fine-tuning for production optimization"
        }
    }
    
    return scenarios_data.get(scenario, scenarios_data["Domain-specific terminology"])

def create_reranking_viz(original_docs: List[Dict], reranked_docs: List[Dict]) -> go.Figure:
    """Create visualization for re-ranking results"""
    titles = [doc["title"] for doc in original_docs]
    original_scores = [doc["initial_score"] for doc in original_docs]
    reranked_scores = [doc["rerank_score"] for doc in reranked_docs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Original Score',
        x=titles,
        y=original_scores,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Re-ranked Score',
        x=titles,
        y=reranked_scores,
        marker_color='orange'
    ))
    
    fig.update_layout(
        title='Document Re-ranking Comparison',
        xaxis_title='Documents',
        yaxis_title='Relevance Score',
        barmode='group',
        height=400
    )
    
    return fig

def create_context_processing_viz(processing_result: Dict[str, Any]) -> go.Figure:
    """Create visualization for context processing results"""
    metrics = ["Original Words", "Processed Words"]
    values = [processing_result["original_words"], processing_result["processed_words"]]
    
    fig = px.bar(
        x=metrics,
        y=values,
        title="Context Processing Results",
        labels={"x": "Metric", "y": "Word Count"},
        color=values,
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(showlegend=False, height=400)
    
    return fig

# ============================================================================
# GENERAL UTILITY FUNCTIONS
# ============================================================================

def create_overview_metrics() -> Dict[str, int]:
    """Create overview metrics for the home page"""
    return {
        "total_techniques": 13,
        "vector_databases": 3,
        "main_categories": 4,
        "interactive_demos": 12
    }

def create_download_link(df: pd.DataFrame, filename: str, link_text: str = "Download CSV") -> str:
    """Create a download link for a pandas DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


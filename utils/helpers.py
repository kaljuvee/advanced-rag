"""
Utility functions for the Advanced RAG Techniques MVP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import io
import base64

# Initialize embedding model (cached)
@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the sentence transformer model"""
    return SentenceTransformer(model_name)

def create_download_link(df: pd.DataFrame, filename: str, link_text: str = "Download CSV") -> str:
    """Create a download link for a pandas DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def display_technique_card(title: str, description: str, details: Dict[str, Any] = None):
    """Display a technique in a styled card format"""
    st.markdown(f"""
    <div class="technique-card">
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if details:
        with st.expander("View Details", expanded=False):
            for key, value in details.items():
                if isinstance(value, list):
                    st.markdown(f"**{key}:**")
                    for item in value:
                        st.markdown(f"- {item}")
                elif isinstance(value, dict):
                    st.markdown(f"**{key}:**")
                    for subkey, subvalue in value.items():
                        st.markdown(f"- **{subkey}**: {subvalue}")
                else:
                    st.markdown(f"**{key}**: {value}")

def chunk_text_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Chunk text using fixed-size strategy with overlap"""
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "start": start,
            "end": min(end, len(text)),
            "size": len(chunk_text),
            "method": "fixed_size"
        })
        
        start = end - overlap
        chunk_id += 1
        
        if end >= len(text):
            break
    
    return chunks

def chunk_text_recursive(text: str, separators: List[str] = None, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Chunk text using recursive strategy"""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    
    chunks = []
    
    def _split_text(text: str, separators: List[str], current_separator_index: int = 0) -> List[str]:
        if current_separator_index >= len(separators):
            return [text]
        
        separator = separators[current_separator_index]
        splits = text.split(separator)
        
        result = []
        for split in splits:
            if len(split) <= max_chunk_size:
                result.append(split)
            else:
                # Try next separator
                sub_splits = _split_text(split, separators, current_separator_index + 1)
                result.extend(sub_splits)
        
        return result
    
    text_chunks = _split_text(text, separators)
    
    for i, chunk_text in enumerate(text_chunks):
        if chunk_text.strip():  # Skip empty chunks
            chunks.append({
                "id": i,
                "text": chunk_text.strip(),
                "size": len(chunk_text.strip()),
                "method": "recursive"
            })
    
    return chunks

def chunk_text_semantic(text: str, model, similarity_threshold: float = 0.7, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Chunk text using semantic similarity"""
    # Split into sentences first
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return [{"id": 0, "text": text, "size": len(text), "method": "semantic"}]
    
    # Get embeddings for sentences
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = []
    current_chunk_embedding = None
    chunk_id = 0
    
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        if not current_chunk:
            current_chunk = [sentence]
            current_chunk_embedding = embedding.reshape(1, -1)
        else:
            # Calculate similarity with current chunk
            similarity = cosine_similarity(embedding.reshape(1, -1), current_chunk_embedding)[0][0]
            
            # Check if adding this sentence would exceed max size
            potential_chunk_text = ". ".join(current_chunk + [sentence])
            
            if similarity >= similarity_threshold and len(potential_chunk_text) <= max_chunk_size:
                current_chunk.append(sentence)
                # Update chunk embedding (average)
                current_chunk_embedding = np.mean([current_chunk_embedding.flatten(), embedding], axis=0).reshape(1, -1)
            else:
                # Finalize current chunk
                chunk_text = ". ".join(current_chunk)
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "size": len(chunk_text),
                    "method": "semantic",
                    "similarity_threshold": similarity_threshold
                })
                
                # Start new chunk
                current_chunk = [sentence]
                current_chunk_embedding = embedding.reshape(1, -1)
                chunk_id += 1
    
    # Add the last chunk
    if current_chunk:
        chunk_text = ". ".join(current_chunk)
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "size": len(chunk_text),
            "method": "semantic",
            "similarity_threshold": similarity_threshold
        })
    
    return chunks

def visualize_chunks(chunks: List[Dict[str, Any]], title: str = "Chunk Analysis"):
    """Visualize chunk statistics"""
    df = pd.DataFrame(chunks)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Chunk Sizes", "Chunk Size Distribution", "Cumulative Text Coverage", "Method Comparison"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Chunk sizes over sequence
    fig.add_trace(
        go.Scatter(x=df['id'], y=df['size'], mode='lines+markers', name='Chunk Size'),
        row=1, col=1
    )
    
    # Size distribution
    fig.add_trace(
        go.Histogram(x=df['size'], nbinsx=20, name='Size Distribution'),
        row=1, col=2
    )
    
    # Cumulative coverage
    cumulative_size = df['size'].cumsum()
    fig.add_trace(
        go.Scatter(x=df['id'], y=cumulative_size, mode='lines', name='Cumulative Size'),
        row=2, col=1
    )
    
    # Method comparison (if multiple methods)
    if 'method' in df.columns and df['method'].nunique() > 1:
        method_stats = df.groupby('method')['size'].agg(['mean', 'std', 'count']).reset_index()
        fig.add_trace(
            go.Bar(x=method_stats['method'], y=method_stats['mean'], name='Avg Size by Method'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text=title, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Avg Chunk Size", f"{df['size'].mean():.0f}")
    with col3:
        st.metric("Min Size", df['size'].min())
    with col4:
        st.metric("Max Size", df['size'].max())

def visualize_embeddings_2d(texts: List[str], labels: List[str] = None, method: str = "PCA"):
    """Visualize text embeddings in 2D space"""
    model = load_embedding_model()
    embeddings = model.encode(texts)
    
    if method == "PCA":
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'text': [text[:100] + "..." if len(text) > 100 else text for text in texts],
        'label': labels if labels else [f"Text {i}" for i in range(len(texts))]
    })
    
    fig = px.scatter(
        df, x='x', y='y', 
        color='label',
        hover_data=['text'],
        title=f"Text Embeddings Visualization ({method})"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return df

def calculate_similarity_matrix(texts: List[str]) -> np.ndarray:
    """Calculate cosine similarity matrix for a list of texts"""
    model = load_embedding_model()
    embeddings = model.encode(texts)
    return cosine_similarity(embeddings)

def visualize_similarity_matrix(texts: List[str], labels: List[str] = None):
    """Visualize similarity matrix as a heatmap"""
    similarity_matrix = calculate_similarity_matrix(texts)
    
    if labels is None:
        labels = [f"Text {i+1}" for i in range(len(texts))]
    
    fig = px.imshow(
        similarity_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Text Similarity Matrix"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Return similarity matrix as DataFrame for download
    return pd.DataFrame(similarity_matrix, index=labels, columns=labels)

def simulate_vector_search(query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """Simulate vector search and return ranked results"""
    model = load_embedding_model()
    
    # Encode query and documents
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(documents)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Create results with rankings
    results = []
    for i, (doc, sim) in enumerate(zip(documents, similarities)):
        results.append({
            "rank": i + 1,
            "document": doc,
            "similarity": sim,
            "document_id": i
        })
    
    # Sort by similarity (descending) and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Update ranks after sorting
    for i, result in enumerate(results[:top_k]):
        result["rank"] = i + 1
    
    return results[:top_k]

def create_performance_comparison_chart(methods: List[str], metrics: Dict[str, List[float]], title: str = "Performance Comparison"):
    """Create a radar chart comparing different methods across multiple metrics"""
    
    fig = go.Figure()
    
    for method in methods:
        values = [metrics[metric][methods.index(method)] for metric in metrics.keys()]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(metrics.keys()),
            fill='toself',
            name=method
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title
    )
    
    st.plotly_chart(fig, use_container_width=True)

def format_code_block(code: str, language: str = "python") -> str:
    """Format code in a styled block"""
    return f"""
```{language}
{code}
```
"""

def display_metrics_grid(metrics: Dict[str, Any], columns: int = 4):
    """Display metrics in a grid layout"""
    cols = st.columns(columns)
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if isinstance(value, (int, float)):
                st.metric(key, f"{value:.3f}" if isinstance(value, float) else value)
            else:
                st.metric(key, str(value))

def create_technique_comparison_table(techniques: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table for different techniques"""
    comparison_data = []
    
    for technique_name, technique_info in techniques.items():
        row = {"Technique": technique_name}
        
        # Extract common fields
        if "description" in technique_info:
            row["Description"] = technique_info["description"]
        
        if "pros" in technique_info:
            row["Pros"] = "; ".join(technique_info["pros"])
        
        if "cons" in technique_info:
            row["Cons"] = "; ".join(technique_info["cons"])
        
        if "use_cases" in technique_info:
            row["Use Cases"] = "; ".join(technique_info["use_cases"])
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Color schemes for consistent styling
COLOR_SCHEMES = {
    "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "secondary": ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "gradient": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"]
}


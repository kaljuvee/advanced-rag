"""
Vector Database Utilities for Advanced RAG MVP
Provides integration with Faiss, ChromaDB, and Weaviate
"""

import numpy as np
import pandas as pd
import time
import streamlit as st
from typing import List, Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Vector Database Imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
        "category": "AI/ML",
        "embedding": None
    },
    {
        "id": "doc_2", 
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "category": "Deep Learning",
        "embedding": None
    },
    {
        "id": "doc_3",
        "title": "Natural Language Processing",
        "content": "NLP combines computational linguistics with machine learning to help computers understand human language.",
        "category": "NLP",
        "embedding": None
    },
    {
        "id": "doc_4",
        "title": "Computer Vision Applications",
        "content": "Computer vision enables machines to interpret and understand visual information from the world around them.",
        "category": "Computer Vision",
        "embedding": None
    },
    {
        "id": "doc_5",
        "title": "Reinforcement Learning Basics",
        "content": "Reinforcement learning is about training agents to make decisions by learning from rewards and penalties.",
        "category": "Reinforcement Learning",
        "embedding": None
    }
]

def generate_mock_embeddings(texts: List[str], dimension: int = 384) -> np.ndarray:
    """Generate mock embeddings for demonstration purposes"""
    np.random.seed(42)  # For reproducible results
    embeddings = []
    
    for text in texts:
        # Create somewhat realistic embeddings based on text characteristics
        base_embedding = np.random.normal(0, 0.1, dimension)
        
        # Add some semantic meaning based on keywords
        if "machine learning" in text.lower():
            base_embedding[:10] += 0.5
        if "deep learning" in text.lower():
            base_embedding[10:20] += 0.5
        if "neural" in text.lower():
            base_embedding[20:30] += 0.5
        if "language" in text.lower():
            base_embedding[30:40] += 0.5
        if "vision" in text.lower():
            base_embedding[40:50] += 0.5
        if "reinforcement" in text.lower():
            base_embedding[50:60] += 0.5
            
        # Normalize
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        embeddings.append(base_embedding)
    
    return np.array(embeddings)

class FaissVectorDB:
    """Faiss Vector Database Implementation"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.is_ready = FAISS_AVAILABLE
        
    def initialize(self):
        """Initialize Faiss index"""
        if not FAISS_AVAILABLE:
            return False
        
        try:
            # Create a flat L2 index
            self.index = faiss.IndexFlatL2(self.dimension)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Faiss: {e}")
            return False
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the index"""
        if not self.index:
            return False
            
        try:
            texts = [doc["content"] for doc in documents]
            embeddings = generate_mock_embeddings(texts, self.dimension)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            
            # Store documents
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["embedding"] = embeddings[i]
                self.documents.append(doc_copy)
            
            return True
        except Exception as e:
            st.error(f"Failed to add documents to Faiss: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.index or len(self.documents) == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = generate_mock_embeddings([query], self.dimension)[0]
            
            # Search
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), k
            )
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    result = self.documents[idx].copy()
                    result["score"] = float(1 / (1 + distance))  # Convert distance to similarity
                    result["rank"] = i + 1
                    results.append(result)
            
            return results
        except Exception as e:
            st.error(f"Failed to search in Faiss: {e}")
            return []

class ChromaVectorDB:
    """ChromaDB Vector Database Implementation"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.is_ready = CHROMADB_AVAILABLE
        
    def initialize(self):
        """Initialize ChromaDB client"""
        if not CHROMADB_AVAILABLE:
            return False
        
        try:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name="rag_documents",
                get_or_create=True
            )
            return True
        except Exception as e:
            st.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the collection"""
        if not self.collection:
            return False
            
        try:
            texts = [doc["content"] for doc in documents]
            embeddings = generate_mock_embeddings(texts)
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{"title": doc["title"], "category": doc["category"]} for doc in documents],
                ids=[doc["id"] for doc in documents]
            )
            return True
        except Exception as e:
            st.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.collection:
            return []
        
        try:
            query_embedding = generate_mock_embeddings([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "title": results['metadatas'][0][i]['title'],
                    "content": results['documents'][0][i],
                    "category": results['metadatas'][0][i]['category'],
                    "score": float(1 - results['distances'][0][i]),  # Convert distance to similarity
                    "rank": i + 1
                }
                search_results.append(result)
            
            return search_results
        except Exception as e:
            st.error(f"Failed to search in ChromaDB: {e}")
            return []

class WeaviateVectorDB:
    """Weaviate Vector Database Implementation"""
    
    def __init__(self):
        self.client = None
        self.is_ready = WEAVIATE_AVAILABLE
        self.needs_setup = True
        
    def initialize(self):
        """Initialize Weaviate client"""
        if not WEAVIATE_AVAILABLE:
            return False
        
        # For demo purposes, we'll simulate Weaviate without actual connection
        # In production, you would need Weaviate credentials
        self.needs_setup = True
        return True
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to Weaviate (simulated)"""
        # Simulated for demo - would need actual Weaviate instance
        return True
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents (simulated)"""
        # Simulated results for demo
        simulated_results = []
        for i, doc in enumerate(SAMPLE_DOCUMENTS[:k]):
            result = doc.copy()
            result["score"] = 0.9 - (i * 0.1)  # Decreasing similarity scores
            result["rank"] = i + 1
            simulated_results.append(result)
        
        return simulated_results

def get_database_status():
    """Get the status of all vector databases"""
    status = {
        "Faiss": {
            "available": FAISS_AVAILABLE,
            "status": "Ready" if FAISS_AVAILABLE else "Not Installed",
            "description": "CPU-based similarity search library by Facebook AI"
        },
        "ChromaDB": {
            "available": CHROMADB_AVAILABLE,
            "status": "Ready" if CHROMADB_AVAILABLE else "Not Installed", 
            "description": "Open-source embedding database"
        },
        "Weaviate": {
            "available": WEAVIATE_AVAILABLE,
            "status": "Needs Setup" if WEAVIATE_AVAILABLE else "Not Installed",
            "description": "Cloud-native vector database (requires credentials)"
        }
    }
    return status

def create_performance_comparison(results_dict: Dict[str, List[Dict]]) -> go.Figure:
    """Create performance comparison visualization"""
    
    # Prepare data for comparison
    db_names = []
    avg_scores = []
    result_counts = []
    
    for db_name, results in results_dict.items():
        if results:
            db_names.append(db_name)
            scores = [r.get("score", 0) for r in results]
            avg_scores.append(np.mean(scores))
            result_counts.append(len(results))
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Similarity Score', 'Number of Results'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add bar charts
    fig.add_trace(
        go.Bar(x=db_names, y=avg_scores, name="Avg Score", marker_color="lightblue"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=db_names, y=result_counts, name="Result Count", marker_color="lightgreen"),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Vector Database Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def create_similarity_heatmap(results: List[Dict]) -> go.Figure:
    """Create similarity score heatmap"""
    if not results:
        return go.Figure()
    
    # Prepare data
    titles = [r.get("title", f"Doc {i+1}") for i, r in enumerate(results)]
    scores = [r.get("score", 0) for r in results]
    
    # Create heatmap data
    heatmap_data = np.array(scores).reshape(1, -1)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=titles,
        y=["Similarity"],
        colorscale="Viridis",
        showscale=True
    ))
    
    fig.update_layout(
        title="Document Similarity Scores",
        xaxis_title="Documents",
        height=300
    )
    
    return fig

def benchmark_databases(query: str = "machine learning algorithms") -> Dict[str, Any]:
    """Benchmark all available databases"""
    results = {}
    
    # Initialize databases
    faiss_db = FaissVectorDB()
    chroma_db = ChromaVectorDB()
    weaviate_db = WeaviateVectorDB()
    
    databases = {
        "Faiss": faiss_db,
        "ChromaDB": chroma_db,
        "Weaviate": weaviate_db
    }
    
    for db_name, db in databases.items():
        start_time = time.time()
        
        try:
            # Initialize
            if db.initialize():
                # Add documents
                db.add_documents(SAMPLE_DOCUMENTS)
                
                # Search
                search_results = db.search(query, k=3)
                
                end_time = time.time()
                
                results[db_name] = {
                    "results": search_results,
                    "execution_time": end_time - start_time,
                    "status": "Success"
                }
            else:
                results[db_name] = {
                    "results": [],
                    "execution_time": 0,
                    "status": "Failed to initialize"
                }
        except Exception as e:
            results[db_name] = {
                "results": [],
                "execution_time": 0,
                "status": f"Error: {str(e)}"
            }
    
    return results


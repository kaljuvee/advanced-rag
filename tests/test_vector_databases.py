"""
Test Vector Database Utilities
Tests vector database functionality with sample data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.vector_db_utils import (
    FaissVectorDB, ChromaVectorDB, WeaviateVectorDB,
    get_database_status, benchmark_databases, SAMPLE_DOCUMENTS
)
import json
from datetime import datetime
import time

def test_database_status():
    """Test database availability and status"""
    print("Testing Database Status...")
    
    status = get_database_status()
    
    results = []
    for db_name, info in status.items():
        result = {
            "database": db_name,
            "available": info["available"],
            "status": info["status"],
            "description": info["description"],
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    return results

def test_faiss_operations():
    """Test Faiss vector database operations"""
    print("Testing Faiss Operations...")
    
    results = []
    
    try:
        # Initialize Faiss
        faiss_db = FaissVectorDB(dimension=384)
        init_success = faiss_db.initialize()
        
        result = {
            "test": "faiss_initialization",
            "success": init_success,
            "available": faiss_db.is_ready,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        if init_success:
            # Test adding documents
            start_time = time.time()
            add_success = faiss_db.add_documents(SAMPLE_DOCUMENTS)
            add_time = time.time() - start_time
            
            result = {
                "test": "faiss_add_documents",
                "success": add_success,
                "documents_added": len(SAMPLE_DOCUMENTS),
                "execution_time": add_time,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            if add_success:
                # Test search
                test_queries = [
                    "machine learning algorithms",
                    "neural networks deep learning",
                    "natural language processing"
                ]
                
                for query in test_queries:
                    start_time = time.time()
                    search_results = faiss_db.search(query, k=3)
                    search_time = time.time() - start_time
                    
                    result = {
                        "test": "faiss_search",
                        "query": query,
                        "results_count": len(search_results),
                        "execution_time": search_time,
                        "top_score": search_results[0]["score"] if search_results else 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
    
    except Exception as e:
        result = {
            "test": "faiss_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    return results

def test_chromadb_operations():
    """Test ChromaDB vector database operations"""
    print("Testing ChromaDB Operations...")
    
    results = []
    
    try:
        # Initialize ChromaDB
        chroma_db = ChromaVectorDB()
        init_success = chroma_db.initialize()
        
        result = {
            "test": "chromadb_initialization",
            "success": init_success,
            "available": chroma_db.is_ready,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        if init_success:
            # Test adding documents
            start_time = time.time()
            add_success = chroma_db.add_documents(SAMPLE_DOCUMENTS)
            add_time = time.time() - start_time
            
            result = {
                "test": "chromadb_add_documents",
                "success": add_success,
                "documents_added": len(SAMPLE_DOCUMENTS),
                "execution_time": add_time,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            if add_success:
                # Test search
                test_queries = [
                    "machine learning algorithms",
                    "neural networks deep learning",
                    "natural language processing"
                ]
                
                for query in test_queries:
                    start_time = time.time()
                    search_results = chroma_db.search(query, k=3)
                    search_time = time.time() - start_time
                    
                    result = {
                        "test": "chromadb_search",
                        "query": query,
                        "results_count": len(search_results),
                        "execution_time": search_time,
                        "top_score": search_results[0]["score"] if search_results else 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
    
    except Exception as e:
        result = {
            "test": "chromadb_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    return results

def test_weaviate_operations():
    """Test Weaviate vector database operations (simulated)"""
    print("Testing Weaviate Operations...")
    
    results = []
    
    try:
        # Initialize Weaviate
        weaviate_db = WeaviateVectorDB()
        init_success = weaviate_db.initialize()
        
        result = {
            "test": "weaviate_initialization",
            "success": init_success,
            "available": weaviate_db.is_ready,
            "needs_setup": weaviate_db.needs_setup,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        # Test simulated operations
        add_success = weaviate_db.add_documents(SAMPLE_DOCUMENTS)
        result = {
            "test": "weaviate_add_documents",
            "success": add_success,
            "documents_added": len(SAMPLE_DOCUMENTS),
            "note": "Simulated operation",
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        # Test simulated search
        search_results = weaviate_db.search("machine learning", k=3)
        result = {
            "test": "weaviate_search",
            "query": "machine learning",
            "results_count": len(search_results),
            "note": "Simulated operation",
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    except Exception as e:
        result = {
            "test": "weaviate_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    return results

def test_benchmark_comparison():
    """Test benchmark comparison across databases"""
    print("Testing Benchmark Comparison...")
    
    test_queries = [
        "machine learning algorithms",
        "deep learning neural networks",
        "natural language processing",
        "computer vision applications"
    ]
    
    results = []
    
    for query in test_queries:
        start_time = time.time()
        benchmark_result = benchmark_databases(query)
        total_time = time.time() - start_time
        
        result = {
            "test": "benchmark_comparison",
            "query": query,
            "total_execution_time": total_time,
            "databases_tested": len(benchmark_result),
            "successful_databases": len([r for r in benchmark_result.values() if r["status"] == "Success"]),
            "timestamp": datetime.now().isoformat(),
            "detailed_results": benchmark_result
        }
        results.append(result)
    
    return results

def run_all_tests():
    """Run all vector database tests"""
    print("=" * 50)
    print("VECTOR DATABASE TESTS")
    print("=" * 50)
    
    # Run all tests
    status_results = test_database_status()
    print(f"Database Status Tests: {len(status_results)} test cases completed")
    
    faiss_results = test_faiss_operations()
    print(f"Faiss Tests: {len(faiss_results)} test cases completed")
    
    chromadb_results = test_chromadb_operations()
    print(f"ChromaDB Tests: {len(chromadb_results)} test cases completed")
    
    weaviate_results = test_weaviate_operations()
    print(f"Weaviate Tests: {len(weaviate_results)} test cases completed")
    
    benchmark_results = test_benchmark_comparison()
    print(f"Benchmark Tests: {len(benchmark_results)} test cases completed")
    
    # Save results
    os.makedirs("../test-results", exist_ok=True)
    
    # Save individual test results
    all_results = {
        "status_tests": status_results,
        "faiss_tests": faiss_results,
        "chromadb_tests": chromadb_results,
        "weaviate_tests": weaviate_results,
        "benchmark_tests": benchmark_results
    }
    
    # Save as JSON
    with open("../test-results/vector_database_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for test_type, test_results in all_results.items():
        for result in test_results:
            summary_row = {
                "test_type": test_type,
                "test_name": result.get("test", "unknown"),
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0),
                "timestamp": result.get("timestamp", "")
            }
            summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("../test-results/vector_database_test_summary.csv", index=False)
    
    # Generate overall summary
    total_tests = sum(len(results) for results in all_results.values())
    successful_tests = sum(1 for results in all_results.values() for result in results if result.get("success", False))
    
    summary = {
        "test_suite": "Vector Database Tests",
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
        "test_breakdown": {k: len(v) for k, v in all_results.items()},
        "timestamp": datetime.now().isoformat(),
        "results_files": [
            "vector_database_test_results.json",
            "vector_database_test_summary.csv"
        ]
    }
    
    with open("../test-results/vector_database_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nTest Results Saved:")
    print("- vector_database_test_results.json")
    print("- vector_database_test_summary.csv")
    print("- vector_database_summary.json")
    
    return summary

if __name__ == "__main__":
    summary = run_all_tests()
    print(f"\nAll tests completed! Total: {summary['total_tests']} tests, Success Rate: {summary['success_rate']:.1f}%")


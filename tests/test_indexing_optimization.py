"""
Test Indexing Optimization Utilities
Tests text cleaning and chunking functions with sample data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.helpers_basic import clean_text, apply_chunking
import json
from datetime import datetime

def test_text_cleaning():
    """Test text cleaning functionality with sample data"""
    print("Testing Text Cleaning...")
    
    # Sample test data
    test_texts = [
        "This is a SAMPLE text with    extra spaces, special characters @#$%, and inconsistent formatting.",
        "Another text with URLs like https://example.com and email@domain.com that need cleaning.",
        "Text with\n\nmultiple\n\nline breaks and    excessive    whitespace.",
        "UPPERCASE TEXT that needs normalization and cleaning!!!",
        "Mixed content: Visit www.example.com or email support@company.org for help."
    ]
    
    results = []
    
    for i, text in enumerate(test_texts):
        # Test different cleaning configurations
        configs = [
            {"remove_special": True, "normalize_spaces": True, "remove_urls": True},
            {"remove_special": False, "normalize_spaces": True, "remove_urls": True},
            {"remove_special": True, "normalize_spaces": False, "remove_urls": True},
            {"remove_special": True, "normalize_spaces": True, "remove_urls": False}
        ]
        
        for j, config in enumerate(configs):
            cleaned = clean_text(
                text, 
                config["remove_special"], 
                config["normalize_spaces"], 
                config["remove_urls"]
            )
            
            result = {
                "test_id": f"text_{i+1}_config_{j+1}",
                "original_text": text,
                "cleaned_text": cleaned,
                "original_length": len(text),
                "cleaned_length": len(cleaned),
                "reduction_percentage": ((len(text) - len(cleaned)) / len(text)) * 100,
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
    
    return results

def test_chunking_strategies():
    """Test chunking strategies with sample documents"""
    print("Testing Chunking Strategies...")
    
    # Sample documents
    sample_docs = [
        """Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.""",
        
        """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics and drug design.""",
        
        """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them."""
    ]
    
    results = []
    
    # Test different chunking configurations
    chunk_configs = [
        {"size": 100, "overlap": 10, "method": "Fixed Size"},
        {"size": 200, "overlap": 20, "method": "Fixed Size"},
        {"size": 150, "overlap": 15, "method": "Recursive"},
        {"size": 200, "overlap": 25, "method": "Sentence-based"}
    ]
    
    for doc_id, doc in enumerate(sample_docs):
        for config_id, config in enumerate(chunk_configs):
            chunks = apply_chunking(
                doc, 
                config["size"], 
                config["overlap"], 
                config["method"]
            )
            
            result = {
                "test_id": f"doc_{doc_id+1}_config_{config_id+1}",
                "document_id": doc_id + 1,
                "original_length": len(doc),
                "chunk_count": len(chunks),
                "chunk_sizes": [len(chunk) for chunk in chunks],
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                "config": config,
                "chunks": chunks,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
    
    return results

def run_all_tests():
    """Run all indexing optimization tests"""
    print("=" * 50)
    print("INDEXING OPTIMIZATION TESTS")
    print("=" * 50)
    
    # Run text cleaning tests
    cleaning_results = test_text_cleaning()
    print(f"Text Cleaning Tests: {len(cleaning_results)} test cases completed")
    
    # Run chunking tests
    chunking_results = test_chunking_strategies()
    print(f"Chunking Strategy Tests: {len(chunking_results)} test cases completed")
    
    # Save results
    os.makedirs("../test-results", exist_ok=True)
    
    # Save cleaning results
    cleaning_df = pd.DataFrame(cleaning_results)
    cleaning_df.to_csv("../test-results/text_cleaning_test_results.csv", index=False)
    
    with open("../test-results/text_cleaning_test_results.json", "w") as f:
        json.dump(cleaning_results, f, indent=2)
    
    # Save chunking results
    chunking_df = pd.DataFrame([
        {
            "test_id": r["test_id"],
            "document_id": r["document_id"],
            "original_length": r["original_length"],
            "chunk_count": r["chunk_count"],
            "avg_chunk_size": r["avg_chunk_size"],
            "chunk_method": r["config"]["method"],
            "chunk_size": r["config"]["size"],
            "overlap": r["config"]["overlap"],
            "timestamp": r["timestamp"]
        }
        for r in chunking_results
    ])
    chunking_df.to_csv("../test-results/chunking_test_results.csv", index=False)
    
    with open("../test-results/chunking_test_results.json", "w") as f:
        json.dump(chunking_results, f, indent=2)
    
    # Generate summary report
    summary = {
        "test_suite": "Indexing Optimization",
        "total_tests": len(cleaning_results) + len(chunking_results),
        "text_cleaning_tests": len(cleaning_results),
        "chunking_tests": len(chunking_results),
        "timestamp": datetime.now().isoformat(),
        "results_files": [
            "text_cleaning_test_results.csv",
            "text_cleaning_test_results.json",
            "chunking_test_results.csv", 
            "chunking_test_results.json"
        ]
    }
    
    with open("../test-results/indexing_optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nTest Results Saved:")
    print("- text_cleaning_test_results.csv")
    print("- text_cleaning_test_results.json")
    print("- chunking_test_results.csv")
    print("- chunking_test_results.json")
    print("- indexing_optimization_summary.json")
    
    return summary

if __name__ == "__main__":
    summary = run_all_tests()
    print(f"\nAll tests completed successfully! Total: {summary['total_tests']} tests")


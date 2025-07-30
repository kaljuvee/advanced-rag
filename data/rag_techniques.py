# Advanced RAG Techniques Data Structure

RAG_TECHNIQUES = {
    "indexing_optimization": {
        "title": "Indexing Optimization Techniques",
        "description": "Indexing optimization techniques improve the quality and structure of data before it enters the vector database.",
        "techniques": {
            "data_preprocessing": {
                "title": "Data Pre-processing",
                "description": "Cleans and prepares raw text data for optimal indexing and retrieval performance.",
                "methods": {
                    "text_cleaning": {
                        "name": "Text Cleaning",
                        "description": "Removes noise, special characters, and formatting artifacts",
                        "steps": ["Remove HTML tags", "Clean special characters", "Normalize whitespace", "Handle encoding issues"]
                    },
                    "normalization": {
                        "name": "Text Normalization", 
                        "description": "Standardizes text format and structure",
                        "steps": ["Convert to lowercase", "Remove punctuation", "Expand contractions", "Handle abbreviations"]
                    },
                    "deduplication": {
                        "name": "Deduplication",
                        "description": "Removes duplicate or near-duplicate content",
                        "steps": ["Exact match removal", "Fuzzy matching", "Semantic similarity", "Hash-based detection"]
                    }
                }
            },
            "chunking_strategies": {
                "title": "Chunking Strategies",
                "description": "Divides large documents into smaller, manageable pieces for better retrieval granularity.",
                "strategies": {
                    "fixed_size": {
                        "name": "Fixed Size Chunking",
                        "description": "Splits text into chunks of predetermined character or token count",
                        "pros": ["Simple implementation", "Predictable chunk sizes", "Fast processing"],
                        "cons": ["May break semantic units", "Context loss at boundaries", "Inflexible"]
                    },
                    "recursive": {
                        "name": "Recursive Character Text Splitting",
                        "description": "Hierarchically splits text using multiple separators",
                        "pros": ["Preserves structure", "Flexible boundaries", "Better context retention"],
                        "cons": ["More complex", "Variable chunk sizes", "Slower processing"]
                    },
                    "sentence_based": {
                        "name": "Sentence-based Chunking",
                        "description": "Splits text at sentence boundaries for semantic coherence",
                        "pros": ["Semantic coherence", "Natural boundaries", "Better readability"],
                        "cons": ["Variable sizes", "Language dependent", "Complex sentences"]
                    },
                    "semantic": {
                        "name": "Semantic Chunking",
                        "description": "Groups related content based on semantic similarity",
                        "pros": ["Semantic coherence", "Topic preservation", "Better retrieval"],
                        "cons": ["Computationally expensive", "Complex implementation", "Model dependent"]
                    }
                }
            }
        }
    },
    "pre_retrieval_optimization": {
        "title": "Pre-retrieval Optimization Techniques",
        "description": "Pre-retrieval optimization techniques enhance user queries before the retrieval process begins.",
        "techniques": {
            "query_transformation": {
                "name": "Query Transformation",
                "description": "Modifies and enhances user queries to improve retrieval effectiveness.",
                "methods": {
                    "query_rewriting": {
                        "name": "Query Rewriting",
                        "description": "Reformulates queries for better matching with indexed content",
                        "techniques": ["Synonym expansion", "Paraphrasing", "Grammar correction", "Terminology normalization"]
                    },
                    "query_expansion": {
                        "name": "Query Expansion", 
                        "description": "Adds related terms and concepts to broaden search scope",
                        "techniques": ["Synonym addition", "Related terms", "Context expansion", "Domain-specific terms"]
                    },
                    "query_clarification": {
                        "name": "Query Clarification",
                        "description": "Resolves ambiguities and adds missing context",
                        "techniques": ["Ambiguity resolution", "Context addition", "Intent clarification", "Scope definition"]
                    }
                }
            },
            "query_decomposition": {
                "name": "Query Decomposition",
                "description": "Breaks down complex queries into simpler, more manageable sub-queries.",
                "benefits": [
                    "Handles complex multi-part questions",
                    "Improves retrieval precision",
                    "Enables parallel processing",
                    "Better handles compound queries"
                ],
                "approaches": [
                    "Logical decomposition",
                    "Temporal decomposition", 
                    "Topical decomposition",
                    "Dependency-based decomposition"
                ]
            },
            "query_routing": {
                "name": "Query Routing",
                "description": "Directs queries to the most appropriate retrieval pipeline or data source.",
                "strategies": [
                    "Intent classification",
                    "Domain-specific routing",
                    "Complexity-based routing"
                ],
                "benefits": [
                    "Optimized retrieval paths",
                    "Improved efficiency",
                    "Better resource utilization",
                    "Enhanced accuracy"
                ]
            }
        }
    },
    "retrieval_optimization": {
        "title": "Retrieval Optimization Strategies",
        "description": "Retrieval optimization strategies improve retrieval results by directly manipulating how external data is retrieved in relation to the user query.",
        "techniques": {
            "metadata_filtering": {
                "name": "Metadata Filtering",
                "description": "Filters retrieved documents based on metadata attributes like date, author, category, or custom tags to improve relevance.",
                "filter_types": {
                    "temporal": {
                        "name": "Temporal Filtering",
                        "description": "Filter by time-based attributes like publication date, last modified, or creation time",
                        "use_cases": ["Recent news", "Historical analysis", "Time-sensitive queries"],
                        "examples": ["Last 30 days", "2023 publications", "Before 2020"]
                    },
                    "categorical": {
                        "name": "Categorical Filtering",
                        "description": "Filter by predefined categories, topics, or classification labels",
                        "use_cases": ["Domain-specific search", "Content type filtering", "Topic-based retrieval"],
                        "examples": ["Technology articles", "Medical papers", "Legal documents"]
                    },
                    "authorship": {
                        "name": "Authorship Filtering",
                        "description": "Filter by author, organization, or source credibility",
                        "use_cases": ["Expert opinions", "Trusted sources", "Institutional content"],
                        "examples": ["Peer-reviewed", "Government sources", "Specific authors"]
                    },
                    "quality": {
                        "name": "Quality Filtering",
                        "description": "Filter by quality metrics like citation count, rating, or review scores",
                        "use_cases": ["High-quality content", "Popular articles", "Well-reviewed sources"],
                        "examples": ["High citation count", "4+ star rating", "Editor's choice"]
                    }
                }
            },
            "hybrid_search": {
                "name": "Hybrid Search",
                "description": "Combines multiple search approaches (keyword, semantic, vector) to leverage the strengths of each method.",
                "methods": {
                    "keyword_search": {
                        "name": "Keyword Search",
                        "description": "Traditional text-based search using exact term matching and TF-IDF scoring",
                        "strengths": ["Fast execution", "Exact matches", "Interpretable results"],
                        "weaknesses": ["Limited semantic understanding", "Vocabulary mismatch", "No context awareness"]
                    },
                    "semantic_search": {
                        "name": "Semantic Search",
                        "description": "Vector-based search using embeddings to capture semantic meaning and context",
                        "strengths": ["Semantic understanding", "Context awareness", "Handles synonyms"],
                        "weaknesses": ["Computationally expensive", "Less precise for exact terms", "Black box results"]
                    },
                    "hybrid_approach": {
                        "name": "Hybrid Approach",
                        "description": "Weighted combination of keyword and semantic search results",
                        "strengths": ["Best of both worlds", "Balanced precision and recall", "Flexible weighting"],
                        "weaknesses": ["Complex tuning", "Higher computational cost", "Parameter sensitivity"]
                    }
                }
            },
            "vector_outliers": {
                "name": "Vector Search Outlier Exclusion",
                "description": "Identifies and excludes retrieved documents with unusually low similarity scores that may be irrelevant or noisy.",
                "methods": [
                    "Similarity threshold filtering",
                    "Statistical outlier detection (Z-score)",
                    "Percentile-based filtering",
                    "Clustering-based outlier detection",
                    "Isolation forest for anomaly detection"
                ],
                "benefits": [
                    "Improved result quality",
                    "Reduced noise in retrieved documents",
                    "Better user experience",
                    "More focused context for generation",
                    "Reduced computational overhead"
                ]
            },
            "embedding_finetuning": {
                "name": "Embedding Model Fine-tuning",
                "description": "Adapts pre-trained embedding models to specific domains or tasks to improve retrieval performance for specialized use cases.",
                "approaches": {
                    "domain_adaptation": {
                        "name": "Domain Adaptation",
                        "description": "Fine-tune embeddings on domain-specific data to capture specialized vocabulary and concepts",
                        "complexity": "Medium",
                        "use_cases": ["Medical texts", "Legal documents", "Technical manuals", "Scientific papers"]
                    },
                    "task_specific": {
                        "name": "Task-Specific Fine-tuning",
                        "description": "Optimize embeddings for specific retrieval tasks or query types",
                        "complexity": "High",
                        "use_cases": ["Question answering", "Document similarity", "Code search", "Image-text matching"]
                    },
                    "contrastive_learning": {
                        "name": "Contrastive Learning",
                        "description": "Train embeddings to distinguish between relevant and irrelevant document pairs",
                        "complexity": "High",
                        "use_cases": ["Improving retrieval precision", "Hard negative mining", "Similarity learning"]
                    },
                    "multi_task": {
                        "name": "Multi-task Learning",
                        "description": "Train embeddings on multiple related tasks simultaneously",
                        "complexity": "Very High",
                        "use_cases": ["General-purpose embeddings", "Cross-domain applications", "Transfer learning"]
                    }
                }
            }
        }
    },
    "post_retrieval_optimization": {
        "title": "Post-retrieval Optimization Techniques",
        "description": "Post-retrieval optimization techniques enhance the quality of generated responses after the retrieval process has been completed.",
        "techniques": {
            "reranking": {
                "title": "Re-ranking",
                "description": "Combines the speed of vector search with the contextual richness of a re-ranking model.",
                "process": [
                    "Vector search retrieves candidates quickly",
                    "Re-ranking model processes query and documents together",
                    "Captures more contextual nuances",
                    "Re-orders results to improve quality"
                ],
                "note": "Should over-retrieve chunks to filter out less relevant ones later"
            },
            "context_postprocessing": {
                "title": "Context Post-processing",
                "description": "Post-processes retrieved context for generation through enhancement or compression.",
                "methods": {
                    "context_enhancement": {
                        "name": "Context Enhancement with Metadata",
                        "description": "Enhances retrieved context with additional metadata information",
                        "technique": "Sentence window retrieval - chunks into smaller pieces but stores larger context window in metadata"
                    },
                    "context_compression": {
                        "name": "Context Compression",
                        "description": "Extracts only the most meaningful information from retrieved data",
                        "approaches": [
                            "Embedding-based compression",
                            "Lexical-based compression"
                        ],
                        "benefits": [
                            "Reduces data volume",
                            "Lowers retrieval and operational costs",
                            "Eliminates irrelevant content"
                        ]
                    }
                }
            },
            "prompt_engineering": {
                "title": "Prompt Engineering",
                "description": "Optimizes LLM prompts to improve the quality and accuracy of generated output.",
                "techniques": {
                    "chain_of_thought": {
                        "name": "Chain of Thought (CoT)",
                        "description": "Asks model to 'think step-by-step' and break down complex reasoning tasks",
                        "use_case": "When retrieved documents contain conflicting or dense information"
                    },
                    "tree_of_thoughts": {
                        "name": "Tree of Thoughts (ToT)",
                        "description": "Builds on CoT by evaluating responses at each step and generating multiple solutions",
                        "use_case": "When there are many potential pieces of evidence requiring evaluation"
                    },
                    "react": {
                        "name": "ReAct (Reasoning and Acting)",
                        "description": "Combines CoT with agents for dynamic interaction with external data sources",
                        "benefit": "Enables LLMs to dynamically interact with retrieved documents"
                    }
                }
            },
            "llm_fine_tuning": {
                "title": "LLM Fine-tuning",
                "description": "Fine-tunes pre-trained LLMs on domain-specific datasets to improve performance for specialized tasks.",
                "benefits": [
                    "Adapts general knowledge to specific domains",
                    "Improves quality of generated responses",
                    "Better handles specialized terminology",
                    "Captures domain-specific nuances"
                ],
                "data_types": {
                    "labeled": "Positive/negative examples for classification tasks",
                    "unlabeled": "Domain-specific articles for knowledge expansion"
                },
                "process": "Iterative weight updates through backpropagation on domain-specific dataset"
            }
        }
    },
    "post_retrieval_optimization": {
        "title": "Post-retrieval Optimization Techniques",
        "description": "Techniques that enhance the quality and relevance of retrieved information after the initial retrieval step",
        "techniques": {
            "reranking": {
                "name": "Re-ranking",
                "description": "Reorders retrieved documents based on additional relevance signals and query-document alignment",
                "methods": {
                    "cross_encoder": {
                        "name": "Cross-encoder Re-ranking",
                        "description": "Uses transformer models to score query-document pairs jointly",
                        "approach": "Deep learning based",
                        "best_for": "High accuracy requirements"
                    },
                    "semantic_similarity": {
                        "name": "Semantic Similarity Re-ranking",
                        "description": "Reorders based on semantic similarity between query and documents",
                        "approach": "Embedding based",
                        "best_for": "Conceptual relevance"
                    },
                    "query_document_alignment": {
                        "name": "Query-Document Alignment",
                        "description": "Measures how well document content aligns with query intent",
                        "approach": "Feature based",
                        "best_for": "Intent matching"
                    }
                }
            },
            "context_postprocessing": {
                "name": "Context Post-processing",
                "description": "Refines and optimizes retrieved context before passing to the generation model",
                "techniques": {
                    "redundancy_removal": {
                        "name": "Redundancy Removal",
                        "description": "Eliminates duplicate or highly similar information from retrieved context",
                        "purpose": "Reduce noise",
                        "impact": "Improved focus"
                    },
                    "summarization": {
                        "name": "Context Summarization",
                        "description": "Condenses retrieved information while preserving key details",
                        "purpose": "Length optimization",
                        "impact": "Better efficiency"
                    },
                    "key_extraction": {
                        "name": "Key Information Extraction",
                        "description": "Identifies and extracts the most relevant information from context",
                        "purpose": "Relevance enhancement",
                        "impact": "Higher precision"
                    },
                    "relevance_reordering": {
                        "name": "Relevance-based Reordering",
                        "description": "Reorders context segments by relevance to the query",
                        "purpose": "Priority optimization",
                        "impact": "Better attention"
                    }
                }
            },
            "prompt_engineering": {
                "name": "Prompt Engineering",
                "description": "Optimizes prompts to improve the quality of generated responses using retrieved context",
                "strategies": {
                    "chain_of_thought": {
                        "name": "Chain-of-Thought Prompting",
                        "description": "Encourages step-by-step reasoning in responses",
                        "use_case": "Complex reasoning tasks",
                        "example": "Let's think step by step about this problem..."
                    },
                    "few_shot": {
                        "name": "Few-shot Prompting",
                        "description": "Provides examples to guide response format and style",
                        "use_case": "Consistent output format",
                        "example": "Here are some examples of good responses..."
                    },
                    "zero_shot": {
                        "name": "Zero-shot Prompting",
                        "description": "Direct instruction without examples",
                        "use_case": "Simple, direct tasks",
                        "example": "Based on the context, answer the following question..."
                    },
                    "role_based": {
                        "name": "Role-based Prompting",
                        "description": "Assigns a specific role or persona to the AI",
                        "use_case": "Domain expertise simulation",
                        "example": "As an expert in machine learning, explain..."
                    }
                }
            },
            "llm_finetuning": {
                "name": "LLM Fine-tuning",
                "description": "Adapts language models for specific RAG tasks and domains",
                "approaches": {
                    "full_finetuning": {
                        "name": "Full Fine-tuning",
                        "description": "Updates all model parameters for specific tasks",
                        "complexity": "High",
                        "resources": "Significant computational resources",
                        "best_for": "Domain-specific applications"
                    },
                    "lora": {
                        "name": "LoRA (Low-Rank Adaptation)",
                        "description": "Efficient fine-tuning using low-rank matrices",
                        "complexity": "Medium",
                        "resources": "Moderate computational resources",
                        "best_for": "Task adaptation with limited resources"
                    },
                    "prompt_tuning": {
                        "name": "Prompt Tuning",
                        "description": "Learns optimal prompt embeddings for tasks",
                        "complexity": "Low",
                        "resources": "Minimal computational resources",
                        "best_for": "Quick task adaptation"
                    }
                },
                "considerations": {
                    "benefits": [
                        "Improved task-specific performance",
                        "Better domain adaptation",
                        "Reduced hallucination",
                        "Consistent output format"
                    ],
                    "challenges": [
                        "Requires quality training data",
                        "Computational overhead",
                        "Risk of overfitting",
                        "Model maintenance complexity"
                    ]
                }
            }
        }
    }
}

# Vector Database Information
VECTOR_DATABASES = {
    "faiss": {
        "name": "Faiss",
        "description": "Facebook AI Similarity Search - A library for efficient similarity search and clustering of dense vectors",
        "pros": ["Very fast", "Memory efficient", "Good for large-scale", "No external dependencies"],
        "cons": ["No built-in persistence", "Limited metadata support", "CPU/GPU specific builds"],
        "use_cases": ["Large-scale similarity search", "Real-time applications", "Memory-constrained environments"],
        "credentials_needed": False
    },
    "chromadb": {
        "name": "ChromaDB",
        "description": "Open-source embedding database designed to make it easy to build AI applications",
        "pros": ["Easy to use", "Built-in persistence", "Good metadata support", "Python-native"],
        "cons": ["Newer ecosystem", "Limited scalability", "Single-node by default"],
        "use_cases": ["Prototyping", "Small to medium datasets", "Local development"],
        "credentials_needed": False
    },
    "weaviate": {
        "name": "Weaviate",
        "description": "Open-source vector database that stores both objects and vectors",
        "pros": ["Production-ready", "GraphQL API", "Excellent metadata support", "Multi-modal"],
        "cons": ["More complex setup", "Resource intensive", "Learning curve"],
        "use_cases": ["Production applications", "Complex queries", "Multi-modal data"],
        "credentials_needed": True,
        "credentials_info": "Requires Weaviate Cloud (WCD) API key or self-hosted instance"
    }
}

# Sample data for demonstrations
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
        "metadata": {"category": "AI", "author": "John Doe", "date": "2023-01-15", "source": "tech_blog"}
    },
    {
        "id": "doc2", 
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "metadata": {"category": "AI", "author": "Jane Smith", "date": "2023-02-20", "source": "research_paper"}
    },
    {
        "id": "doc3",
        "title": "Natural Language Processing Basics",
        "content": "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
        "metadata": {"category": "NLP", "author": "Bob Johnson", "date": "2023-03-10", "source": "tutorial"}
    },
    {
        "id": "doc4",
        "title": "Vector Databases Explained",
        "content": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently for similarity search.",
        "metadata": {"category": "Database", "author": "Alice Brown", "date": "2023-04-05", "source": "tech_blog"}
    },
    {
        "id": "doc5",
        "title": "Retrieval Augmented Generation",
        "content": "RAG combines the power of large language models with external knowledge retrieval to provide more accurate and up-to-date responses.",
        "metadata": {"category": "AI", "author": "Charlie Wilson", "date": "2023-05-12", "source": "research_paper"}
    }
]

SAMPLE_QUERIES = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain vector similarity search",
    "What are the benefits of RAG systems?",
    "How does NLP process human language?"
]


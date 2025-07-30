# Advanced RAG Techniques MVP

A comprehensive Streamlit application demonstrating Advanced RAG (Retrieval-Augmented Generation) techniques based on the Weaviate Advanced RAG Techniques ebook. This MVP showcases various optimization strategies across the entire RAG pipeline with interactive demos, visualizations, and downloadable data.

## 🚀 Features

### 📊 Comprehensive RAG Technique Coverage
- **Indexing Optimization**: Data preprocessing and chunking strategies
- **Pre-retrieval Optimization**: Query decomposition, routing, and enhancement
- **Retrieval Optimization**: Metadata filtering, hybrid search, and performance optimization
- **Post-retrieval Optimization**: Re-ranking, context processing, prompt engineering, and LLM fine-tuning

### 🗄️ Vector Database Integration
- **Faiss**: CPU-based similarity search library by Facebook AI
- **ChromaDB**: Open-source embedding database
- **Weaviate**: Cloud-native vector database (setup required)

### 🛠️ Interactive Demonstrations
- Text cleaning and preprocessing tools
- Document chunking with multiple strategies
- Query decomposition and routing visualizations
- Hybrid search comparisons
- Re-ranking and context optimization
- Prompt engineering strategies

### 📈 Visualizations & Analytics
- Performance comparison charts
- Similarity score heatmaps
- Chunk size distributions
- Query routing visualizations
- Re-ranking effectiveness metrics

### 📥 Downloadable Data
- Technique comparison tables
- Test results and benchmarks
- Processing metrics
- Configuration examples

## 🏗️ Project Structure

```
advanced-rag-mvp/
├── Home.py                          # Main Streamlit entry point
├── pages/                           # Streamlit pages (following conventions)
│   ├── 1_Indexing_Optimization.py
│   ├── 2_Pre_Retrieval_Optimization.py
│   ├── 3_Retrieval_Optimization.py
│   ├── 4_Post_Retrieval_Optimization.py
│   └── 5_Vector_Database_Comparison.py
├── utils/                           # Business logic utilities
│   ├── helpers_basic.py            # Core helper functions
│   └── vector_db_utils.py          # Vector database utilities
├── data/                           # Data structures and configurations
│   └── rag_techniques.py          # RAG techniques data
├── tests/                          # Test suite
│   ├── test_indexing_optimization.py
│   ├── test_vector_databases.py
│   └── run_all_tests.py
├── test-results/                   # Test output directory
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/kaljuvee/advanced-rag.git
cd advanced-rag
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

### Start the Streamlit Application
```bash
streamlit run Home.py
```

The application will be available at `http://localhost:8501`

### Alternative: Specify Port and Host
```bash
streamlit run Home.py --server.port 8501 --server.address 0.0.0.0
```

## 🧪 Testing

### Run All Tests
```bash
cd tests
python run_all_tests.py
```

### Run Individual Test Suites
```bash
# Test indexing optimization utilities
python test_indexing_optimization.py

# Test vector database functionality
python test_vector_databases.py
```

### Test Results
Test results are automatically saved to the `test-results/` directory:
- CSV files for tabular data
- JSON files for detailed results
- Summary reports for each test suite

## 🗄️ Vector Database Setup

### Faiss (Ready to Use)
Faiss is included and ready to use out of the box. No additional setup required.

### ChromaDB (Ready to Use)
ChromaDB is included and ready to use locally. No additional setup required.

### Weaviate (Requires Setup)
Weaviate requires either:

#### Option 1: Weaviate Cloud
1. Sign up at [Weaviate Cloud](https://console.weaviate.cloud/)
2. Create a cluster
3. Get your cluster URL and API key
4. Update the connection settings in `utils/vector_db_utils.py`

#### Option 2: Local Docker Instance
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

## 📖 Usage Guide

### Navigation
The application uses Streamlit's multi-page structure:
- **Home**: Overview and navigation guide
- **Indexing Optimization**: Text preprocessing and chunking
- **Pre-retrieval Optimization**: Query transformation and routing
- **Retrieval Optimization**: Search strategies and filtering
- **Post-retrieval Optimization**: Re-ranking and prompt engineering
- **Vector Database Comparison**: Database benchmarking and comparison

### Interactive Demos
Each page includes interactive demonstrations:
1. **Configure parameters** using sliders, dropdowns, and input fields
2. **Run demonstrations** by clicking action buttons
3. **View results** in tables, charts, and visualizations
4. **Download data** using the download buttons

### Example Workflows

#### Text Processing Workflow
1. Navigate to "Indexing Optimization"
2. Use the "Text Cleaning" tab to clean sample text
3. Switch to "Chunking Strategies" to segment documents
4. Compare different chunking methods
5. Download results for analysis

#### Vector Database Comparison
1. Navigate to "Vector Database Comparison"
2. Review database status and features
3. Run performance benchmarks
4. Compare results across databases
5. Download benchmark data

## 🔧 Configuration

### Customizing Techniques
Edit `data/rag_techniques.py` to add or modify RAG techniques:
```python
RAG_TECHNIQUES = {
    "your_category": {
        "title": "Your Category Title",
        "description": "Category description",
        "techniques": {
            "your_technique": {
                "title": "Technique Title",
                "description": "Technique description"
            }
        }
    }
}
```

### Adding New Pages
1. Create a new file in `pages/` following the naming convention: `N_Page_Name.py`
2. Import required utilities from `utils/`
3. Follow the existing page structure and styling

### Extending Utilities
Add new functions to `utils/helpers_basic.py` or create new utility modules in `utils/`

## 🧪 Testing Framework

### Test Structure
- **Unit Tests**: Test individual utility functions
- **Integration Tests**: Test component interactions
- **Sample Data Tests**: Use realistic data for validation
- **Results Output**: All test results saved to `test-results/`

### Adding New Tests
1. Create test files in `tests/` directory
2. Follow the naming convention: `test_*.py`
3. Use sample data for realistic testing
4. Save results to `test-results/` directory

## 📊 Performance Considerations

### Optimization Tips
- **Large Documents**: Use appropriate chunk sizes (200-500 characters)
- **Vector Databases**: Choose based on your scale and requirements
- **Interactive Demos**: Results are simulated for demonstration purposes
- **Production Use**: Replace mock data with real embeddings and databases

### Scaling Recommendations
- **Small Projects**: ChromaDB for simplicity
- **Medium Projects**: Faiss for performance
- **Large Projects**: Weaviate for enterprise features

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add docstrings for new functions
- Include type hints where appropriate

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Weaviate**: For the comprehensive Advanced RAG Techniques ebook
- **Streamlit**: For the excellent web application framework
- **Vector Database Communities**: Faiss, ChromaDB, and Weaviate teams

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/kaljuvee/advanced-rag/issues)
- **Documentation**: Refer to this README and inline code comments
- **Community**: Join discussions in the repository

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with full RAG technique coverage
- Interactive demonstrations for all categories
- Vector database integration and comparison
- Comprehensive testing framework
- Complete documentation

---

**Built with ❤️ using Streamlit | Based on Weaviate's Advanced RAG Techniques ebook**


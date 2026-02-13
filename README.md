# Agentic RAG System

A Retrieval-Augmented Generation (RAG) system that combines vector databases with LLMs to provide intelligent document-based question answering.

## Overview

Agentic RAG is an AI-powered system that processes PDF documents and answers user queries by:
1. **Loading and Embedding**: Converting PDF documents into vector embeddings using Sentence Transformers
2. **Indexing**: Storing embeddings in FAISS vector database for fast similarity search
3. **Retrieval**: Finding relevant document chunks based on user queries
4. **Generation**: Presenting relevant context from documents for user queries

## Features

- 📄 **PDF Processing**: Automatically loads and chunks PDF documents
- 🔍 **Semantic Search**: Uses vector similarity to find relevant document sections
- 💾 **Vector Database**: FAISS-based indexing for fast retrieval
- 🤖 **LLM Integration**: OpenAI/Groq integration for advanced processing
- 🛠️ **Web Tools**: SerperDevTool and ScrapeWebsiteTool for extended capabilities
- 📊 **CrewAI Integration**: Multi-agent orchestration for complex tasks

## Project Structure

```
agenticRAG/
├── rag.py                    # Main RAG application
├── requirements.txt          # Project dependencies
├── genai-principles.pdf      # Sample PDF document
├── file_summary.txt          # Document summary
├── README.md                 # This file
└── .gitignore               # Git ignore configuration
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/AbinashB017/AgenticRAG.git
cd agenticRAG
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
SERPER_API_KEY=your_serper_key
GEMINI=your_gemini_key
LANGCHAIN_API_KEY=your_langchain_key
HF_TOKEN=your_huggingface_token
```

## Usage

### Running the RAG System

```bash
python rag.py
```

The system will:
1. Load and embed the PDF document
2. Process the example query: "What is Agentic RAG?"
3. Retrieve relevant document sections
4. Display the results

### Customizing Queries

Edit `rag.py` and modify the query variable in the `main()` function:

```python
def main():
    pdf_path = "genai-principles.pdf"
    query = "Your custom question here"  # Change this
    # ... rest of code
```

## Dependencies

Key libraries used:
- **LangChain**: LLM framework and document processing
- **FAISS**: Vector database for similarity search
- **Sentence Transformers**: Document embedding model
- **CrewAI**: Multi-agent orchestration
- **OpenAI**: GPT-based LLM integration
- **HuggingFace**: Embedding models and tools

See `requirements.txt` for complete list of dependencies.

## Architecture

### Vector Database Flow
```
PDF Document
    ↓
PyPDFLoader (Load)
    ↓
RecursiveCharacterTextSplitter (Chunk)
    ↓
HuggingFaceEmbeddings (Embed)
    ↓
FAISS (Index & Store)
```

### Query Processing Flow
```
User Query
    ↓
Vector Embedding
    ↓
Similarity Search (Top-K)
    ↓
Context Retrieval
    ↓
Output Generation
```

## Models

- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **LLM**: GPT-3.5-turbo (OpenAI) or LLama alternatives (Groq)
- **Chunk Size**: 1000 tokens with 50 token overlap

## Configuration

### Vector Database Parameters
- `chunk_size`: 1000 (number of characters per chunk)
- `chunk_overlap`: 50 (overlap between chunks)
- `similarity_search_k`: 5 (number of documents to retrieve)

### LLM Parameters
- `temperature`: 0.0 (for routing), 0.7 (for generation)
- `max_tokens`: 500

## Features

### Current Features
✅ PDF document loading and processing
✅ Vector embedding and indexing
✅ Semantic similarity search
✅ Document chunk retrieval
✅ Context-based question answering

### Planned Features
🚀 Web scraping for missing information
🚀 Multi-document RAG support
🚀 Advanced query routing
🚀 Caching for improved performance
🚀 API deployment

## Troubleshooting

### Common Issues

1. **Module Not Found Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Errors**
   - Ensure `.env` file exists with valid API keys
   - Check API key quotas and permissions

3. **CUDA/GPU Issues**
   - Falls back to CPU automatically
   - For GPU support, install CUDA-compatible PyTorch

4. **Memory Issues**
   - Reduce `chunk_size` or `max_tokens`
   - Use `faiss-cpu` for memory efficiency

## Performance Tips

- **First Run**: Embedding model download may take time
- **Query Speed**: Subsequent queries are ~100-500ms
- **Memory**: ~2GB for typical PDF documents
- **Optimization**: Use GPU for faster embeddings with large documents

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please open an issue on GitHub:
https://github.com/AbinashB017/AgenticRAG/issues

## Acknowledgments

- LangChain for document processing framework
- FAISS for vector similarity search
- Sentence Transformers for embeddings
- OpenAI and Groq for LLM APIs
- CrewAI for multi-agent orchestration

## References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Vector Database](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [CrewAI Framework](https://www.crewai.io/)

---

**Built with ❤️ for AI-powered document retrieval and question answering**

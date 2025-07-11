# REM RAG Backend

The Random Emergent Memory RAG system backend - a novel approach to building AI systems that develop genuine understanding through temporal synthesis and emergent pattern recognition.

## Structure

```
rem_rag_backend/
├── src/                    # Core backend code
│   ├── core/              # Main processing logic (reading/REM cycles)
│   ├── data_processing/   # Chunking, embedding
│   ├── vector_store/      # ChromaDB interface
│   ├── llm/              # LLM interfaces
│   ├── scraper/          # Foreign Affairs scraper
│   └── utils/            # Helper functions
├── scripts/               # Processing and utility scripts
│   ├── query/            # Query and testing scripts
│   ├── test/             # Test scripts
│   └── deprecated/       # Old versions for reference
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY=your-key-here

# Run the 2000s decade processing
python scripts/run_2000s_true_batch.py

# Query the system
python scripts/query/ask_rag.py "What patterns emerge in US foreign policy?"
```

## Key Components

- **Reading Cycle**: Processes documents chronologically, relating new information to existing knowledge
- **REM Cycle**: Random connection of disparate facts to discover emergent patterns
- **Batch Processing**: Uses OpenAI Batch API for 50% cost savings
- **Vector Store**: ChromaDB for local, lightweight storage with metadata filtering

## Data Flow

1. Articles are loaded from the Foreign Affairs dataset
2. Documents are chunked and embedded
3. Each chunk is related to existing knowledge (synthesis)
4. Periodically, REM cycles randomly connect knowledge nodes
5. Queries retrieve both direct matches and synthesized insights

See the main project README for more details on the philosophy and approach.
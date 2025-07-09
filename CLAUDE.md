# REM RAG - Random Emergent Memory RAG

A novel approach to building AI systems that develop genuine understanding through temporal synthesis and emergent pattern recognition.

## Project Overview

REM RAG is an experimental RAG (Retrieval-Augmented Generation) system that mimics human learning processes to build deep understanding of international politics from 100 years of Foreign Affairs articles. Unlike traditional RAG systems that simply retrieve and return information, REM RAG:

- **Learns incrementally** by relating each new piece of information to existing knowledge
- **Dreams** through random connection of disparate facts (REM cycles)  
- **Builds coherent models** of entities and concepts over time
- **Discovers patterns** not explicitly stated in any single document

## Key Innovation: Two-Phase Learning

### 1. READING Phase (Conscious Learning)
- Process documents chronologically
- Each new chunk is immediately related to existing knowledge
- Extract entities and build evolving models of actors/concepts
- Every piece of information passes through: "How does this relate to what we already know?"

### 2. REM Phase (Unconscious Synthesis)  
- Randomly sample 3 knowledge nodes (including 1 from current year)
- Ask: "What deeper pattern connects these passages?"
- Synthesize discoveries back into the knowledge base
- Run monthly during testing, yearly in production

## Data Source

- **Dataset**: https://huggingface.co/datasets/bitsinthesky/foreign_affairs_2024june20
- **Content**: 100 years of Foreign Affairs articles (subscriber access)
- **Why FA?**: Consistent quality, written by practitioners, covers full sweep of modern international relations

## Technical Stack

- **Vector Store**: ChromaDB (local, lightweight, excellent metadata filtering)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4o-mini (fast, cheap, good at structured tasks)
- **Orchestration**: LangGraph (state machine for READING→REM cycles)
- **Language**: Python 3.11+

## Project Structure

```
rem_rag_v2/
├── src/
│   ├── core/           # Main processing logic
│   ├── data_processing/# Chunking, embedding
│   ├── vector_store/   # ChromaDB interface
│   ├── llm/           # LLM interfaces
│   └── utils/         # Helper functions
├── docs/
│   ├── PHILOSOPHY.md  # Why we built this way
│   └── ARCHITECTURE.md# Technical design
├── tests/             # Test suites
├── notebooks/         # Experiments
└── data/             # Local data storage
```

## Quick Start

```bash
# Clone the repository
git clone [repository-url]
cd rem_rag_v2

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY=your-key-here

# Run 1962 prototype
python -m src.experiments.year_1962
```

## Core Concepts

### Entity Explosion
When we read "Khrushchev removed missiles from Cuba", we generate:
- What we learn about Khrushchev
- What we learn about Soviet Union
- What we learn about Cuba  
- What we learn about nuclear deterrence

Each learning is embedded separately and related to existing knowledge about that entity.

### Three-Node Synthesis
We use 3 nodes (not 2) in REM cycles because:
- 2 nodes = simple relationships ("A relates to B")
- 3 nodes = abstract patterns ("A, B, C share underlying principle P")

### Maximizing Connections
Every synthesis (from READING or REM) also asks: "How does this relate to what we already know?"
This ensures maximum interconnection of ideas.

## Experiments

### 1962 Prototype
Test the system on a single pivotal year:
- January-March: Berlin Crisis aftermath
- April-June: Laos negotiations
- July-September: Rising tensions
- October: Cuban Missile Crisis
- November-December: Aftermath

Track how understanding of key concepts (deterrence, brinksmanship, Cold War dynamics) evolves.

## Future Vision

### Browser Plugin
A Chrome extension that:
- Activates when reading news articles
- Provides historical context and patterns
- Highlights what's not being said
- Acts like a wise professor looking over your shoulder

### Evaluation Metrics
1. **Coherence**: Do entity models stay consistent?
2. **Prediction**: Can we anticipate expert commentary?
3. **Insight**: Do REM dreams surface non-obvious patterns?
4. **Robustness**: How many synthetic generations before quality degrades?

## Development Philosophy

This isn't just another RAG system. We're attempting to capture how understanding actually develops:
- Through continuous integration of new information
- Via unexpected connections during reflection
- With deep temporal awareness
- Building towards genuine wisdom, not just retrieval

## Current Status

- [x] Conceptual design complete
- [x] Architecture documented
- [ ] Core implementation
- [ ] 1962 prototype
- [ ] Full century processing
- [ ] Browser plugin
- [ ] Evaluation framework

## Contributing

This is an experimental research project. Key areas for contribution:
- Alternative embedding strategies
- Different REM cycle algorithms  
- Evaluation methodologies
- Entity resolution improvements

## Key Decisions Log

1. **Why ChromaDB?** Lightweight, local-first, great metadata filtering
2. **Why 3 nodes?** Forces abstract pattern recognition
3. **Why chronological?** Respects how understanding evolves
4. **Why Foreign Affairs?** Century of consistent, high-quality analysis

## Contact

[Your contact information]

---

*"The goal is not to build a system that knows about international politics, but one that understands it."*
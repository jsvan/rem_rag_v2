# REM RAG Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph State Machine               │
│                                                          │
│  ┌────────────┐         ┌────────────────┐             │
│  │   START    │────────▶│  READING Cycle  │             │
│  └────────────┘         └────────┬────────┘             │
│                                  │                       │
│                                  ▼                       │
│                         ┌────────────────┐               │
│                         │  Check End of  │               │
│                         │     Month?     │               │
│                         └────┬──────┬────┘               │
│                              │ No   │ Yes                │
│                              ▼      ▼                    │
│                         Continue  ┌────────────────┐     │
│                                  │   REM Cycle    │     │
│                                  │  (Monthly for  │     │
│                                  │  testing, yearly│     │
│                                  │  for production)│     │
│                                  └────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Processing Pipeline

```python
Document → Chunker → Embedder → Vector Store
                ↓
         Entity Extractor
                ↓
         Entity Analyzer → Vector Store
```

### 2. Vector Store Schema (ChromaDB)

```python
{
    "id": str,  # Unique identifier
    "text": str,  # The actual content
    "embedding": List[float],  # OpenAI text-embedding-3-small
    "metadata": {
        "type": str,  # "raw_chunk", "synthesis", "entity_insight", "rem_dream"
        "year": int,  # Year of reference
        "source_article_id": str,  # Original article
        "entity": Optional[str],  # For entity-specific insights
        "generation": int,  # 0 for raw, 1+ for synthetic
        "created_at": datetime,
        "parent_ids": List[str]  # IDs that generated this node
    }
}
```

### 3. Processing Algorithms

#### READING Cycle (Per Document)

```python
for chunk in document.chunks:
    # 1. Embed and store raw chunk
    chunk_id = vectorstore.add(chunk, type="raw_chunk")
    
    # 2. Find nearest neighbors
    neighbors = vectorstore.query(chunk, k=5)
    
    # 3. Generate synthesis
    synthesis_prompt = f"""
    New information: {chunk}
    
    Related knowledge: {neighbors}
    
    How does this new information relate to what we already know?
    Focus on how it confirms, contradicts, or extends existing understanding.
    """
    synthesis = llm.generate(synthesis_prompt)
    vectorstore.add(synthesis, type="synthesis", parent_ids=[chunk_id] + neighbor_ids)
    
    # 4. Extract and analyze entities
    entity_prompt = f"""
    List the entities and abstract concepts this passage refers to.
    For each, what do we learn about it?
    
    Format:
    - Entity/Concept: What we learn
    """
    entities = llm.generate(entity_prompt)
    
    # 5. Process each entity
    for entity, learning in parse_entities(entities):
        # Find what we already know about this entity
        entity_neighbors = vectorstore.query(
            learning, 
            k=3, 
            filter={"entity": entity}
        )
        
        entity_synthesis_prompt = f"""
        About {entity}, we just learned: {learning}
        
        Previous understanding: {entity_neighbors}
        
        How does this fit with what we already know about {entity}?
        """
        entity_synthesis = llm.generate(entity_synthesis_prompt)
        vectorstore.add(
            entity_synthesis, 
            type="entity_insight",
            entity=entity,
            parent_ids=[chunk_id]
        )
```

#### REM Cycle (Monthly/Yearly)

```python
def rem_cycle(num_dreams=100):
    for _ in range(num_dreams):
        # 1. Sample 3 random nodes (1 must be current year)
        current_year_node = vectorstore.sample(
            n=1, 
            filter={"year": current_year}
        )
        other_nodes = vectorstore.sample(n=2)
        nodes = [current_year_node] + other_nodes
        
        # 2. Generate connecting question
        question_prompt = f"""
        What is the implicit question at the heart which binds 
        the following passages together? Reply with the question only.
        
        Passage 1 ({nodes[0].year}): {nodes[0].text}
        Passage 2 ({nodes[1].year}): {nodes[1].text}
        Passage 3 ({nodes[2].year}): {nodes[2].text}
        """
        question = llm.generate(question_prompt)
        
        # 3. Find related knowledge
        related = vectorstore.query(question, k=5)
        
        # 4. Synthesize everything
        synthesis_prompt = f"""
        {SUMMARIZATION_PROMPT}
        
        Original passages:
        {format_nodes(nodes)}
        
        Related knowledge:
        {format_nodes(related)}
        """
        synthesis = llm.generate(synthesis_prompt)
        
        # 5. Store REM dream with neighbor synthesis
        synthesis_neighbors = vectorstore.query(synthesis, k=5)
        final_synthesis_prompt = f"""
        New synthesis: {synthesis}
        
        Related knowledge: {synthesis_neighbors}
        
        How does this new synthesis relate to what we already know?
        """
        final_synthesis = llm.generate(final_synthesis_prompt)
        
        vectorstore.add(
            final_synthesis,
            type="rem_dream",
            parent_ids=[n.id for n in nodes]
        )
```

### 4. Prompt Templates

```python
IMPLANT_SYNTHESIS_PROMPT = """New information: {new_info}

Existing knowledge: {existing_knowledge}

Write exactly 1 short paragraph (50-100 words) explaining how this new information relates to existing knowledge. Focus on:
- Key similarities or contradictions
- How it extends or challenges what we know
- The most important new insight (if any)

Be concise and specific. Keep your response to a single, impactful paragraph.

If this adds nothing new or is simply a repetition of what we already know, respond with exactly: NOTHING"""

REM_QUESTION_PROMPT = """What is the implicit question at the heart which binds the following passages together? Reply with the question only."""

REM_SYNTHESIS_PROMPT = """Write a concise synthesis in exactly 1-2 short paragraphs (no more than 150 words total) that:
1. Answers the implicit question
2. Reveals non-obvious patterns across the time periods
3. Offers insights that emerge from the juxtaposition

Be direct and specific. Focus on the most important insight."""
```

### 5. LangGraph State Definition

```python
class REMState(TypedDict):
    current_document: Document
    current_chunk: str
    current_year: int
    documents_processed: int
    month: int
    short_term_memory: List[str]
    entity_queue: List[Tuple[str, str]]  # (entity, learning)
```

### 6. Key Design Decisions

1. **Embedding Model**: OpenAI `text-embedding-3-small`
   - Good performance/cost ratio
   - 1536 dimensions
   - Handles nuanced political text well

2. **LLM**: GPT-4o-mini for all synthesis
   - Fast and cheap for our use case
   - Good at following structured prompts
   - Concise outputs (50-150 words)

3. **Vector Store**: ChromaDB
   - Local-first, no external dependencies
   - Excellent metadata filtering
   - Scales to millions of vectors

4. **Chunking Strategy**: Sentence-aware chunking
   - 300-word chunks (not token-based)
   - Splits at sentence boundaries
   - Maintains semantic coherence
   - Based on elegant recursive algorithm

5. **Implant Architecture**: 
   - Modular function used everywhere
   - Creates separate synthesis nodes
   - Filters redundant information
   - Preserves original content

### 7. Performance Considerations

- **Embedding Cache**: Store embeddings with documents to avoid re-computation
- **Neighbor Limits**: K=5 for most queries (tunable)
- **REM Frequency**: Monthly for testing, yearly for production
- **Synthesis Depth**: Track generation number to analyze impact

### 8. Future Optimizations

1. **Pruning**: Remove highly similar synthetic nodes (cosine > 0.95)
2. **Hierarchical Clustering**: Group related syntheses
3. **Temporal Weighting**: Recent events get higher relevance
4. **Cross-validation**: Test synthetic nodes against held-out articles

### 9. Evaluation Metrics

1. **Coherence**: Do entity models remain consistent?
2. **Coverage**: What % of expert commentary do we predict?
3. **Novelty**: Do REM dreams surface non-obvious connections?
4. **Depth**: How many generations until quality degrades?
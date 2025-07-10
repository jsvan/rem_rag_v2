# Gradio Frontend with MCP-Style Tooling for REM RAG

## Overview

This document outlines the architecture and implementation plan for a Gradio-based frontend that uses MCP (Model Context Protocol) style tooling to enable iterative query refinement and knowledge exploration in the REM RAG system.

## Architecture Components

### 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Chat Panel  │  │ Query Trace  │  │ Knowledge Graph │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Orchestration Layer                       │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Conversation │  │Query Agent  │  │ Response        │   │
│  │ Manager      │  │(Tool User)  │  │ Synthesizer     │   │
│  └──────────────┘  └─────────────┘  └─────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      MCP Tool Server                         │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Knowledge    │  │ Entity      │  │ Temporal        │   │
│  │ Search       │  │ Explorer    │  │ Pattern Finder  │   │
│  └──────────────┘  └─────────────┘  └─────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  REM RAG Vector │
                    │     Store       │
                    └─────────────────┘
```

### 2. Directory Structure

```
frontend/
├── FRONTEND_TODO.md            # This file
├── gradio_app.py              # Main Gradio application
├── components/
│   ├── __init__.py
│   ├── chat_interface.py      # Chat UI component
│   ├── query_trace.py         # Query exploration visualizer
│   └── knowledge_graph.py     # Interactive knowledge graph
├── mcp_tools/
│   ├── __init__.py
│   ├── tools.py               # MCP tool definitions
│   ├── server.py              # MCP server implementation
│   └── client.py              # MCP client for Gradio
├── agents/
│   ├── __init__.py
│   ├── query_agent.py         # Iterative query refinement
│   ├── synthesis_agent.py     # Response generation
│   └── prompts.py             # Agent prompts
└── utils/
    ├── __init__.py
    ├── conversation_state.py  # Chat state management
    └── formatters.py          # Response formatting
```

## MCP Tool Specifications

### Core Tools

```python
@tool
def search_knowledge(
    query: str,
    node_types: List[str] = None,
    year_range: Tuple[int, int] = None,
    limit: int = 10
) -> SearchResult:
    """
    Search the knowledge base with filters.
    
    Returns:
        SearchResult with chunks, metadata, and relevance scores
    """

@tool
def explore_entity(
    entity: str,
    include_related: bool = True,
    depth: int = 1
) -> EntityKnowledge:
    """
    Get comprehensive knowledge about an entity.
    
    Returns:
        EntityKnowledge with evolution over time, relationships
    """

@tool
def find_temporal_patterns(
    topic: str,
    min_year: int = None,
    max_year: int = None
) -> List[TemporalPattern]:
    """
    Identify how a topic evolved over time.
    
    Returns:
        List of patterns with supporting evidence
    """

@tool
def synthesize_follow_up_queries(
    initial_results: List[dict],
    original_question: str,
    gaps_identified: List[str]
) -> List[str]:
    """
    Generate queries to fill knowledge gaps.
    
    Returns:
        List of refined queries targeting specific gaps
    """

@tool
def get_rem_insights(
    topic: str,
    limit: int = 5
) -> List[REMInsight]:
    """
    Retrieve REM synthesis nodes about a topic.
    
    Returns:
        High-level insights from REM cycles
    """
```

## Gradio Interface Design

### Main Components

#### 1. Chat Interface
```python
# components/chat_interface.py
class ChatInterface:
    def __init__(self):
        self.history = []
        self.current_context = []
    
    def create_interface(self):
        with gr.Column():
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="Ask about international politics...",
                label="Your Question"
            )
            
            with gr.Row():
                submit = gr.Button("Ask")
                clear = gr.Button("Clear")
                
            # Advanced options
            with gr.Accordion("Advanced Options", open=False):
                max_iterations = gr.Slider(1, 5, 3, label="Max Query Iterations")
                show_reasoning = gr.Checkbox(True, label="Show Reasoning Process")
                compare_mode = gr.Checkbox(False, label="Compare with Raw GPT")
```

#### 2. Query Trace Panel
```python
# components/query_trace.py
class QueryTracePanel:
    def create_interface(self):
        with gr.Column():
            gr.Markdown("## Query Exploration Process")
            
            # Iteration tabs
            with gr.Tabs() as iterations:
                for i in range(5):  # Max 5 iterations
                    with gr.TabItem(f"Iteration {i+1}", visible=False) as tab:
                        queries = gr.JSON(label="Queries Generated")
                        results = gr.Dataframe(label="Results Retrieved")
                        gaps = gr.Textbox(label="Gaps Identified")
```

#### 3. Knowledge Graph Viewer
```python
# components/knowledge_graph.py
class KnowledgeGraphViewer:
    def create_interface(self):
        with gr.Column():
            gr.Markdown("## Knowledge Connections")
            
            # Interactive graph using Plotly
            graph = gr.Plot(label="Entity Relationships")
            
            # Node details
            selected_node = gr.JSON(label="Selected Node Details")
            
            # Graph controls
            with gr.Row():
                depth = gr.Slider(1, 3, 1, label="Connection Depth")
                layout = gr.Dropdown(
                    ["force", "circular", "hierarchical"],
                    value="force",
                    label="Layout"
                )
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up MCP tool server with basic tools
- [ ] Create minimal Gradio interface with chat
- [ ] Implement conversation state management
- [ ] Basic query agent with single iteration

### Phase 2: Iterative Refinement (Week 2)
- [ ] Implement multi-iteration query refinement
- [ ] Add query trace visualization
- [ ] Create synthesis agent for responses
- [ ] Add comparison mode (RAG vs Raw)

### Phase 3: Advanced Features (Week 3)
- [ ] Knowledge graph visualization
- [ ] Entity exploration interface
- [ ] Temporal pattern detection
- [ ] Export conversation history

### Phase 4: Polish & Optimization (Week 4)
- [ ] Performance optimization
- [ ] Better error handling
- [ ] User preferences/settings
- [ ] Deploy with Docker

## Example Workflow

### User Journey

1. **User asks**: "What patterns emerged about sovereignty?"

2. **Initial Query Generation**:
   ```
   - "sovereignty patterns international relations"
   - "national sovereignty evolution 20th century"
   - "sovereignty challenges globalization"
   ```

3. **First Iteration Results**: 30 chunks retrieved

4. **Gap Analysis**:
   ```
   - Missing: Early 20th century perspective
   - Missing: Economic sovereignty aspects
   - Missing: Non-Western perspectives
   ```

5. **Follow-up Queries**:
   ```
   - "sovereignty League of Nations 1920s"
   - "economic sovereignty trade agreements"
   - "post-colonial sovereignty Africa Asia"
   ```

6. **Second Iteration**: 20 additional chunks

7. **Final Synthesis**: Comprehensive response with 50 sources

## Key Implementation Details

### Query Agent Logic
```python
class IterativeQueryAgent:
    async def process_question(self, question: str, max_iterations: int = 3):
        context = []
        query_history = []
        
        for iteration in range(max_iterations):
            # Generate queries
            if iteration == 0:
                queries = await self.generate_initial_queries(question)
            else:
                gaps = await self.identify_gaps(context, question)
                if not gaps:
                    break
                queries = await self.generate_followup_queries(gaps, query_history)
            
            # Execute searches
            new_results = await self.execute_queries(queries)
            context.extend(new_results)
            query_history.extend(queries)
            
            # Check if we have sufficient context
            if await self.has_sufficient_context(context, question):
                break
        
        return context, query_history
```

### MCP Tool Integration
```python
from mcp import ToolServer, tool

class REMRAGToolServer(ToolServer):
    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store
        self.register_tools()
    
    def register_tools(self):
        self.add_tool("search_knowledge", self.search_knowledge)
        self.add_tool("explore_entity", self.explore_entity)
        # ... register other tools
```

## Configuration

### Environment Variables
```bash
# .env
OPENAI_API_KEY=your-key
MCP_SERVER_PORT=8765
GRADIO_SERVER_PORT=7860
MAX_QUERY_ITERATIONS=3
ENABLE_KNOWLEDGE_GRAPH=true
```

### Gradio Settings
```python
# config.py
GRADIO_CONFIG = {
    "theme": "soft",
    "title": "REM RAG Explorer",
    "favicon": "🧠",
    "analytics_enabled": False,
    "cache_examples": True
}
```

## Testing Strategy

### Unit Tests
- Tool functionality
- Query generation logic
- Gap identification
- Response synthesis

### Integration Tests
- Full conversation flow
- Multi-iteration queries
- State management
- Error handling

### User Testing
- Response quality evaluation
- UI/UX feedback
- Performance under load
- Edge case handling

## Future Enhancements

1. **Collaborative Features**
   - Share conversations
   - Annotate responses
   - Expert review mode

2. **Advanced Visualization**
   - 3D knowledge graph
   - Timeline view
   - Heatmap of knowledge density

3. **Personalization**
   - User expertise level
   - Preferred response style
   - Custom query strategies

4. **Integration**
   - API endpoints
   - Slack/Discord bots
   - Browser extension

## Success Metrics

- **Query Coverage**: % of relevant knowledge retrieved
- **Iteration Efficiency**: Average iterations needed
- **Response Quality**: User satisfaction ratings
- **Performance**: Query latency and throughput
- **Engagement**: Conversation depth and duration

## Dependencies

```python
# requirements.txt
gradio>=4.0.0
mcp>=0.1.0  # Model Context Protocol
plotly>=5.0.0  # For knowledge graph
asyncio
aiohttp
pydantic>=2.0.0
python-dotenv
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your settings

# Run MCP server
python frontend/mcp_tools/server.py

# Run Gradio app
python frontend/gradio_app.py
```

---

This architecture enables the LLM to act as an intelligent research assistant, iteratively exploring the knowledge base to provide comprehensive, well-sourced answers to complex questions about international politics.
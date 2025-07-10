"""Configuration for the REM RAG Frontend"""

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Query Settings
DEFAULT_K_NEIGHBORS = 10
MAX_QUERY_ITERATIONS = 3
MAX_CHUNKS_PER_ITERATION = 30

# Response Generation Prompts
RAG_PROMPT = """You are an expert on international politics with deep knowledge from 100 years of Foreign Affairs articles. 

Based on the following context from our knowledge base, please answer the user's question with wisdom and insight.

Context passages:
{context}

User Question: {question}

Instructions:
1. Look for deeper patterns and fundamental dynamics rather than just summarizing the passages
2. Synthesize insights across the different time periods and contexts
3. Identify what's NOT being said or what tensions exist between different viewpoints
4. Focus on the "why" behind patterns, not just describing them
5. Draw out implications and paradoxes
6. If the context reveals something surprising or counter-intuitive, highlight it

Please provide a thoughtful, insightful answer that goes beyond surface-level summary. Think like a wise professor who can see patterns others miss.

IMPORTANT: 
- Keep your answer brief, to about 2 paragraphs
- Write at a 7th grade reading level using simple, clear language
- Format your response in plain text only. Do not use markdown formatting, bullet points, bold text, or any other formatting. Use regular paragraph structure with clear topic sentences."""

RAW_RESPONSE_PROMPT = """You are an expert on international politics and history. 

User Question: {question}

Please provide a thoughtful, insightful answer that identifies deep patterns and fundamental dynamics. Focus on wisdom and synthesis rather than just listing examples.

IMPORTANT: 
- Keep your answer brief, to about 2 paragraphs
- Write at a 7th grade reading level using simple, clear language
- Format your response in plain text only. Do not use markdown formatting, bullet points, bold text, or any other formatting. Use regular paragraph structure with clear topic sentences."""

# Iterative Query Prompts
INITIAL_QUERY_PROMPT = """Given this question about international politics: "{question}"

Generate 3 diverse search queries that would help find relevant information to answer this question comprehensively. Consider different angles, time periods, and aspects of the topic.

Return the queries as a JSON list."""

GAP_ANALYSIS_PROMPT = """Based on the following question and the context retrieved so far, identify what important information is still missing.

Question: {question}

Context retrieved:
{context_summary}

What key aspects, time periods, perspectives, or details are missing that would help provide a more complete answer? List 2-3 specific gaps."""

FOLLOWUP_QUERY_PROMPT = """Based on these identified gaps in our knowledge:
{gaps}

Generate 2-3 new search queries that would help fill these specific gaps. Make the queries targeted and specific.

Return the queries as a JSON list."""

# Gradio Interface Settings
GRADIO_CONFIG = {
    "theme": "soft",
    "title": "REM RAG Explorer - Foreign Affairs Knowledge Base",
    "favicon": "🧠",
    "analytics_enabled": False,
    "cache_examples": True,
    "examples": [
        "What patterns emerged about sovereignty?",
        "How has nuclear deterrence strategy evolved?",
        "What can we learn about great power competition?",
        "How do economic crises affect international cooperation?",
        "What role does technology play in modern diplomacy?"
    ]
}

# MCP Tool Server Settings
MCP_SERVER_CONFIG = {
    "host": "localhost",
    "port": 8765,
    "max_connections": 10,
    "timeout": 30
}

# UI Component Settings
CHAT_INTERFACE_CONFIG = {
    "height": 600,
    "placeholder": "Ask about international politics, foreign policy, or global affairs...",
    "submit_button_text": "Ask",
    "clear_button_text": "Clear Chat",
    "show_timestamps": True
}

QUERY_TRACE_CONFIG = {
    "max_iterations_display": 5,
    "show_raw_queries": True,
    "show_result_counts": True,
    "highlight_gaps": True
}

KNOWLEDGE_GRAPH_CONFIG = {
    "initial_depth": 1,
    "max_depth": 3,
    "node_size_by": "relevance",  # or "connections"
    "default_layout": "force",
    "color_by": "node_type"  # or "year", "entity"
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_ttl": 3600,  # 1 hour
    "batch_size": 10,
    "concurrent_requests": 3,
    "request_timeout": 30
}

# Export Settings
EXPORT_CONFIG = {
    "formats": ["markdown", "json", "pdf"],
    "include_metadata": True,
    "include_sources": True,
    "include_query_trace": False
}
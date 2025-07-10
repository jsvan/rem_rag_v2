"""Configuration settings for REM RAG"""

from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_DIMENSIONS = 1536

# Vector store settings
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "rem_rag_v2"

# Processing settings
CHUNK_SIZE = 1000  # tokens (for SmartChunker - deprecated)
CHUNK_OVERLAP = 100  # tokens (for SmartChunker - deprecated)
CHUNK_MAX_WORDS = 300  # words (for SentenceAwareChunker)
CHUNK_MIN_CHARS = 150  # minimum characters to keep a chunk
NEIGHBORS_COUNT = 3  # for similarity search
REM_NEIGHBORS = 3  # nodes to sample in REM cycle

# REM cycle settings
REM_CYCLE_FREQUENCY = "monthly"  # "monthly" for testing, "yearly" for production
REM_SCALING_FACTOR = 0.25  # n/4 scaling - REM cycles = total non-REM nodes / 4

# Prompts
SUMMARIZATION_PROMPT = """Summarize the following passages. Write a single unified answer, about 100 words long, and do not enumerate over each document in the context, or add your own analysis. Do not refer to "the texts", or "the passages", simply combine and distill their information together with as many details as possible, and a focus on the insights and wisdom contained within. Mention the years at play in your answer."""

SYNTHESIS_PROMPT = """How does this new information relate to what we already know? Focus on how it confirms, contradicts, or extends existing understanding."""

ENTITY_EXTRACTION_PROMPT = """Extract entities and concepts from this passage ONLY where the passage provides substantive information about them.

For each entity where we learn something meaningful:
- Write a paragraph that includes the entity name and ONLY what this specific passage tells us about it
- Use direct evidence from the text
- Do not add external knowledge or interpretation

If an entity is merely mentioned without substantive information, skip it.
If the passage provides no meaningful learnings about any entities, return an empty list."""

# Updated implant synthesis prompt with NOTHING option
IMPLANT_SYNTHESIS_PROMPT = """New information: {new_info}

Existing knowledge: {existing_knowledge}

Based on comparing these passages, state the key insight or pattern that emerges. Write exactly 1 paragraph (50-100 words) that:

- States the actual pattern/insight directly (not how passages "relate")
- Uses specific terms and concepts from the passages
- Focuses on what we learn about the topic itself
- Avoids meta-language like "reinforces", "extends", "highlights"

Example good synthesis: "Humanitarian interventions create moral hazard by encouraging weaker parties to escalate conflicts expecting foreign support, while simultaneously nations hesitate to deploy ground troops, resulting in delayed interventions that fail to prevent atrocities like Rwanda."

If this adds nothing new or is simply repetition, respond with exactly: NOTHING"""

REM_QUESTION_PROMPT = """You are a wise historian pondering these three seemingly unrelated passages from different times and contexts. As you contemplate them together, a surprising question emerges - perhaps about a paradox, an echo across time, a pattern that breaks, or an assumption all three share without realizing it.

What fascinating question arises when you let these specific passages speak to each other? 

The best questions often:
- Notice what's absent as much as what's present
- Find the unexpected thread that connects disparate events  
- Reveal tensions between how things appear and how they actually work
- Discover what all three passages take for granted
- Point to patterns that only become visible across long time spans

Let your mind wander and find the question that makes someone say "I never thought of it that way."

Reply with only the question itself."""

ARTICLE_SUMMARY_PROMPT = """Summarize this article in 1-2 paragraphs (100-150 words). 
Focus on the main arguments, key insights, and conclusions. 
Be direct and specific, avoiding meta-commentary about the article itself."""

# Node type definitions for tracking different kinds of knowledge
NODE_TYPES = {
    "learning": "Original entity learning extracted from text",
    "synthesis": "Valuable synthesis comparing new and existing knowledge",
    "rem": "Pattern discovered during REM cycle",
    "learning_nothing": "Learning that added nothing new to existing knowledge",
    "chunk": "Original text chunk from article",
    "summary": "Article-level summary capturing main insights"
}
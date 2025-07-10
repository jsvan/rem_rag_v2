#!/usr/bin/env python3
"""
Simple RAG interface for asking questions about the knowledge base.

Usage:
    python ask_rag.py "What did we learn about nuclear deterrence?"
    python ask_rag.py --interactive
    python ask_rag.py --show-sources "What patterns emerged about sovereignty?"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vector_store.chromadb_store import REMVectorStore
from src.llm.openai_client import LLMClient
from src.config import LLM_MODEL


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


def format_context(results: Dict) -> str:
    """Format search results into context string."""
    context_parts = []
    
    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])
    
    for i, (text, metadata) in enumerate(zip(documents, metadatas)):
        context_part = f"[{i+1}] "
        context_part += f"({metadata.get('node_type', 'unknown')}, "
        context_part += f"{metadata.get('year', 'unknown')})"
        
        if metadata.get('title'):
            context_part += f" - {metadata['title']}"
        if metadata.get('entity'):
            context_part += f" - About: {metadata['entity']}"
            
        # Append year directly to text for better temporal reasoning
        year = metadata.get('year', 'unknown')
        context_part += f"\n{text} (Year: {year})\n"
        context_parts.append(context_part)
    
    return "\n---\n".join(context_parts)


def format_sources(results: Dict) -> str:
    """Format sources for display."""
    sources = []
    
    metadatas = results.get('metadatas', [])
    
    for i, metadata in enumerate(metadatas):
        source = f"{i+1}. "
        source += f"{metadata.get('node_type', 'unknown').upper()} "
        source += f"({metadata.get('year', 'unknown')})"
        
        if metadata.get('title'):
            source += f" - {metadata['title']}"
        if metadata.get('entity'):
            source += f" - Entity: {metadata['entity']}"
            
        sources.append(source)
    
    return "\n".join(sources)


async def ask_question(
    question: str,
    vector_store: REMVectorStore,
    llm_client: LLMClient,
    show_sources: bool = False,
    show_chunks: bool = True,
    k: int = 10
) -> str:
    """Ask a question using RAG."""
    
    # First, generate a raw OpenAI response without any context
    print("\n" + "="*80)
    print("GENERATING RAW OPENAI RESPONSE (NO CONTEXT)...")
    print("="*80)
    
    raw_prompt = f"""You are an expert on international politics and history. 

User Question: {question}

Please provide a thoughtful, insightful answer that identifies deep patterns and fundamental dynamics. Focus on wisdom and synthesis rather than just listing examples.

IMPORTANT: 
- Keep your answer brief, to about 2 paragraphs
- Write at a 7th grade reading level using simple, clear language
- Format your response in plain text only. Do not use markdown formatting, bullet points, bold text, or any other formatting. Use regular paragraph structure with clear topic sentences."""
    
    raw_response = await llm_client.generate(raw_prompt)
    
    # Search for relevant context
    print(f"\nSearching for relevant context (k={k})...")
    results = vector_store.query(question, k=k)
    
    if not results or not results.get('documents'):
        return "No relevant context found in the knowledge base."
    
    print(f"Found {len(results['documents'])} relevant passages.")
    
    # Format context
    context = format_context(results)
    
    # Print the chunks if requested
    if show_chunks:
        print("\n" + "="*80)
        print("RETRIEVED CHUNKS")
        print("="*80)
        
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        
        for i, (text, metadata) in enumerate(zip(documents, metadatas)):
            print(f"\n{'='*40} CHUNK {i+1} {'='*40}")
            print(f"Type: {metadata.get('node_type', 'unknown')}")
            print(f"Year: {metadata.get('year', 'unknown')}")
            if metadata.get('title'):
                print(f"Title: {metadata['title']}")
            if metadata.get('entity'):
                print(f"Entity: {metadata['entity']}")
            print("-" * 80)
            print(text)
            print("-" * 80)
    
    # Generate RAG response
    print("\n" + "="*80)
    print("GENERATING RAG-ENHANCED RESPONSE...")
    print("="*80)
    
    prompt = RAG_PROMPT.format(context=context, question=question)
    rag_response = await llm_client.generate(prompt)
    
    # Combine both responses
    combined_response = f"""RAW OPENAI RESPONSE (NO CONTEXT):
{'-'*80}
{raw_response}

RAG-ENHANCED RESPONSE (WITH CONTEXT):
{'-'*80}
{rag_response}"""
    
    # Add sources if requested
    if show_sources:
        sources = format_sources(results)
        combined_response += f"\n\nSOURCES:\n{sources}"
    
    return combined_response


async def interactive_mode(vector_store: REMVectorStore, llm_client: LLMClient):
    """Run in interactive mode."""
    print("\nREM RAG Interactive Mode")
    print("Ask questions about international politics from 100 years of Foreign Affairs.")
    print("Type 'quit' or 'exit' to end.\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            response = await ask_question(
                question, 
                vector_store, 
                llm_client,
                show_sources=True
            )
            
            print("\n" + "="*80)
            print("COMPARISON: RAW vs RAG-ENHANCED RESPONSES")
            print("="*80)
            print(f"\n{response}\n")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about the REM RAG knowledge base"
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (if not provided, runs in interactive mode)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--show-sources", "-s",
        action="store_true",
        help="Show source passages used for the answer"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of context passages to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Don't display the retrieved chunks (only show the response)"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing REM RAG...")
    vector_store = REMVectorStore()
    llm_client = LLMClient()
    
    # Check if we have data
    stats = vector_store.get_stats()
    print(f"Knowledge base contains {stats['total_documents']} nodes")
    
    if stats['total_documents'] == 0:
        print("Warning: Knowledge base is empty. Please run processing scripts first.")
        return
    
    # Run in appropriate mode
    if args.interactive or not args.question:
        await interactive_mode(vector_store, llm_client)
    else:
        response = await ask_question(
            args.question,
            vector_store,
            llm_client,
            show_sources=args.show_sources,
            show_chunks=not args.no_chunks,
            k=args.k
        )
        print("\n" + "="*80)
        print("COMPARISON: RAW vs RAG-ENHANCED RESPONSES")
        print("="*80)
        print(f"\n{response}\n")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test script to compare the original SmartChunker with the new SentenceAwareChunker
"""

import os
import sys
import json
import glob

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_processing.chunker import SmartChunker
from src.data_processing.sentence_chunker import SentenceAwareChunker


def load_sample_article():
    """Load a sample 1922 article for testing"""
    data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    pattern = os.path.join(data_dir, "1922_*.json")
    
    files = sorted(glob.glob(pattern))
    if not files:
        print("No 1922 articles found!")
        return None
    
    # Load the first article
    with open(files[0], 'r', encoding='utf-8') as f:
        article_data = json.load(f)
        
    return {
        'text': article_data.get('content', ''),
        'title': article_data.get('title', 'Unknown'),
        'author': article_data.get('author', 'Unknown'),
        'year': article_data.get('year', 1922),
        'article_id': 'test-001'
    }


def analyze_chunks(chunks, chunker_name):
    """Analyze and display chunk statistics"""
    print(f"\n{'='*60}")
    print(f"📊 {chunker_name} Analysis")
    print(f"{'='*60}")
    
    if not chunks:
        print("No chunks created!")
        return
    
    # Calculate statistics
    word_counts = []
    char_counts = []
    
    for chunk in chunks:
        text = chunk['text']
        # Remove article context if present
        if '[Article:' in text:
            text = text.split('\n\n', 1)[1] if '\n\n' in text else text
        
        word_counts.append(len(text.split()))
        char_counts.append(len(text))
    
    avg_words = sum(word_counts) / len(word_counts)
    avg_chars = sum(char_counts) / len(char_counts)
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Average words per chunk: {avg_words:.1f}")
    print(f"Average chars per chunk: {avg_chars:.1f}")
    print(f"Word count range: {min(word_counts)} - {max(word_counts)}")
    print(f"Char count range: {min(char_counts)} - {max(char_counts)}")
    
    # Show first chunk as example
    print(f"\n📄 Example chunk (first chunk):")
    print("-" * 40)
    first_chunk = chunks[0]['text']
    if len(first_chunk) > 500:
        print(first_chunk[:500] + "...")
    else:
        print(first_chunk)
    print("-" * 40)
    print(f"Word count: {len(first_chunk.split())}")
    print(f"Character count: {len(first_chunk)}")
    
    # Check for sentence boundaries
    print("\n🔍 Sentence boundary check (first 3 chunks):")
    for i, chunk in enumerate(chunks[:3]):
        text = chunk['text']
        if '[Article:' in text:
            text = text.split('\n\n', 1)[1] if '\n\n' in text else text
        
        # Check start and end
        start = text[:50] if len(text) > 50 else text
        end = text[-50:] if len(text) > 50 else text
        
        print(f"\nChunk {i+1}:")
        print(f"  Start: {start}...")
        print(f"  End: ...{end}")
        
        # Check if ends with sentence
        last_char = text.strip()[-1] if text.strip() else ''
        print(f"  Ends with sentence: {'Yes' if last_char in '.!?' else 'No'} ('{last_char}')")


def main():
    """Run the comparison"""
    print("🔬 Chunker Comparison Test")
    print("=" * 60)
    
    # Load sample article
    article = load_sample_article()
    if not article:
        return
    
    print(f"\n📚 Testing with article: {article['title']}")
    print(f"Original text length: {len(article['text'])} chars, {len(article['text'].split())} words")
    
    # Test SmartChunker
    print("\n\n1️⃣ Testing SmartChunker (token-based, 1000 tokens)")
    smart_chunker = SmartChunker()
    try:
        smart_chunks = smart_chunker.chunk_article(article)
        analyze_chunks(smart_chunks, "SmartChunker")
    except Exception as e:
        print(f"❌ SmartChunker failed: {e}")
        smart_chunks = []
    
    # Test SentenceAwareChunker
    print("\n\n2️⃣ Testing SentenceAwareChunker (word-based, 300 words)")
    sentence_chunker = SentenceAwareChunker()
    try:
        sentence_chunks = sentence_chunker.chunk_article(article)
        analyze_chunks(sentence_chunks, "SentenceAwareChunker")
    except Exception as e:
        print(f"❌ SentenceAwareChunker failed: {e}")
        sentence_chunks = []
    
    # Comparison summary
    print("\n\n📋 COMPARISON SUMMARY")
    print("=" * 60)
    
    if smart_chunks and sentence_chunks:
        print(f"SmartChunker: {len(smart_chunks)} chunks")
        print(f"SentenceAwareChunker: {len(sentence_chunks)} chunks")
        print(f"Chunk count ratio: {len(sentence_chunks) / len(smart_chunks):.2f}x")
        
        # Check sentence boundaries
        smart_sentence_ends = sum(1 for c in smart_chunks 
                                if c['text'].strip()[-1] in '.!?' if c['text'].strip() else False)
        sentence_sentence_ends = sum(1 for c in sentence_chunks 
                                   if c['text'].strip()[-1] in '.!?' if c['text'].strip() else False)
        
        print(f"\nChunks ending with sentence:")
        print(f"  SmartChunker: {smart_sentence_ends}/{len(smart_chunks)} "
              f"({100*smart_sentence_ends/len(smart_chunks):.1f}%)")
        print(f"  SentenceAwareChunker: {sentence_sentence_ends}/{len(sentence_chunks)} "
              f"({100*sentence_sentence_ends/len(sentence_chunks):.1f}%)")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
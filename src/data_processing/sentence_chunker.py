"""Sentence-aware chunker based on the original rem_rag design"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SentenceAwareChunker:
    """
    Sentence-aware chunking that splits on sentence boundaries.
    
    Based on the original rem_rag chunker design but with enhancements:
    - Splits at sentence boundaries (. ! ?)
    - Finds middle-most sentence for balanced chunks
    - Configurable word limits
    - Maintains article context
    """
    
    def __init__(self, max_words: int = 300, min_chars: int = 150):
        """
        Initialize the sentence-aware chunker.
        
        Args:
            max_words: Maximum words per chunk (default 300, slightly larger than original 250)
            min_chars: Minimum characters to keep a chunk (filters out very short text)
        """
        self.max_words = max_words
        self.min_chars = min_chars
    
    def split_long_paragraph(self, text: str) -> Tuple[str, str]:
        """
        Split a long paragraph at the middle-most sentence boundary.
        
        This is the key algorithm from the original chunker:
        finds the sentence boundary closest to the middle of the text.
        
        Args:
            text: Text to split
            
        Returns:
            Tuple of (first_part, second_part)
        """
        # Find all sentence boundaries (. ! ?)
        sentence_endings = []
        for i, char in enumerate(text):
            if char in '.!?' and i < len(text) - 1:
                # Make sure it's followed by space or end of text
                if i == len(text) - 1 or text[i + 1].isspace():
                    sentence_endings.append(i)
        
        if not sentence_endings:
            # No sentence boundaries found, try semicolons as fallback
            for i, char in enumerate(text):
                if char == ';':
                    sentence_endings.append(i)
        
        if not sentence_endings:
            # Still no boundaries, split at middle word
            words = text.split()
            mid_word = len(words) // 2
            if mid_word > 0:
                first_part = ' '.join(words[:mid_word])
                second_part = ' '.join(words[mid_word:])
                return first_part, second_part
            else:
                return text, ""
        
        # Find the sentence boundary closest to the middle
        mid = len(text) // 2
        closest_boundary = min(sentence_endings, key=lambda x: abs(x - mid))
        
        # Split at the boundary (include the punctuation in first part)
        return text[:closest_boundary + 1].strip(), text[closest_boundary + 1:].strip()
    
    def split_paragraphs(self, document: str, article_title: str = "") -> List[str]:
        """
        Split document into chunks respecting sentence boundaries.
        
        Based on the original's recursive approach using a stack.
        
        Args:
            document: Full document text
            article_title: Title for context
            
        Returns:
            List of text chunks
        """
        # Split into initial paragraphs
        paragraphs = document.split('\n')
        
        good_chunks = []
        bad_chunks = []
        
        # Process each paragraph
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Add to bad_chunks for processing
            bad_chunks.append(para)
        
        # Process chunks that need splitting (using stack approach from original)
        while bad_chunks:
            test_subject = bad_chunks.pop(0)
            word_count = len(test_subject.split())
            
            if word_count > self.max_words:
                # Split the chunk
                first_part, second_part = self.split_long_paragraph(test_subject)
                
                # Add parts back for further processing if needed
                if first_part:
                    bad_chunks.insert(0, first_part)
                if second_part:
                    bad_chunks.insert(1, second_part)
                    
            elif len(test_subject) < self.min_chars:
                # Too short, skip it
                logger.debug(f"Skipping short chunk: {len(test_subject)} chars")
                continue
                
            else:
                # Good size, add context and keep it
                if article_title:
                    chunk_with_context = f"[Article: {article_title}]\n\n{test_subject}"
                else:
                    chunk_with_context = test_subject
                    
                good_chunks.append(chunk_with_context)
        
        return good_chunks
    
    def chunk_article(self, article: Dict) -> List[Dict]:
        """
        Chunk article using sentence-aware splitting.
        
        Args:
            article: Article dict with 'text', 'title', 'article_id', etc.
            
        Returns:
            List of chunk dicts with text and metadata
        """
        text = article.get('text', '')
        title = article.get('title', '')
        article_id = article.get('article_id', '')
        
        if not text:
            return []
        
        # Get chunks using the sentence-aware splitter
        text_chunks = self.split_paragraphs(text, title)
        
        # Create chunk dictionaries with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'article_id': article_id,
                    'article_title': title,
                    'chunk_index': i,
                    'year': article.get('year', 1922),
                    'total_chunks': len(text_chunks),
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text),
                    'chunker': 'sentence_aware'
                }
            })
        
        logger.info(
            f"Chunked article '{title}' into {len(chunks)} chunks "
            f"(original: {len(text.split())} words)"
        )
        
        return chunks
    
    def merge_tiny_chunks(self, chunks: List[Dict], min_words: int = 50) -> List[Dict]:
        """
        Merge chunks that are too tiny with adjacent chunks.
        
        Args:
            chunks: List of chunks
            min_words: Minimum words per chunk
            
        Returns:
            List of potentially merged chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            current_words = current['metadata']['word_count']
            
            # If chunk is too small and not the last one
            if current_words < min_words and i < len(chunks) - 1:
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                
                # Combine texts (remove duplicate title context if present)
                current_text = current['text']
                next_text = next_chunk['text']
                
                # Remove title context from second chunk if both have it
                if '[Article:' in current_text and '[Article:' in next_text:
                    next_text = re.sub(r'\[Article:.*?\]\n\n', '', next_text, count=1)
                
                merged_text = current_text + '\n\n' + next_text
                
                merged_chunk = {
                    'text': merged_text,
                    'metadata': {
                        **current['metadata'],
                        'word_count': len(merged_text.split()),
                        'char_count': len(merged_text),
                        'merged': True,
                        'merged_chunks': [current['metadata']['chunk_index'], 
                                        next_chunk['metadata']['chunk_index']]
                    }
                }
                
                merged.append(merged_chunk)
                i += 2  # Skip next chunk
            else:
                merged.append(current)
                i += 1
        
        # Re-index chunks
        for idx, chunk in enumerate(merged):
            chunk['metadata']['chunk_index'] = idx
            chunk['metadata']['total_chunks'] = len(merged)
        
        return merged
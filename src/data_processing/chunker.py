"""Smart paragraph-aware chunking for Foreign Affairs articles"""

import re
import logging
from typing import List, Dict, Optional, Tuple
import tiktoken

from ..config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class SmartChunker:
    """
    Intelligent chunking that respects paragraph boundaries and maintains context.
    
    Features:
    - Respects paragraph boundaries
    - Maintains semantic coherence
    - Adds context from article title
    - Tracks chunk relationships
    """
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize the smart chunker.
        
        Args:
            chunk_size: Target size in tokens (not characters)
            chunk_overlap: Token overlap between chunks
            model_name: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs, handling various formats.
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split on double newlines (most common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Handle single-line breaks that might be paragraphs
        expanded_paragraphs = []
        for para in paragraphs:
            # If a "paragraph" is very long, it might have single line breaks
            if self.count_tokens(para) > self.chunk_size * 1.5:
                # Try splitting on single line breaks
                sub_paras = para.split('\n')
                sub_paras = [p.strip() for p in sub_paras if p.strip()]
                
                # Only use sub-paragraphs if they're reasonable sizes
                if len(sub_paras) > 1 and all(len(p) > 50 for p in sub_paras):
                    expanded_paragraphs.extend(sub_paras)
                else:
                    expanded_paragraphs.append(para)
            else:
                expanded_paragraphs.append(para)
        
        return expanded_paragraphs
    
    def find_section_headers(self, paragraphs: List[str]) -> List[Tuple[int, str]]:
        """
        Identify potential section headers in paragraphs.
        
        Returns list of (index, header_text) tuples.
        """
        headers = []
        
        header_patterns = [
            # All caps headers
            (r'^[A-Z\s]{3,}$', 1.0),
            # Title case short lines
            (r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}$', 0.8),
            # Roman numerals
            (r'^(?:I{1,3}|IV|V|VI{1,3}|IX|X)\.\s*\w+', 0.9),
            # Numbered sections
            (r'^\d+\.\s*\w+', 0.9),
        ]
        
        for i, para in enumerate(paragraphs):
            if len(para) < 100:  # Headers are usually short
                for pattern, confidence in header_patterns:
                    if re.match(pattern, para):
                        headers.append((i, para))
                        break
        
        return headers
    
    def chunk_article(self, article: Dict) -> List[Dict]:
        """
        Chunk article intelligently while maintaining context.
        
        Returns list of chunks with metadata.
        """
        text = article['text']
        title = article.get('title', '')
        
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        if not paragraphs:
            return []
        
        # Find section headers
        headers = self.find_section_headers(paragraphs)
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_section = title  # Start with article title as section
        chunk_idx = 0
        
        # Add title context to beginning
        if title:
            title_context = f"[Article: {title}]\n\n"
            title_tokens = self.count_tokens(title_context)
        else:
            title_context = ""
            title_tokens = 0
        
        for i, para in enumerate(paragraphs):
            para_tokens = self.count_tokens(para)
            
            # Check if this is a section header
            header_info = next((h for h in headers if h[0] == i), None)
            if header_info:
                current_section = header_info[1]
            
            # Check if adding this paragraph would exceed chunk size
            if current_chunk and (current_tokens + para_tokens + title_tokens > self.chunk_size):
                # Create chunk from current content
                chunk_text = title_context + '\n\n'.join(current_chunk)
                
                chunks.append({
                    'text': chunk_text,
                    'article_id': article['article_id'],
                    'chunk_index': chunk_idx,
                    'year': article['year'],
                    'title': title,
                    'section': current_section,
                    'metadata': {
                        **article.get('metadata', {}),
                        'paragraph_start': i - len(current_chunk),
                        'paragraph_end': i - 1,
                        'token_count': self.count_tokens(chunk_text),
                        'has_title_context': bool(title)
                    }
                })
                
                chunk_idx += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last few paragraphs for overlap
                    overlap_tokens = 0
                    overlap_paras = []
                    
                    for para in reversed(current_chunk):
                        para_tok = self.count_tokens(para)
                        if overlap_tokens + para_tok <= self.chunk_overlap:
                            overlap_paras.insert(0, para)
                            overlap_tokens += para_tok
                        else:
                            break
                    
                    current_chunk = overlap_paras
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = title_context + '\n\n'.join(current_chunk)
            
            chunks.append({
                'text': chunk_text,
                'article_id': article['article_id'],
                'chunk_index': chunk_idx,
                'year': article['year'],
                'title': title,
                'section': current_section,
                'metadata': {
                    **article.get('metadata', {}),
                    'paragraph_start': len(paragraphs) - len(current_chunk),
                    'paragraph_end': len(paragraphs) - 1,
                    'token_count': self.count_tokens(chunk_text),
                    'has_title_context': bool(title),
                    'total_chunks': None  # Will be set below
                }
            })
        
        # Update total chunks count
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        logger.info(
            f"Chunked article '{title}' into {len(chunks)} chunks "
            f"(original: {len(paragraphs)} paragraphs, {self.count_tokens(text)} tokens)"
        )
        
        return chunks
    
    def merge_small_chunks(self, chunks: List[Dict], min_size: int = 200) -> List[Dict]:
        """
        Merge chunks that are too small with adjacent chunks.
        
        Args:
            chunks: List of chunks
            min_size: Minimum chunk size in tokens
            
        Returns:
            List of merged chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If chunk is too small and not the last one
            if current['metadata']['token_count'] < min_size and i < len(chunks) - 1:
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                
                merged_text = current['text'] + '\n\n' + next_chunk['text']
                merged_chunk = {
                    **current,
                    'text': merged_text,
                    'metadata': {
                        **current['metadata'],
                        'token_count': self.count_tokens(merged_text),
                        'merged': True,
                        'paragraph_end': next_chunk['metadata']['paragraph_end']
                    }
                }
                
                merged.append(merged_chunk)
                i += 2  # Skip next chunk since we merged it
            else:
                merged.append(current)
                i += 1
        
        # Re-index chunks
        for idx, chunk in enumerate(merged):
            chunk['chunk_index'] = idx
            chunk['metadata']['total_chunks'] = len(merged)
        
        return merged
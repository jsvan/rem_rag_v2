"""Foreign Affairs dataset loader from HuggingFace"""

import logging
from typing import List, Dict, Optional, Iterator
from datetime import datetime
import re

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ForeignAffairsLoader:
    """
    Load and preprocess Foreign Affairs articles from HuggingFace.
    
    Dataset: https://huggingface.co/datasets/bitsinthesky/foreign_affairs_2024june20
    """
    
    def __init__(self, dataset_name: str = "bitsinthesky/foreign_affairs_2024june20"):
        """Initialize the loader."""
        self.dataset_name = dataset_name
        self.dataset = None
        self.articles_df = None
    
    def load_dataset(self, split: str = "train") -> pd.DataFrame:
        """
        Load the Foreign Affairs dataset.
        
        Returns:
            DataFrame with articles
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            logger.warning("HUGGING_FACE_HUB_TOKEN not found in environment. If the dataset is private, this will fail.")
        
        # Load from HuggingFace
        self.dataset = load_dataset(self.dataset_name, split=split, token=hf_token)
        
        # Convert to pandas for easier manipulation
        self.articles_df = self.dataset.to_pandas()
        
        logger.info(f"Loaded {len(self.articles_df)} articles")
        
        # Parse years if not already present
        if 'year' not in self.articles_df.columns:
            self.articles_df['year'] = self.articles_df.apply(self._extract_year, axis=1)
        
        # Sort by year for chronological processing
        self.articles_df = self.articles_df.sort_values('year')
        
        return self.articles_df
    
    def _extract_year(self, row: pd.Series) -> Optional[int]:
        """
        Extract publication year from article metadata.
        
        Tries multiple strategies:
        1. Look for 'year' field
        2. Parse from 'date' or 'publication_date' field
        3. Extract from text content
        """
        # Check direct year field
        if 'year' in row and pd.notna(row['year']):
            try:
                return int(row['year'])
            except:
                pass
        
        # Check date fields
        for date_field in ['date', 'publication_date', 'published_date']:
            if date_field in row and pd.notna(row[date_field]):
                try:
                    # Try parsing as datetime
                    if isinstance(row[date_field], str):
                        # Look for year pattern
                        year_match = re.search(r'(19\d{2}|20\d{2})', row[date_field])
                        if year_match:
                            return int(year_match.group(1))
                except:
                    pass
        
        # Last resort: look in content
        if 'content' in row and pd.notna(row['content']):
            # Look for publication info at start
            content_start = str(row['content'])[:500]
            year_match = re.search(r'(19\d{2}|20\d{2})', content_start)
            if year_match:
                year = int(year_match.group(1))
                # Sanity check - Foreign Affairs started in 1922
                if 1922 <= year <= datetime.now().year:
                    return year
        
        logger.warning(f"Could not extract year for article: {row.get('title', 'Unknown')}")
        return None
    
    def get_articles_by_year(self, year: int) -> pd.DataFrame:
        """Get all articles from a specific year."""
        if self.articles_df is None:
            self.load_dataset()
        
        return self.articles_df[self.articles_df['year'] == year]
    
    def get_articles_by_year_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Get articles within a year range."""
        if self.articles_df is None:
            self.load_dataset()
        
        return self.articles_df[
            (self.articles_df['year'] >= start_year) & 
            (self.articles_df['year'] <= end_year)
        ]
    
    def iter_articles(self, start_year: Optional[int] = None) -> Iterator[Dict]:
        """
        Iterate through articles chronologically.
        
        Yields:
            Dict with article data including year
        """
        if self.articles_df is None:
            self.load_dataset()
        
        df = self.articles_df
        if start_year:
            df = df[df['year'] >= start_year]
        
        for _, row in df.iterrows():
            yield row.to_dict()
    
    def prepare_for_processing(self, article: Dict) -> Dict:
        """
        Prepare article for REM RAG processing.
        
        Returns dict with standardized fields:
        - text: Full article text
        - title: Article title
        - year: Publication year
        - article_id: Unique identifier
        - metadata: Additional metadata
        """
        # Combine title and content for full text
        title = article.get('title', '')
        content = article.get('content', article.get('text', ''))
        
        if title and content:
            full_text = f"{title}\n\n{content}"
        else:
            full_text = title or content
        
        # Generate ID if not present
        article_id = article.get('id', f"{article.get('year', 'unknown')}_{hash(title)}")
        
        return {
            'text': full_text,
            'title': title,
            'year': article.get('year'),
            'article_id': str(article_id),
            'metadata': {
                k: v for k, v in article.items() 
                if k not in ['content', 'text', 'title', 'year', 'id']
            }
        }
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.articles_df is None:
            self.load_dataset()
        
        year_counts = self.articles_df['year'].value_counts().sort_index()
        
        return {
            'total_articles': len(self.articles_df),
            'year_range': f"{self.articles_df['year'].min()}-{self.articles_df['year'].max()}",
            'articles_by_decade': {
                f"{decade}s": len(self.articles_df[
                    (self.articles_df['year'] >= decade) & 
                    (self.articles_df['year'] < decade + 10)
                ])
                for decade in range(1920, 2030, 10)
            },
            'missing_years': len(self.articles_df[self.articles_df['year'].isna()]),
            'columns': list(self.articles_df.columns)
        }


class ArticleChunker:
    """
    Initial chunking of articles for processing.
    
    This is a simple chunker - the smart paragraph-aware chunker
    will be implemented separately.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_article(self, article: Dict) -> List[Dict]:
        """
        Chunk article into smaller pieces.
        
        Returns list of chunks with metadata.
        """
        text = article['text']
        chunks = []
        
        # Simple character-based chunking for now
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'article_id': article['article_id'],
                    'chunk_index': chunk_idx,
                    'year': article['year'],
                    'title': article['title'],
                    'metadata': {
                        **article.get('metadata', {}),
                        'chunk_start': start,
                        'chunk_end': end,
                        'total_chunks': None  # Will be set after all chunks created
                    }
                })
                chunk_idx += 1
            
            # Move start position
            start = end - self.chunk_overlap
        
        # Update total chunks count
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
"""Utility for loading Foreign Affairs article data"""

import json
import glob
import os
from typing import List, Dict, Any


def load_years_data(years: List[int], data_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load Foreign Affairs articles for specified years.
    
    Args:
        years: List of years to load (e.g., [1922, 1923])
        data_dir: Directory containing article JSON files
        
    Returns:
        List of article dictionaries with text, title, year, article_id
    """
    if data_dir is None:
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    
    articles = []
    
    for year in years:
        pattern = os.path.join(data_dir, f"{year}_*.json")
        
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    
                    # Extract article ID from filename
                    filename = os.path.basename(filepath)
                    article_id = filename.replace('.json', '')
                    
                    # Transform to expected format
                    article = {
                        'text': article_data.get('content', ''),
                        'title': article_data.get('title', ''),
                        'year': year,
                        'article_id': article_id,
                        'metadata': {
                            'publication': article_data.get('publication', 'Foreign Affairs'),
                            'url': article_data.get('url', ''),
                            'index': article_data.get('index', 0)
                        }
                    }
                    
                    # Only add if we have content
                    if article['text']:
                        articles.append(article)
                        
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    return articles


def load_single_article(article_id: str, data_dir: str = None) -> Dict[str, Any]:
    """
    Load a single article by ID.
    
    Args:
        article_id: Article ID (e.g., "1922_1")
        data_dir: Directory containing article JSON files
        
    Returns:
        Article dictionary or None if not found
    """
    if data_dir is None:
        data_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    
    filepath = os.path.join(data_dir, f"{article_id}.json")
    
    if not os.path.exists(filepath):
        return None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
            
        # Extract year from article_id
        year = int(article_id.split('_')[0])
        
        return {
            'text': article_data.get('content', ''),
            'title': article_data.get('title', ''),
            'year': year,
            'article_id': article_id,
            'metadata': {
                'publication': article_data.get('publication', 'Foreign Affairs'),
                'url': article_data.get('url', ''),
                'index': article_data.get('index', 0)
            }
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
#!/usr/bin/env python3
"""
Create a HuggingFace dataset from the scraped Foreign Affairs articles.

This script:
1. Loads all JSON articles from the data directory
2. Converts them to HuggingFace Dataset format
3. Adds metadata and validates the data
4. Saves the dataset locally for upload
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

try:
    from datasets import Dataset, DatasetDict, Features, Value
except ImportError:
    print("Error: Please install the datasets library")
    print("Run: pip install datasets")
    sys.exit(1)


class ForeignAffairsDatasetCreator:
    """Create HuggingFace dataset from Foreign Affairs JSON files"""
    
    def __init__(self, articles_dir: str):
        self.articles_dir = Path(articles_dir)
        if not self.articles_dir.exists():
            raise ValueError(f"Articles directory not found: {articles_dir}")
        
        # Define the dataset features schema
        self.features = Features({
            'article_id': Value('string'),
            'title': Value('string'),
            'content': Value('string'),
            'author': Value('string'),
            'year': Value('int32'),
            'volume': Value('int32'),
            'issue': Value('int32'),
            'url': Value('string'),
            'scraped_at': Value('string'),
            'content_length': Value('int32'),
            'download_date': Value('string'),
            'filename': Value('string')
        })
    
    def load_articles(self) -> List[Dict[str, Any]]:
        """Load all JSON articles from the directory"""
        articles = []
        json_files = sorted(glob.glob(str(self.articles_dir / "*.json")))
        
        print(f"Found {len(json_files)} JSON files")
        
        for filepath in tqdm(json_files, desc="Loading articles"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                # Add filename-based article_id
                filename = os.path.basename(filepath)
                article_id = filename.replace('.json', '')
                
                # Ensure all required fields exist with defaults
                processed_article = {
                    'article_id': article_id,
                    'title': article.get('title', ''),
                    'content': article.get('content', ''),
                    'author': article.get('author', 'Unknown'),
                    'year': article.get('year', 0),
                    'volume': article.get('volume', 0),
                    'issue': article.get('issue', 0),
                    'url': article.get('url', ''),
                    'scraped_at': article.get('scraped_at', ''),
                    'content_length': article.get('content_length', len(article.get('content', ''))),
                    'download_date': article.get('download_date', ''),
                    'filename': filename
                }
                
                # Skip articles with no content
                if processed_article['content']:
                    articles.append(processed_article)
                else:
                    print(f"Warning: Skipping {filename} - no content")
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        return articles
    
    def create_dataset(self, articles: List[Dict[str, Any]]) -> Dataset:
        """Create HuggingFace Dataset from articles"""
        print(f"\nCreating dataset from {len(articles)} articles...")
        
        # Create the dataset
        dataset = Dataset.from_list(articles, features=self.features)
        
        # Sort by year and article_id for consistency
        dataset = dataset.sort(['year', 'article_id'])
        
        return dataset
    
    def analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze the dataset and return statistics"""
        stats = {
            'total_articles': len(dataset),
            'year_range': {
                'min': min(dataset['year']),
                'max': max(dataset['year'])
            },
            'total_content_length': sum(dataset['content_length']),
            'avg_content_length': sum(dataset['content_length']) / len(dataset),
            'articles_by_year': {},
            'authors': {
                'total': len(set(dataset['author'])),
                'unknown_count': sum(1 for author in dataset['author'] if author == 'Unknown')
            }
        }
        
        # Count articles by year
        for year in dataset['year']:
            stats['articles_by_year'][year] = stats['articles_by_year'].get(year, 0) + 1
        
        return stats
    
    def save_dataset(self, dataset: Dataset, output_dir: str = "data/huggingface_dataset"):
        """Save the dataset locally"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving dataset to {output_path}...")
        dataset.save_to_disk(str(output_path))
        
        # Also save as Parquet for efficient loading
        parquet_path = output_path / "foreign_affairs.parquet"
        dataset.to_parquet(str(parquet_path))
        
        print(f"✅ Dataset saved to {output_path}")
        print(f"✅ Parquet file saved to {parquet_path}")
        
        return output_path
    
    def create_dataset_card(self, stats: Dict[str, Any]) -> str:
        """Create a dataset card with metadata"""
        card = f"""---
language:
- en
license: other
task_categories:
- text-generation
- question-answering
pretty_name: Foreign Affairs Essays Collection (1922-2024)
size_categories:
- 1K<n<10K
tags:
- international-relations
- politics
- history
- foreign-policy
dataset_info:
  features:
  - name: article_id
    dtype: string
  - name: title
    dtype: string
  - name: content
    dtype: string
  - name: author
    dtype: string
  - name: year
    dtype: int32
  - name: volume
    dtype: int32
  - name: issue
    dtype: int32
  - name: url
    dtype: string
  - name: scraped_at
    dtype: string
  - name: content_length
    dtype: int32
  - name: download_date
    dtype: string
  - name: filename
    dtype: string
  splits:
  - name: train
    num_examples: {stats['total_articles']}
---

# Foreign Affairs Essays Collection (1922-2024)

## Dataset Description

This dataset contains {stats['total_articles']:,} essays from Foreign Affairs magazine, spanning from {stats['year_range']['min']} to {stats['year_range']['max']}. Foreign Affairs is a leading journal of international relations and U.S. foreign policy.

### Dataset Summary

- **Total Articles**: {stats['total_articles']:,}
- **Year Range**: {stats['year_range']['min']}-{stats['year_range']['max']}
- **Average Article Length**: {stats['avg_content_length']:,.0f} characters
- **Total Content**: {stats['total_content_length'] / 1_000_000:.1f}M characters
- **Unique Authors**: {stats['authors']['total']:,}

### Data Fields

- `article_id`: Unique identifier derived from the filename
- `title`: Article title
- `content`: Full text content of the article
- `author`: Author name (or "Unknown" if not available)
- `year`: Publication year
- `volume`: Volume number
- `issue`: Issue number within the volume
- `url`: Original URL on foreignaffairs.com
- `scraped_at`: Timestamp when the article was scraped
- `content_length`: Length of the content in characters
- `download_date`: Date when the article was downloaded
- `filename`: Original filename

### Usage

```python
from datasets import load_dataset

# Load from HuggingFace (private repository)
dataset = load_dataset("YOUR_USERNAME/foreign_affairs_essays_1922_2024", use_auth_token=True)

# Or load from local disk
dataset = load_from_disk("path/to/dataset")

# Example: Get all articles from 1962
articles_1962 = dataset.filter(lambda x: x['year'] == 1962)

# Example: Search for articles about Cold War
cold_war_articles = dataset.filter(
    lambda x: 'cold war' in x['content'].lower() or 'cold war' in x['title'].lower()
)
```

### Source Data

The articles were scraped from Foreign Affairs website. Note that access to full articles requires a subscription.

### Licensing Information

This dataset is for research purposes only. The content is copyrighted by Foreign Affairs and the Council on Foreign Relations. Users must comply with Foreign Affairs' terms of service.

### Citation

If you use this dataset, please cite:

```
@misc{{foreign_affairs_dataset,
  title={{Foreign Affairs Essays Collection (1922-2024)}},
  year={{2025}},
  note={{Private dataset for research purposes}}
}}
```

### Acknowledgments

Foreign Affairs has been published by the Council on Foreign Relations since 1922.
"""
        return card


def main():
    """Create the HuggingFace dataset"""
    # Define paths
    articles_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays/articles"
    output_dir = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/huggingface_dataset"
    
    print("🚀 Foreign Affairs HuggingFace Dataset Creator")
    print("=" * 60)
    
    # Create dataset
    creator = ForeignAffairsDatasetCreator(articles_dir)
    
    # Load articles
    print("\n📚 Loading articles...")
    articles = creator.load_articles()
    
    if not articles:
        print("❌ No articles loaded!")
        return
    
    # Create dataset
    dataset = creator.create_dataset(articles)
    
    # Analyze dataset
    print("\n📊 Analyzing dataset...")
    stats = creator.analyze_dataset(dataset)
    
    # Print statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"  • Total articles: {stats['total_articles']:,}")
    print(f"  • Year range: {stats['year_range']['min']}-{stats['year_range']['max']}")
    print(f"  • Average length: {stats['avg_content_length']:,.0f} characters")
    print(f"  • Unique authors: {stats['authors']['total']:,}")
    print(f"  • Unknown authors: {stats['authors']['unknown_count']:,}")
    
    # Sample year distribution
    print(f"\n📅 Articles by decade:")
    decade_counts = {}
    for year, count in stats['articles_by_year'].items():
        decade = (year // 10) * 10
        decade_counts[decade] = decade_counts.get(decade, 0) + count
    
    for decade in sorted(decade_counts.keys()):
        print(f"  • {decade}s: {decade_counts[decade]:,} articles")
    
    # Save dataset
    output_path = creator.save_dataset(dataset, output_dir)
    
    # Create and save dataset card
    print("\n📝 Creating dataset card...")
    dataset_card = creator.create_dataset_card(stats)
    card_path = output_path / "README.md"
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(dataset_card)
    print(f"✅ Dataset card saved to {card_path}")
    
    # Save statistics
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics saved to {stats_path}")
    
    print("\n✨ Dataset creation complete!")
    print(f"\nNext steps:")
    print(f"1. Review the dataset at: {output_path}")
    print(f"2. Run the upload script to push to HuggingFace")


if __name__ == "__main__":
    main()
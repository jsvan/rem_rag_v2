"""
Inspect the Foreign Affairs dataset structure
"""

import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.fa_loader import ForeignAffairsLoader


def inspect_dataset():
    """Load and inspect the dataset structure"""
    print("🔍 Inspecting Foreign Affairs Dataset")
    print("=" * 50)
    
    # Load dataset
    loader = ForeignAffairsLoader()
    print("\n📚 Loading dataset...")
    all_data = loader.load_dataset()
    
    # Convert to list if DataFrame
    if hasattr(all_data, 'to_dict'):
        print("  Converting DataFrame to list of dicts...")
        all_data = all_data.to_dict('records')
    
    print(f"\n📊 Dataset info:")
    print(f"  Total articles: {len(all_data)}")
    
    # Print first article's full structure
    if all_data:
        print(f"\n📄 First article structure:")
        first = all_data[0]
        print(json.dumps(first, indent=2, default=str)[:1000] + "...")
        
        # Check a few more to see year patterns
        print(f"\n🗓️ Year extraction patterns:")
        for i in range(min(10, len(all_data))):
            article = all_data[i]
            year = article.get('year')
            url = article.get('url', '')
            title = article.get('title', 'No title')
            
            # Try to extract year from URL
            import re
            url_year = None
            if url:
                match = re.search(r'/(\d{4})-\d{2}-\d{2}/', url)
                if match:
                    url_year = match.group(1)
            
            print(f"  Article {i+1}:")
            print(f"    Title: {title[:60]}...")
            print(f"    Year field: {year}")
            print(f"    URL: {url}")
            print(f"    Year from URL: {url_year}")
            print()
    
    # Find all 1962 articles
    print("\n🎯 Finding 1962 articles:")
    count_1962 = 0
    for article in all_data:
        year = article.get('year')
        if year is None and 'url' in article:
            import re
            match = re.search(r'/(\d{4})-\d{2}-\d{2}/', article['url'])
            if match:
                year = int(match.group(1))
        
        if year == 1962:
            count_1962 += 1
            if count_1962 <= 3:  # Show first 3
                print(f"  {count_1962}. {article.get('title', 'No title')[:60]}...")
                print(f"     URL: {article.get('url', 'No URL')}")
    
    print(f"\n✅ Total 1962 articles found: {count_1962}")


if __name__ == "__main__":
    inspect_dataset()
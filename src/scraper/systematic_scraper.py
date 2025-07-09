"""Systematic Foreign Affairs scraper using year/volume/issue iteration with cookie auth."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystematicFAScraper:
    BASE_URL = "https://www.foreignaffairs.com"
    
    def __init__(self, data_dir: Path, cookie_file: Path, rate_limit: float = 0.5):
        self.data_dir = Path(data_dir)
        self.cookie_file = Path(cookie_file)
        self.rate_limit = rate_limit
        
        # Create directories
        self.issues_dir = self.data_dir / "issues"
        self.articles_dir = self.data_dir / "articles"
        self.issues_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.data_dir / "scrape_progress.json"
        self.progress = self.load_progress()
        
        # Setup session with cookies
        self.session = self.setup_session()
    
    def load_progress(self) -> Dict:
        """Load progress from file or initialize."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'last_year': 1922,
            'last_volume': 1,
            'last_issue': 0,
            'total_issues': 0,
            'total_articles': 0,
            'failed_issues': [],
            'failed_articles': []
        }
    
    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def setup_session(self) -> requests.Session:
        """Setup requests session with cookies."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Load cookies if file exists
        if self.cookie_file.exists():
            logger.info(f"Loading cookies from {self.cookie_file}")
            with open(self.cookie_file, 'r') as f:
                cookies = json.load(f)
                
            # Convert cookies to requests format
            for cookie in cookies:
                session.cookies.set(
                    cookie['name'],
                    cookie['value'],
                    domain=cookie.get('domain', '.foreignaffairs.com'),
                    path=cookie.get('path', '/')
                )
            logger.info(f"Loaded {len(cookies)} cookies")
        else:
            logger.warning(f"Cookie file not found: {self.cookie_file}")
        
        return session
    
    def get_issue_url(self, year: int, volume: int, issue: int) -> str:
        """Generate issue URL."""
        return f"{self.BASE_URL}/issues/{year}/{volume}/{issue}"
    
    def scrape_issue_page(self, year: int, volume: int, issue: int) -> Optional[List[Dict]]:
        """Scrape an issue page for article metadata."""
        url = self.get_issue_url(year, volume, issue)
        
        try:
            response = self.session.get(url, timeout=30)
            
            # Check if we got an HTML 404 page (FA returns 200 with "Page Not Found" content)
            if "Page Not Found" in response.text or "We can't find the page you've requested" in response.text:
                logger.debug(f"Issue not found (HTML 404): {url}")
                return None
            
            # Check actual HTTP status
            if response.status_code == 404:
                logger.debug(f"Issue not found (HTTP 404): {url}")
                return None
            
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = []
            
            # Find all article divs
            article_divs = soup.find_all('div', class_='mb-20 col-9 col-md-8 ml-0')
            
            for div in article_divs:
                try:
                    # Extract title and URL
                    title_elem = div.find('h3', class_='body-m')
                    if not title_elem:
                        continue
                    
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                    
                    title = link_elem.text.strip()
                    article_url = link_elem.get('href', '')
                    if not article_url.startswith('http'):
                        article_url = self.BASE_URL + article_url
                    
                    # Extract author
                    author_elem = div.find('p', class_='body-s')
                    author = None
                    if author_elem:
                        # Check if author is in a link
                        author_link = author_elem.find('a')
                        if author_link:
                            author = author_link.text.strip()
                        else:
                            author = author_elem.text.strip()
                    
                    articles.append({
                        'title': title,
                        'url': article_url,
                        'author': author,
                        'year': year,
                        'volume': volume,
                        'issue': issue,
                        'scraped_at': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error parsing article in issue {year}/{volume}/{issue}: {e}")
            
            logger.info(f"Found {len(articles)} articles in {year} Vol.{volume} Issue {issue}")
            return articles
            
        except requests.exceptions.RequestException as e:
            if '404' not in str(e):
                logger.error(f"Error fetching {url}: {e}")
            return None
    
    def scrape_article_content(self, article: Dict) -> Optional[Dict]:
        """Scrape full article content."""
        url = article['url']
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple selectors for content
            content_selectors = [
                ('div', {'class': 'body-content'}),
                ('div', {'class': 'article-body'}),
                ('div', {'class': 'article-content'}),
                ('div', {'class': 'paywall'}),
                ('section', {'class': 'article-content'}),
                ('div', {'class': 'article-text'})
            ]
            
            article_body = None
            for tag, attrs in content_selectors:
                article_body = soup.find(tag, attrs)
                if article_body:
                    break
            
            if article_body:
                # Remove scripts and styles
                for elem in article_body(['script', 'style']):
                    elem.decompose()
                
                text = article_body.get_text(separator='\n', strip=True)
                
                # Check for paywall
                if len(text) < 500 or "Subscribe to Foreign Affairs" in text:
                    logger.warning(f"Possible paywall content for: {article['title']}")
                
                return {
                    **article,
                    'content': text,
                    'content_length': len(text),
                    'download_date': datetime.now().isoformat()
                }
            else:
                # Fallback to paragraphs
                paragraphs = soup.find_all('p')
                text_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        text_parts.append(text)
                
                if text_parts:
                    content = '\n\n'.join(text_parts)
                    return {
                        **article,
                        'content': content,
                        'content_length': len(content),
                        'download_date': datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
            return None
    
    def iterate_issues(self, start_year: int = 1922, end_year: int = 2025):
        """Iterate through all issues systematically."""
        year = max(start_year, self.progress['last_year'])
        volume = self.progress['last_volume']
        issue = self.progress['last_issue'] + 1
        
        logger.info(f"Starting from {year} Vol.{volume} Issue {issue}")
        
        consecutive_issue_failures = 0
        consecutive_volume_failures = 0
        max_issues_per_volume = 12  # Try up to 12 issues per volume
        
        while year <= end_year:
            # Try current issue with retry
            articles = None
            retry_count = 0
            max_retries = 2
            
            while articles is None and retry_count <= max_retries:
                if retry_count > 0:
                    logger.info(f"Retrying {year} Vol.{volume} Issue {issue} (attempt {retry_count + 1})...")
                    time.sleep(3)  # Wait 3 seconds before retry
                
                articles = self.scrape_issue_page(year, volume, issue)
                retry_count += 1
            
            if articles is not None:
                # Success - save issue data
                issue_filename = f"{year}_vol{volume}_issue{issue}.json"
                issue_path = self.issues_dir / issue_filename
                
                with open(issue_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'year': year,
                        'volume': volume,
                        'issue': issue,
                        'article_count': len(articles),
                        'articles': articles
                    }, f, indent=2, ensure_ascii=False)
                
                self.progress['total_issues'] += 1
                self.progress['total_articles'] += len(articles)
                
                # Download articles
                logger.info(f"Downloading {len(articles)} articles from {year} Vol.{volume} Issue {issue}...")
                for idx, article in enumerate(articles, 1):
                    time.sleep(self.rate_limit)  # Rate limiting
                    
                    # Create article filename with year, volume, and issue
                    safe_title = "".join(c for c in article['title'] 
                                       if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title.replace(' ', '_')[:100]
                    article_filename = f"{year}_v{volume:03d}_i{issue:02d}_{safe_title}.json"
                    article_path = self.articles_dir / article_filename
                    
                    # Skip if already downloaded
                    if article_path.exists():
                        logger.debug(f"  [{idx}/{len(articles)}] Skipping existing: {article['title'][:50]}...")
                        continue
                    
                    logger.info(f"  [{idx}/{len(articles)}] Downloading: {article['title'][:60]}...")
                    
                    # Download content
                    full_article = self.scrape_article_content(article)
                    if full_article:
                        with open(article_path, 'w', encoding='utf-8') as f:
                            json.dump(full_article, f, indent=2, ensure_ascii=False)
                    else:
                        self.progress['failed_articles'].append(article['url'])
                
                # Update progress
                self.progress['last_year'] = year
                self.progress['last_volume'] = volume
                self.progress['last_issue'] = issue
                
                logger.info(f"Completed {year} Vol.{volume} Issue {issue}")
                
                # Save progress periodically
                if self.progress['total_issues'] % 5 == 0:
                    self.save_progress()
                    logger.info(f"Overall progress: {self.progress['total_issues']} issues, "
                              f"{self.progress['total_articles']} articles")
                
                # Move to next issue
                issue += 1
                consecutive_issue_failures = 0
                
            else:
                # Failed to get issue after retries
                consecutive_issue_failures += 1
                logger.info(f"Issue {year} Vol.{volume} Issue {issue} not found")
                
                # Try more issues before giving up on this volume
                if issue < max_issues_per_volume:
                    issue += 1
                else:
                    # We've tried enough issues in this volume, move to next volume
                    consecutive_volume_failures += 1
                    volume += 1
                    issue = 1
                    consecutive_issue_failures = 0
                    logger.info(f"Moving to {year} Vol.{volume} after trying {max_issues_per_volume} issues")
                    
                    # If we've failed multiple volumes, move to next year
                    if consecutive_volume_failures >= 2:
                        year += 1
                        issue = 1
                        consecutive_volume_failures = 0
                        logger.info(f"Moving to {year} Vol.{volume} after failing 2 consecutive volumes")
                    
            time.sleep(self.rate_limit)  # Rate limiting
        
        # Final save
        self.save_progress()
        logger.info("Scraping complete!")
        logger.info(f"Total issues: {self.progress['total_issues']}")
        logger.info(f"Total articles: {self.progress['total_articles']}")


def main():
    """Main entry point."""
    data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
    cookie_file = data_dir / ".fa_cookie.json"
    
    scraper = SystematicFAScraper(data_dir, cookie_file, rate_limit=1.1)
    scraper.iterate_issues()


if __name__ == "__main__":
    main()
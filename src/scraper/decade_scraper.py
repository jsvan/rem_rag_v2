"""Decade-based Foreign Affairs scraper that iterates through decade archive pages."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecadeFAScraper:
    BASE_URL = "https://www.foreignaffairs.com"
    
    def __init__(self, data_dir: Path, cookie_file: Path, rate_limit: float = 1.0):
        self.data_dir = Path(data_dir)
        self.cookie_file = Path(cookie_file)
        self.rate_limit = rate_limit
        
        # Create directories
        self.issues_dir = self.data_dir / "issues"
        self.articles_dir = self.data_dir / "articles"
        self.issues_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.data_dir / "decade_scrape_progress.json"
        self.progress = self.load_progress()
        
        # Track what we've already scraped
        self.scraped_issues = self.get_scraped_issues()
        
        # Setup session with cookies
        self.session = self.setup_session()
    
    def load_progress(self) -> Dict:
        """Load progress from file or initialize."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'last_decade': 1920,
            'scraped_decades': [],
            'total_issues': 0,
            'total_articles': 0,
            'failed_issues': [],
            'failed_articles': []
        }
    
    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_scraped_issues(self) -> Set[str]:
        """Get set of already scraped issues."""
        scraped = set()
        for issue_file in self.issues_dir.glob("*.json"):
            # Extract year, volume, issue from filename
            parts = issue_file.stem.split('_')
            if len(parts) >= 3:
                year = parts[0]
                vol = parts[1].replace('vol', '')
                issue = parts[2].replace('issue', '')
                scraped.add(f"{year}/{vol}/{issue}")
        return scraped
    
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
    
    def scrape_decade_archive(self, decade: int) -> List[Dict]:
        """Scrape a decade archive page to get all issue URLs."""
        url = f"{self.BASE_URL}/issues/archive/{decade}"
        logger.info(f"Scraping decade archive: {url}")
        
        try:
            # Add more browser-like headers for this request
            headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            }
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            issues = []
            
            # Look for the Drupal views container
            view_container = soup.find('div', class_='view-issue-listing')
            if not view_container:
                logger.warning(f"Could not find issue listing view for {decade}s")
                return []
            
            # Find all links with issue pattern within the view
            import re
            issue_pattern = re.compile(rf'/issues/({decade//10}\d)/(\d+)/(\d+)')
            issue_links = view_container.find_all('a', href=issue_pattern)
            
            logger.info(f"Found {len(issue_links)} issue links in {decade}s archive")
            
            # Process each link and extract metadata from surrounding elements
            for link in issue_links:
                href = link.get('href', '')
                match = issue_pattern.search(href)
                if match:
                    year = int(match.group(1))
                    volume = int(match.group(2))
                    issue = int(match.group(3))
                    
                    issue_info = {
                        'year': year,
                        'volume': volume,
                        'issue': issue,
                        'url': urljoin(self.BASE_URL, href),
                        'decade': decade
                    }
                    
                    # Get the link text as title
                    link_text = link.get_text(strip=True)
                    if link_text:
                        issue_info['title'] = link_text
                    
                    # Try to find the row containing this link for additional metadata
                    row = link.find_parent('div', class_='views-row')
                    if row:
                        # Look for date/season info in the row
                        date_field = row.find('div', class_='views-field-field-issue-date')
                        if date_field:
                            date_text = date_field.get_text(strip=True)
                            if date_text:
                                issue_info['date'] = date_text
                    
                    issues.append(issue_info)
            
            # Sort by year, volume, issue (descending)
            issues.sort(key=lambda x: (x['year'], x['volume'], x['issue']), reverse=True)
            
            logger.info(f"Processed {len(issues)} issues in {decade}s archive")
            return issues
            
        except Exception as e:
            logger.error(f"Error scraping decade {decade}: {e}")
            return []
    
    def scrape_issue_page(self, issue_info: Dict) -> Optional[List[Dict]]:
        """Scrape an issue page for article metadata."""
        url = issue_info['url']
        year = issue_info['year']
        volume = issue_info['volume']
        issue = issue_info['issue']
        
        # Check if already scraped
        issue_key = f"{year}/{volume}/{issue}"
        if issue_key in self.scraped_issues:
            logger.info(f"Skipping already scraped issue: {issue_key}")
            return None
        
        try:
            response = self.session.get(url, timeout=30)
            
            # Check if we got an HTML 404 page
            if "Page Not Found" in response.text or "We can't find the page you've requested" in response.text:
                logger.debug(f"Issue not found (HTML 404): {url}")
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
            logger.error(f"Error fetching {url}: {e}")
            self.progress['failed_issues'].append(url)
            return None
    
    def scrape_article_content(self, article: Dict) -> Optional[Dict]:
        """Scrape full article content (reused from original scraper)."""
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
            self.progress['failed_articles'].append(url)
            return None
    
    def save_issue_data(self, issue_info: Dict, articles: List[Dict]):
        """Save issue metadata and download articles."""
        year = issue_info['year']
        volume = issue_info['volume'] 
        issue = issue_info['issue']
        
        # Save issue data
        issue_filename = f"{year}_vol{volume}_issue{issue}.json"
        issue_path = self.issues_dir / issue_filename
        
        with open(issue_path, 'w', encoding='utf-8') as f:
            json.dump({
                'year': year,
                'volume': volume,
                'issue': issue,
                'article_count': len(articles),
                'articles': articles,
                'issue_date': issue_info.get('date', ''),
                'issue_title': issue_info.get('title', '')
            }, f, indent=2, ensure_ascii=False)
        
        self.progress['total_issues'] += 1
        self.progress['total_articles'] += len(articles)
        
        # Download articles
        logger.info(f"Downloading {len(articles)} articles from {year} Vol.{volume} Issue {issue}...")
        for idx, article in enumerate(articles, 1):
            time.sleep(self.rate_limit)  # Rate limiting
            
            # Create article filename
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
    
    def scrape_decades(self, start_decade: int = 1920, end_decade: int = 2020):
        """Main method to scrape all decades."""
        for decade in range(start_decade, end_decade + 10, 10):
            if decade in self.progress['scraped_decades']:
                logger.info(f"Skipping already scraped decade: {decade}s")
                continue
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting decade: {decade}s")
            logger.info(f"{'='*50}")
            
            # Get all issues for this decade
            issues = self.scrape_decade_archive(decade)
            
            if not issues:
                logger.warning(f"No issues found for decade {decade}s")
                continue
            
            # Process each issue
            for issue_info in issues:
                time.sleep(self.rate_limit)  # Rate limiting
                
                # Scrape issue page
                articles = self.scrape_issue_page(issue_info)
                
                if articles:
                    self.save_issue_data(issue_info, articles)
                    
                    # Save progress periodically
                    if self.progress['total_issues'] % 5 == 0:
                        self.save_progress()
                        logger.info(f"Progress saved: {self.progress['total_issues']} issues, "
                                  f"{self.progress['total_articles']} articles")
            
            # Mark decade as complete
            self.progress['scraped_decades'].append(decade)
            self.progress['last_decade'] = decade
            self.save_progress()
            
            logger.info(f"Completed decade {decade}s")
            logger.info(f"Total so far: {self.progress['total_issues']} issues, "
                       f"{self.progress['total_articles']} articles")
    
    def test_decade(self, decade: int = 1980):
        """Test scraping a single decade."""
        logger.info(f"Testing decade scraping for {decade}s")
        
        # Get issues for the decade
        issues = self.scrape_decade_archive(decade)
        
        if issues:
            logger.info(f"\nFound {len(issues)} issues:")
            for issue in issues[:5]:  # Show first 5
                logger.info(f"  - {issue['year']} Vol.{issue['volume']} Issue {issue['issue']}: "
                           f"{issue.get('date', 'Unknown date')}")
            
            # Try scraping the first issue
            if issues:
                logger.info(f"\nTesting first issue: {issues[0]['url']}")
                articles = self.scrape_issue_page(issues[0])
                if articles:
                    logger.info(f"Found {len(articles)} articles:")
                    for article in articles[:3]:  # Show first 3
                        logger.info(f"  - {article['title'][:60]}... by {article.get('author', 'Unknown')}")


def main():
    """Main entry point."""
    data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
    cookie_file = data_dir / ".fa_cookie.json"
    
    scraper = DecadeFAScraper(data_dir, cookie_file, rate_limit=1.2)
    
    # Test with 1980s first
    scraper.test_decade(1980)
    
    # Uncomment to run full scraping:
    # scraper.scrape_decades(start_decade=1920, end_decade=2020)


if __name__ == "__main__":
    main()
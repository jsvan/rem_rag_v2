"""Run the decade scraper for all decades from 1920s to 2020s."""
from pathlib import Path
from decade_scraper import DecadeFAScraper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the decade scraper for all decades."""
    data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
    cookie_file = data_dir / ".fa_cookie.json"
    
    # Create scraper with a reasonable rate limit
    scraper = DecadeFAScraper(data_dir, cookie_file, rate_limit=1.5)
    
    # Start from 1920s and go through 2020s
    logger.info("Starting decade-based scraping...")
    logger.info(f"Already scraped issues: {len(scraper.scraped_issues)}")
    
    # Run the full scraping
    scraper.scrape_decades(start_decade=1920, end_decade=2020)
    
    logger.info("\n" + "="*50)
    logger.info("SCRAPING COMPLETE!")
    logger.info(f"Total issues scraped: {scraper.progress['total_issues']}")
    logger.info(f"Total articles scraped: {scraper.progress['total_articles']}")
    logger.info(f"Failed issues: {len(scraper.progress['failed_issues'])}")
    logger.info(f"Failed articles: {len(scraper.progress['failed_articles'])}")


if __name__ == "__main__":
    main()
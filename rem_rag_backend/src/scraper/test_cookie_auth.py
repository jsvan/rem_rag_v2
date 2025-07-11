"""Test cookie authentication before running full scrape."""
import logging
from pathlib import Path
from .systematic_scraper import SystematicFAScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_cookie_authentication():
    """Test that cookie authentication works properly."""
    data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
    cookie_file = data_dir / ".fa_cookie.json"
    
    if not cookie_file.exists():
        logger.error(f"Cookie file not found: {cookie_file}")
        logger.error("Please ensure .fa_cookie.json is in the essays directory")
        return False
    
    logger.info("Initializing scraper with cookies...")
    scraper = SystematicFAScraper(data_dir, cookie_file)
    
    # Test 1: Scrape an issue page
    logger.info("\nTest 1: Scraping issue page (1922 Vol.1 Issue 1)...")
    articles = scraper.scrape_issue_page(1922, 1, 1)
    
    if articles:
        logger.info(f"✓ Successfully scraped issue page with {len(articles)} articles")
        if articles:
            logger.info(f"  First article: {articles[0]['title']} by {articles[0]['author']}")
    else:
        logger.error("✗ Failed to scrape issue page")
        return False
    
    # Test 2: Scrape article content
    if articles:
        logger.info("\nTest 2: Downloading article content...")
        test_article = articles[0]
        content = scraper.scrape_article_content(test_article)
        
        if content and content.get('content'):
            text_length = content['content_length']
            preview = content['content'][:500].replace('\n', ' ')
            
            logger.info(f"✓ Successfully downloaded article ({text_length} characters)")
            logger.info(f"  Preview: {preview}...")
            
            # Check for paywall
            paywall_indicators = [
                "You are reading a free article",
                "Subscribe to Foreign Affairs",
                "Paywall-free reading",
                "Already a subscriber"
            ]
            
            is_paywall = any(indicator in content['content'] for indicator in paywall_indicators)
            
            if is_paywall:
                logger.warning("⚠️  PAYWALL CONTENT DETECTED!")
                logger.warning("Cookie authentication may not be working properly")
                return False
            else:
                logger.info("✓ Full article content retrieved (no paywall)")
                return True
        else:
            logger.error("✗ Failed to download article content")
            return False
    
    return False


def test_recent_article():
    """Test downloading a recent article to ensure cookies work for current content."""
    data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
    cookie_file = data_dir / ".fa_cookie.json"
    
    scraper = SystematicFAScraper(data_dir, cookie_file)
    
    # Try a 2025 article
    logger.info("\nTest 3: Testing recent article (2025)...")
    articles = scraper.scrape_issue_page(2025, 104, 4)
    
    if articles:
        test_article = articles[0]
        content = scraper.scrape_article_content(test_article)
        
        if content and content.get('content') and content['content_length'] > 1000:
            logger.info(f"✓ Successfully downloaded recent article: {test_article['title']}")
            logger.info(f"  Content length: {content['content_length']} characters")
            return True
        else:
            logger.error("✗ Failed to get full content for recent article")
            return False
    else:
        logger.error("✗ Failed to get recent issue")
        return False


def main():
    """Run all tests."""
    logger.info("Testing Foreign Affairs cookie authentication...")
    logger.info("="*60)
    
    # Run tests
    old_article_ok = test_cookie_authentication()
    recent_article_ok = test_recent_article()
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Old articles (1922): {'✓ PASS' if old_article_ok else '✗ FAIL'}")
    logger.info(f"Recent articles (2025): {'✓ PASS' if recent_article_ok else '✗ FAIL'}")
    
    if old_article_ok and recent_article_ok:
        logger.info("\n✓ All tests passed! Ready to run full scrape.")
        logger.info("Run: python -m src.scraper.systematic_scraper")
    else:
        logger.error("\n✗ Tests failed. Please check:")
        logger.error("1. Cookie file exists at data/essays/.fa_cookie.json")
        logger.error("2. Cookies are valid and not expired")
        logger.error("3. You exported cookies while logged into Foreign Affairs")


if __name__ == "__main__":
    main()
"""Scrape only the missing decades to fill gaps."""
from pathlib import Path
from decade_scraper import DecadeFAScraper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the decade scraper for missing decades."""
    data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
    cookie_file = data_dir / ".fa_cookie.json"
    
    # Create scraper with a reasonable rate limit
    scraper = DecadeFAScraper(data_dir, cookie_file, rate_limit=1.5)
    
    logger.info("Starting targeted decade-based scraping...")
    logger.info(f"Already scraped issues: {len(scraper.scraped_issues)}")
    
    # Target specific decades with gaps:
    # - 1920s-1970s: Need more issues (currently only have 1 per year)
    # - 1990s-2020s: Completely missing
    
    decades_to_scrape = [
        1920, 1930, 1940, 1950, 1960, 1970,  # Fill in missing issues
        1990, 2000, 2010, 2020  # Completely missing decades
    ]
    
    for decade in decades_to_scrape:
        if decade not in scraper.progress['scraped_decades']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing decade: {decade}s")
            logger.info(f"{'='*50}")
            
            # Get all issues for this decade
            issues = scraper.scrape_decade_archive(decade)
            
            if not issues:
                logger.warning(f"No issues found for decade {decade}s")
                continue
            
            logger.info(f"Found {len(issues)} total issues for {decade}s")
            
            # Filter out already scraped issues
            new_issues = []
            for issue_info in issues:
                issue_key = f"{issue_info['year']}/{issue_info['volume']}/{issue_info['issue']}"
                if issue_key not in scraper.scraped_issues:
                    new_issues.append(issue_info)
            
            logger.info(f"Need to scrape {len(new_issues)} new issues from {decade}s")
            
            # Process each new issue
            for i, issue_info in enumerate(new_issues):
                logger.info(f"\n[{i+1}/{len(new_issues)}] Processing {issue_info['year']} Vol.{issue_info['volume']} Issue {issue_info['issue']}")
                
                # Rate limiting
                import time
                time.sleep(scraper.rate_limit)
                
                # Scrape issue page
                articles = scraper.scrape_issue_page(issue_info)
                
                if articles:
                    scraper.save_issue_data(issue_info, articles)
                    
                    # Save progress periodically
                    if scraper.progress['total_issues'] % 5 == 0:
                        scraper.save_progress()
                        logger.info(f"Progress saved: {scraper.progress['total_issues']} issues, "
                                  f"{scraper.progress['total_articles']} articles")
            
            # Mark decade as complete
            scraper.progress['scraped_decades'].append(decade)
            scraper.progress['last_decade'] = decade
            scraper.save_progress()
            
            logger.info(f"Completed decade {decade}s")
    
    logger.info("\n" + "="*50)
    logger.info("SCRAPING COMPLETE!")
    logger.info(f"Total issues scraped: {scraper.progress['total_issues']}")
    logger.info(f"Total articles scraped: {scraper.progress['total_articles']}")
    logger.info(f"Failed issues: {len(scraper.progress['failed_issues'])}")
    logger.info(f"Failed articles: {len(scraper.progress['failed_articles'])}")


if __name__ == "__main__":
    main()
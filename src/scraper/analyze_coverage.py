"""Analyze what issues we have vs what's available."""
from pathlib import Path
from collections import defaultdict
import json

data_dir = Path("/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/essays")
issues_dir = data_dir / "issues"

# Count issues by year
issues_by_year = defaultdict(int)
articles_by_year = defaultdict(int)

for issue_file in issues_dir.glob("*.json"):
    try:
        with open(issue_file, 'r') as f:
            data = json.load(f)
            year = data['year']
            issues_by_year[year] += 1
            articles_by_year[year] += data.get('article_count', 0)
    except:
        pass

# Sort and display
print("Current Coverage:")
print("Year | Issues | Articles")
print("-" * 30)

for year in sorted(issues_by_year.keys()):
    print(f"{year:4d} | {issues_by_year[year]:6d} | {articles_by_year[year]:8d}")

print("-" * 30)
print(f"Total: {sum(issues_by_year.values())} issues, {sum(articles_by_year.values())} articles")

# Check for gaps
print("\n\nGaps in coverage:")
all_years = set(range(1922, 2025))
covered_years = set(issues_by_year.keys())
missing_years = sorted(all_years - covered_years)

if missing_years:
    print(f"Missing years ({len(missing_years)}): {missing_years}")
else:
    print("No missing years!")

# Years with only 1 issue (suspicious)
print("\n\nYears with only 1 issue (might be incomplete):")
for year in sorted(issues_by_year.keys()):
    if issues_by_year[year] == 1:
        print(f"  - {year}: {issues_by_year[year]} issue(s)")
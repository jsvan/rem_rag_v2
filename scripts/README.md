# REM RAG Scripts

This directory contains scripts for processing Foreign Affairs articles and running experiments.

## Production Scripts

### `run_2000s_decade.py` ✅
Main script for processing an entire decade with yearly REM cycles.

**Features:**
- Processes articles chronologically by year
- Uses cost-optimized approach (direct storage for themes/REM)
- Runs REM cycles at the end of each year
- Supports batch REM processing (50% cost savings)
- Comprehensive statistics tracking

**Usage:**
```bash
python scripts/run_2000s_decade.py
```

### `run_1922_fixed.py` ✅
Processes 1922 Foreign Affairs articles (the founding year).

**Features:**
- Uses sentence-aware chunking
- Generates article summaries
- Basic theme extraction
- Good for testing on smaller dataset

**Usage:**
```bash
python scripts/run_1922_fixed.py
```

## Utility Scripts

### `analyze_database.py`
Analyzes the vector database contents and generates reports.

**Features:**
- Statistics by node type and year
- Entity frequency analysis
- Sample queries for verification
- Export capabilities

### `clear_database.py`
Clears the ChromaDB collection (use with caution!).

### `test_batch_rem_mock.py`
Tests batch REM processing with mock data.

**Features:**
- Tests batch file creation (no API calls)
- Optional mini batch test (~$0.001)
- Good for debugging batch format

### `test_rem_step.py`
Tests a single REM cycle step.

### `test_year_metadata.py`
Verifies year metadata is correctly added to all node types.

## Data Format

Expected JSON structure for articles:
```json
{
  "title": "Article Title",
  "content": "Full article text...",
  "author": "Author Name",
  "year": 1922,
  "volume": 1,
  "issue": 1,
  "url": "https://..."
}
```

## Architecture Notes

The current implementation uses:
- **Simplified processing** in production scripts for cost efficiency
- **Direct storage** for themes and REM nodes (no implant synthesis)
- **Batch API** for REM processing (50% discount)
- **Year-based filtering** to ensure temporal coherence

For the full READING cycle with entity extraction, see `src/core/reading_cycle.py`.
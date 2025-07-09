# REM RAG Scripts

This directory contains scripts for running experiments and processing Foreign Affairs articles.

## Working Scripts

### `run_1922_fixed.py` ✅
The main working experiment that processes 1922 Foreign Affairs articles.

**Features:**
- Loads articles from local JSON files
- Uses SmartChunker for intelligent paragraph-aware chunking
- Generates direct insight summaries (no meta-commentary)
- Stores in ChromaDB vector database
- Searches for key historical themes

**Usage:**
```bash
python scripts/run_1922_fixed.py
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Local JSON files in `data/essays/articles/`

## Experimental Scripts

### `run_1922_experiment.py` ⚠️
Original experiment design - has interface issues, use `run_1922_fixed.py` instead.

### Other Scripts
- Additional experiment scripts in `src/experiments/`
- These may have interface mismatches with current implementation

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

## Next Steps

1. Add REM cycle integration to the working script
2. Process full 1920s decade (304 articles available)
3. Implement entity extraction and explosion
4. Add evaluation metrics

## Notes

- The 1920s data captures the post-WWI transformation period
- Foreign Affairs was founded in 1922, making it perfect for testing
- Smart chunking preserves semantic coherence better than character-based splitting
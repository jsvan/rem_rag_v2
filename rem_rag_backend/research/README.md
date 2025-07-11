# Vector Space Coverage Analysis

This research module analyzes how well different node types in the REM RAG system cover the embedding space.

## Motivation

The REM RAG system generates several types of synthetic nodes:
- **synthesis**: Connections between new and existing knowledge
- **learning**: Entity-specific insights
- **rem**: Pattern discoveries from 3-node REM cycles

The key question: Are these synthetic nodes filling empty regions in the vector space, or are they clustering near existing content?

## Node Types Analyzed

1. **chunk** - Original text segments from articles
2. **summary** - Article-level summaries  
3. **learning** - Entity-specific learnings
4. **synthesis** - Comparisons with existing knowledge
5. **rem** - Pattern discoveries from REM cycles

## Key Metrics

### 1. Convex Hull Volume
- Measures the total space covered by each node type
- Larger volume indicates broader coverage

### 2. Local Density
- Average density in the neighborhood of each point
- Reveals clustering vs dispersion patterns

### 3. Coverage Radius
- Distance needed to cover X% of the space
- Smaller radius indicates better coverage

### 4. Diversity Index
- Average pairwise distance within node type
- Higher diversity means better spread

### 5. Spatial Entropy
- Measures uniformity of distribution
- Higher entropy indicates more even coverage

## Usage

```bash
# Run full analysis
python research/scripts/run_analysis.py

# Generate report only
python research/scripts/generate_report.py
```

## Output

- Statistical report: `outputs/reports/coverage_analysis_YYYY-MM-DD.json`
- Visualizations: `outputs/plots/`
- Gap analysis: `outputs/reports/coverage_gaps.csv`

## Key Findings

*To be populated after analysis*

## Recommendations

*To be populated after analysis*
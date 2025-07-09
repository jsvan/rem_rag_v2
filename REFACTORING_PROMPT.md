# REM RAG Code Refactoring Task

I need to refactor the REM RAG system to match our improved architectural design where EVERYTHING goes through a universal implant function.

## Current State
The REM RAG system in `/Users/jsv/Projects/foreign_affairs/rem_rag_v2/` currently has misaligned code where:
- The implant function only stores synthesis, not original content
- Reading cycle stores chunks directly instead of through implant
- No article summary generation in core code
- Using k=5 instead of k=3 for neighbor queries

## Target Architecture
As defined in `/Users/jsv/Projects/foreign_affairs/rem_rag_v2/docs/PROCESSING_FLOW.md`:
- ALL content enters through implant_knowledge()
- Implant function ALWAYS stores original content first
- Implant function queries k=3 neighbors and may generate synthesis
- Article summaries are generated and implanted after chunk processing
- Themes are extracted from both chunks and summaries

## Key Files to Modify
1. `src/core/implant.py` - Make it store original content ALWAYS
2. `src/core/reading_cycle.py` - Use implant for all storage, add summary generation
3. `src/core/rem_cycle.py` - Store REM nodes through implant
4. `src/core/entity_processor.py` - Store learnings through implant
5. `src/config.py` - Change NEIGHBORS_COUNT to 3, add summary prompt

## Success Criteria
- Every piece of content (chunks, learnings, summaries, REM nodes) enters via implant
- Implant function is the ONLY place that calls vector_store.add()
- Code matches the flow diagram exactly
- All tests pass with the new architecture

## Implementation Order
1. **Phase 1: Core Implant Rewrite**
   - Modify implant.py to always store original content
   - Update both async and sync versions
   - Ensure k=3 is used
   - Update return structure to include original_id

2. **Phase 2: Reading Cycle Refactor**
   - Remove all direct vector_store.add() calls
   - Use implant for chunk storage
   - Add summary generation after chunk processing
   - Extract themes from summary

3. **Phase 3: Entity/Theme Processing**
   - Update to use implant for learning storage
   - Consider renaming to "theme" for consistency
   - Remove direct storage calls

4. **Phase 4: REM Cycle Update**
   - Store REM nodes through implant
   - Ensure proper metadata structure

5. **Phase 5: Configuration & Testing**
   - Update config.py with new constants
   - Add ARTICLE_SUMMARY_PROMPT
   - Run all tests and fix any breaks

## Code Examples

### Current Implant (WRONG):
```python
async def implant_knowledge(...):
    # Only stores synthesis if valuable
    if synthesis.strip() != "NOTHING":
        synthesis_ids = vector_store.add([synthesis], [metadata])
```

### Target Implant (CORRECT):
```python
async def implant_knowledge(...):
    # ALWAYS store original content first
    original_ids = vector_store.add([new_content], [metadata])
    
    # Then query and possibly synthesize
    # ...
    
    return {
        "original_id": original_ids[0],
        "synthesis_id": synthesis_id if synthesis_stored else None,
        "is_valuable": synthesis_stored
    }
```

Please help me implement these changes systematically, starting with the implant function rewrite.
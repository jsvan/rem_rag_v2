# Chrome Extension: Political-Philosophical Analysis Tool

## Overview
A Chrome extension that acts as a "big brother" analytical companion, reading articles on the user's screen and providing deep political-philosophical critique by:
1. Decomposing articles into discrete axioms (foundational assumptions)
2. Querying each axiom against our REM RAG knowledge base
3. Providing historical context and identifying missing perspectives

## Core Functionality

### Phase 1: Axiom Extraction
**Prompt for article analysis:**
```
Act as a political-philosophical analyst. Given the following article, identify the implicit and explicit argument axioms — the foundational assumptions, values, and beliefs that support the actors' positions or the narrative as a whole.

For each axiom:
1. State it clearly and concisely
2. Generate its logical opposite/negation (for balanced analysis)
3. Indicate whether it's explicit (directly stated) or implicit (assumed)
4. Identify which actor/perspective holds this axiom

Example:
- Axiom: "Economic sanctions are an effective tool for changing state behavior"
- Opposite: "Economic sanctions are ineffective at changing state behavior"
```

### Phase 2: RAG Enhancement
For each extracted axiom AND its opposite:
1. Query the REM RAG system for both the axiom and its negation
2. Find evidence supporting both positions from 100 years of Foreign Affairs
3. Generate synthesized statements for both perspectives using our AI
   - NOT raw chunks, but AI-interpreted insights about both views
4. Identify when/where each perspective has proven true historically

### Phase 3: Final Synthesis
**Process:**
1. Collect all axiom-specific AI statements from Phase 2
2. Pass these statements + original article to LLM
3. Generate a unified, balanced response that:
   - Integrates all historical insights
   - Maintains coherent narrative flow
   - Highlights the most important patterns and blind spots

**Output format:**
```
For each axiom, briefly explain:
- How it functions in the logic of the article
- Historical context from our knowledge base
- Common misunderstandings about this assumption
- Missing ideas or perspectives relevant to understanding these issues
```

## Technical Architecture

### 1. Content Extraction
- [ ] Content script to capture article text from current tab
- [ ] Smart detection of article boundaries (main content vs. ads/navigation)
- [ ] Handle various news site formats (NYT, WSJ, Foreign Affairs, etc.)

### 2. Axiom Decomposition Service
- [ ] LLM integration for axiom extraction
- [ ] Automatic generation of axiom opposites/negations
- [ ] Axiom categorization (economic, political, moral, historical)
- [ ] Confidence scoring for implicit vs. explicit axioms

### 3. RAG Query Pipeline
- [ ] Batch axiom queries to REM RAG backend (both axiom and opposite)
- [ ] Retrieve evidence supporting both perspectives
- [ ] Generate AI-synthesized statements for each view (not raw chunks)
- [ ] Identify conditions under which each perspective holds true
- [ ] Aggregate insights across different time periods

### 4. UI Components
- [ ] Sidebar panel that appears alongside articles
- [ ] Axiom cards with expand/collapse for details
- [ ] Historical timeline visualization for each axiom
- [ ] "What's missing" section highlighting blind spots

### 5. Backend API
- [ ] FastAPI server connecting to REM RAG vector store
- [ ] Axiom query endpoint with semantic search
- [ ] AI synthesis endpoint (similar to ask_rag.py logic)
- [ ] Final synthesis endpoint combining all axiom insights
- [ ] Caching layer for common axioms
- [ ] Rate limiting and user authentication

## Implementation Steps

### Phase 1: MVP (Week 1-2)
1. [ ] Basic Chrome extension structure
2. [ ] Content extraction for major news sites
3. [ ] Simple axiom extraction using GPT-4
4. [ ] Mock UI with static responses

### Phase 2: RAG Integration (Week 3-4)
1. [ ] Connect to REM RAG backend
2. [ ] Implement axiom querying pipeline
3. [ ] Add historical context retrieval
4. [ ] Basic caching system

### Phase 3: Enhanced Analysis (Week 5-6)
1. [ ] Improve axiom extraction accuracy
2. [ ] Add missing perspective identification
3. [ ] Implement timeline visualizations
4. [ ] User feedback and annotation system

### Phase 4: Polish & Scale (Week 7-8)
1. [ ] Performance optimization
2. [ ] Support for more content types (PDFs, academic papers)
3. [ ] Export functionality (save analyses)
4. [ ] User preferences and customization

## Example Use Case

**Article:** "China's Economic Slowdown Threatens Global Stability"

**Step A - Extracted Axioms and Opposites:**
1. *Explicit:* Economic growth is necessary for political stability
   - *Opposite:* Economic growth is not necessary for political stability
2. *Implicit:* China's economic model is fundamentally similar to Western capitalism
   - *Opposite:* China's economic model is fundamentally different from Western capitalism
3. *Implicit:* Global economic interdependence makes isolation impossible
   - *Opposite:* States can achieve meaningful isolation despite global interdependence
4. *Explicit:* Authoritarian systems are more fragile during economic downturns
   - *Opposite:* Authoritarian systems are more resilient during economic downturns

**Step B - RAG-Enhanced Statements (per axiom AND opposite):**
1. **Axiom 1 - "Growth necessary for stability":**
   - *Supporting evidence:* "The fall of the Soviet Union, Arab Spring movements, and Latin American debt crises all show regimes collapsing when economic growth stalled."
   - *Opposite evidence:* "Cuba since 1960, North Korea since 1990s, and Iran under sanctions demonstrate decades of regime stability despite economic stagnation. Ideological control matters more than GDP."

2. **Axiom 2 - "China similar to Western capitalism":**
   - *Supporting evidence:* "China's stock markets, private property rights, and consumer culture mirror Western capitalist features. Foreign investment flows suggest functional similarity."
   - *Opposite evidence:* "China's state-owned enterprises control commanding heights, party cells in private companies, and capital controls reveal a fundamentally different system from Western models."

(Similar dual analysis for axioms 3 & 4...)

**Step C - Final Unified Response:**
"This article's analysis rests on several assumptions that historical evidence complicates. While it assumes economic growth ensures political stability, history shows regime survival depends more on ideological control and threat management than GDP figures. The piece also projects Western economic logic onto China's distinct state-capitalist model, missing how Asian developmental states have historically balanced market forces with political control. The interdependence argument, while valid, overlooks how states have selectively decoupled when core interests diverged, as seen in US-Soviet trade during détente. Rather than authoritarian fragility during downturns, the record suggests such systems often use economic crisis to justify tighter control, making them paradoxically more resilient in hard times."

## Key Design Principles

1. **Non-intrusive:** Enhance reading without disrupting flow
2. **Educational:** Teach historical patterns, not just critique
3. **Balanced:** Query both sides of every assumption to avoid confirmation bias
4. **Actionable:** Help users think more deeply, not just consume
5. **Dialectical:** Present thesis and antithesis before synthesis

## Success Metrics

- User engagement time with analysis
- Number of axioms explored per article
- User feedback on insight quality
- Repeat usage rate

## Future Enhancements

1. **Collaborative annotations:** Users can share axiom analyses
2. **Trend detection:** Track how certain axioms evolve over time in media
3. **Personalized learning:** Adapt to user's interests and knowledge level
4. **Academic integration:** Export analyses for research purposes

## Technical Stack

- **Frontend:** Chrome Extension (Manifest V3), React for UI components
- **Backend:** FastAPI, Python 3.11+
- **LLM:** GPT-4 for axiom extraction, GPT-4o-mini for RAG queries
- **Database:** ChromaDB (existing REM RAG store)
- **Caching:** Redis for API responses
- **Analytics:** PostHog for usage tracking

## Challenges to Address

1. **Real-time performance:** Axiom extraction needs to be fast
2. **Context window limits:** Long articles may need chunking
3. **Bias detection:** Ensure axiom extraction isn't biased
4. **Privacy:** Handle user data and reading habits responsibly
5. **Cost management:** Optimize LLM calls to control API costs

## MVP Deliverables

1. [ ] Chrome extension that extracts article text
2. [ ] Axiom extraction for 3-5 core axioms per article
3. [ ] Basic RAG queries for each axiom
4. [ ] Simple UI showing axioms and historical context
5. [ ] Documentation and setup guide
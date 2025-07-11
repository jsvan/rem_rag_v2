# REM RAG Experiments

## 1962 Prototype - Proof of Concept

### Why 1962?
- Cuban Missile Crisis: Peak Cold War tensions
- Rich cast of characters: Kennedy, Khrushchev, Castro
- Before/during/after crisis coverage
- Tests system's ability to build understanding through dramatic events

### Experimental Design

#### Phase 1: Baseline
- Process all 1962 articles chronologically WITHOUT REM cycles
- Evaluate: Can the system predict expert analysis from December 1962?

#### Phase 2: Monthly REM
- Same articles but with REM cycles each month
- Compare: Do REM dreams improve prediction accuracy?

#### Phase 3: Entity Tracking
- Focus queries on key entities (Kennedy, Khrushchev, nuclear deterrence)
- Measure: Does entity coherence improve across the year?

### Success Metrics

1. **Prediction Accuracy**: Can we anticipate year-end retrospectives?
2. **Insight Quality**: Do REM dreams surface non-obvious connections?
3. **Entity Coherence**: Do our models of key actors remain consistent?
4. **Temporal Awareness**: Does the system recognize pattern shifts?

### Key Questions to Answer

- When experts write "in hindsight..." can our system have already seen it?
- Do synthetic nodes get retrieved more often than raw chunks?
- What's the optimal number of REM dreams per cycle?
- How many generations until synthetic nodes degrade?

## Browser Plugin Evaluation

### Test Scenarios

1. **Contemporary Article**: User reads about Taiwan tensions
   - System should surface: Historical precedents, pattern matching to Cold War dynamics
   
2. **Historical Article**: User reads about 1962 crisis
   - System should provide: Contemporary context, what happened next, long-term impacts

3. **Opinion Piece**: User reads hot take on current events  
   - System should highlight: What's missing, historical counter-examples, unstated assumptions

### Success Criteria
- Users report feeling more informed
- Contextualizations are historically accurate
- Insights feel profound, not obvious
- System doesn't just retrieve, it synthesizes

## Ablation Studies

Test impact of each component:

1. **No Entity Explosion**: Just chunk and embed
2. **No REM Cycles**: Only reading phase
3. **No Synthesis**: Store raw chunks only
4. **Different k**: Test k=3, 5, 10 for neighbors
5. **REM Frequency**: Weekly vs Monthly vs Yearly

## Long-term Experiments

### Personality Development (Phase 2)
Once base system proves valuable:

1. **The Realist**: "What power dynamics underlie these passages?"
2. **The Idealist**: "What shared hopes unite these passages?"
3. **The Historian**: "What patterns repeat across these time periods?"

Run parallel systems, compare their interpretations of same events.

### Adversarial Testing
- Can the system maintain coherence with contradictory information?
- Does it recognize and flag historical revisionism?
- How does it handle propaganda vs analysis?

## Data Collection
Track everything:
- Which nodes get retrieved most
- Semantic distance between questions and retrieved content  
- User feedback on browser plugin suggestions
- Computational costs per phase

This isn't just an experiment - it's an attempt to understand how understanding itself emerges.
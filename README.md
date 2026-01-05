# ML Fundamentals Study Tool

Study system for ML/LLM interviews with spaced repetition (~130 items: concepts, quiz, deep dives, derivations).

## Ask a Question

```bash
./study              # Start session (10 items)
./study -n 20        # Study 20 items
./study -t concepts  # Focus on concepts, quiz, deep, or derive
```

**During study:**
- Press ENTER to reveal answer
- Rate yourself: `1` (no idea) → `2` (partial) → `3` (got it)
- Press `a` to get AI grading on your answer
- Press `c` to chat with Claude about the topic
- Press `s` to skip

## Track Progress

```bash
./study --stats      # View statistics
open progress.html   # Open dashboard in browser
```

Dashboard shows:
- Overall stats (total/studied/mastered)
- Progress by category
- Study activity heatmap
- Item-level details

---

**Content:** 80 concepts • 30 quiz • 12 deep • 5 derivations
**Topics:** Transformers, LLM training, RLHF, distributed systems, optimization
**Progress:** Auto-saved to `.study_state.json`

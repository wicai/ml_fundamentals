# ML Fundamentals Study Tool

Study system for ML/LLM interviews with spaced repetition (~130 items: concepts, quiz, deep dives, derivations).

## Ask a Question

```bash
./study                                    # Start session (1 item)
./study -n 20                              # Study 20 items
./study -t concepts                        # Focus on concepts, quiz, deep, or derive
./study -t coding                          # Focus on coding questions
./study -s code021_llm_classifier_logits   # Study a specific item by ID
```

**During study:**
- Press ENTER to reveal answer
- Rate yourself: `1` (no idea) → `2` (partial) → `3` (got it)
- Press `a` to get AI grading on your answer
- Press `c` to chat with Claude about the topic
- Press `s` to skip

**For coding questions:**
- **First time**: Walkthrough mode - Claude explains the solution step-by-step
- **After walkthrough**: Implementation mode - code it yourself from scratch
- Press `w` anytime to see the walkthrough again

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

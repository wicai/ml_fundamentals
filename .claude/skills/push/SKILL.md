---
name: push
description: Push all changes including study state updates to GitHub. Stages all modified and new files, commits with a descriptive message, and pushes to the remote repository.
disable-model-invocation: false
argument-hint: [optional-commit-message]
---

# Push Study Progress to GitHub

Push all changes including study state (.study_state.json) and any new/modified study materials to GitHub.

## Process

1. **Check current status**:
   ```bash
   git status
   ```

2. **Stage all changes** (including study state and new files):
   ```bash
   git add -A
   ```

3. **Create commit** with the provided message or a default one:
   - If `$ARGUMENTS` is provided, use it as the commit message
   - Otherwise, use "Update study progress and materials"

4. **Push to remote**:
   ```bash
   git push
   ```

## Files typically included

- `.study_state.json` - Study progress tracking
- `concepts/*.md` - Concept explanations
- `coding/*.md` - Coding exercises
- `coding/*_solution.py` - Solution files
- `README.md` - Progress overview

## Usage

```
/push                           # Uses default commit message
/push "Add new concept notes"   # Uses custom commit message
```

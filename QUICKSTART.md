# Quick Start Guide

## Running on Mac

Your Mac has `python3` but not `python`. Use one of these methods:

### Method 1: Use the launcher script (easiest)
```bash
./study              # Run with defaults
./study -n 20        # 20 items
./study --stats      # View progress
```

### Method 2: Use python3 directly
```bash
python3 study.py
python3 study.py -n 15 -t concepts
python3 study.py --stats
```

### Method 3: Create python alias (optional)
```bash
alias python=python3
python study.py
```

## First Time Setup

1. Navigate to the directory:
```bash
cd ~/Documents/interview/ml_fundamentals
```

2. Run your first session:
```bash
./study
```

3. The tool will:
   - Select 10 items (mix of concepts/quiz)
   - Show question → press ENTER → reveal answer
   - Rate yourself 1-3:
     - **1** = No idea
     - **2** = Partial understanding  
     - **3** = Got it completely

4. Progress is automatically saved to `.study_state.json`

## Tips

- Start with 5-10 items per day
- Rate honestly (system learns your weaknesses)
- Items you struggle with will appear more often
- Consistency beats intensity (daily > sporadic)

## Troubleshooting

**Permission denied when running `./study`?**
```bash
chmod +x study
```

**Want to reset progress?**
```bash
rm .study_state.json
```

**See what files exist:**
```bash
ls concepts/ | head -10   # First 10 concepts
ls quiz/                   # All quiz items
```

Happy studying! 🚀

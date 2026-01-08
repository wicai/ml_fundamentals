#!/usr/bin/env python3
"""
ML Fundamentals Study Tool
Interactive study session with spaced repetition
"""
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

class StudySession:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or Path(__file__).parent)
        self.state_file = self.base_dir / '.study_state.json'
        self.state = self.load_state()

    def load_state(self):
        """Load progress tracking state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            'items': {},  # item_id -> {last_seen, interval, ease_factor, correct_streak}
            'session_history': []
        }

    def save_state(self):
        """Save progress tracking state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def parse_item(self, filepath):
        """Parse a markdown study item"""
        with open(filepath) as f:
            content = f.read()

        # Extract metadata and content
        meta = {}
        lines = content.split('\n')

        title = lines[0].replace('# ', '').strip() if lines else filepath.stem

        for line in lines[1:20]:  # Check first 20 lines for metadata
            if line.startswith('**Category:**'):
                meta['category'] = line.split(':**')[1].strip()
            elif line.startswith('**Difficulty:**'):
                meta['difficulty'] = int(line.split(':**')[1].strip()[0])
            elif line.startswith('**Tags:**'):
                meta['tags'] = [t.strip() for t in line.split(':**')[1].split(',')]

        # Extract question and answer (handle different formats)
        # Try "Question" first, then "Prompt" for deep dive items, then "Problem" for derivations
        question_match = re.search(r'## Question\s*\n(.*?)(?=\n## )', content, re.DOTALL)
        if not question_match:
            question_match = re.search(r'## Prompt\s*\n(.*?)(?=\n## |$)', content, re.DOTALL)
        if not question_match:
            question_match = re.search(r'## Problem\s*\n(.*?)(?=\n## |$)', content, re.DOTALL)

        answer_match = re.search(r'## Answer\s*\n(.*?)(?=\n## |$)', content, re.DOTALL)
        if not answer_match:
            # For deep/derive items, use "Notes" as the answer
            answer_match = re.search(r'## Notes\s*\n(.*?)$', content, re.DOTALL)
        if not answer_match:
            # For derivation items, use "Instructions" as the answer/guidance
            answer_match = re.search(r'## Instructions\s*\n(.*?)(?=\n## |$)', content, re.DOTALL)

        followup_match = re.search(r'## Follow-up Questions\s*\n(.*?)$', content, re.DOTALL)

        return {
            'id': filepath.stem,
            'filepath': filepath,
            'title': title,
            'category': meta.get('category', 'unknown'),
            'difficulty': meta.get('difficulty', 3),
            'tags': meta.get('tags', []),
            'question': question_match.group(1).strip() if question_match else '',
            'answer': answer_match.group(1).strip() if answer_match else '',
            'followup': followup_match.group(1).strip() if followup_match else ''
        }

    def get_all_items(self, item_type=None, tag_filter=None):
        """Get all study items, optionally filtered by type and/or tag"""
        items = []

        if item_type is None:
            dirs = ['concepts', 'quiz', 'deep', 'derive', 'coding']
        else:
            dirs = [item_type]

        for dir_name in dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                for filepath in sorted(dir_path.glob('*.md')):
                    item = self.parse_item(filepath)
                    # Filter by tag if specified
                    if tag_filter is None or tag_filter.lower() in [t.lower() for t in item['tags']]:
                        items.append(item)

        return items

    def calculate_priority(self, item):
        """Calculate study priority using spaced repetition"""
        item_id = item['id']

        if item_id not in self.state['items']:
            # New item - high priority
            return 1000

        state = self.state['items'][item_id]
        last_seen = datetime.fromisoformat(state['last_seen'])
        interval_days = state.get('interval', 1)

        # Calculate days since last review
        days_since = (datetime.now() - last_seen).days

        # Priority = how overdue (days_since / interval)
        # Higher = more overdue
        priority = days_since / max(interval_days, 0.1)

        return priority

    def select_items(self, count=10, weights=None, tag_filter=None):
        """Select items for study session"""
        if weights is None:
            weights = {'concepts': 0.6, 'quiz': 0.2, 'deep': 0.0, 'derive': 0.1, 'coding': 0.1}

        selected = []

        for item_type, weight in weights.items():
            n = int(count * weight)
            if n == 0 and random.random() < weight * count:
                n = 1  # Probabilistically include at least one

            items = self.get_all_items(item_type, tag_filter=tag_filter)
            if not items:
                continue

            # Sort by priority (SRS)
            items.sort(key=self.calculate_priority, reverse=True)

            # Take top N, with some randomness
            pool_size = min(len(items), n * 3)
            pool = items[:pool_size]
            selected.extend(random.sample(pool, min(n, len(pool))))

        # Ensure we always return at least one item if requested and available
        if count > 0 and len(selected) == 0:
            # Get all available items and pick the highest priority one
            all_items = self.get_all_items(tag_filter=tag_filter)
            if all_items:
                all_items.sort(key=self.calculate_priority, reverse=True)
                selected.append(all_items[0])

        random.shuffle(selected)
        return selected

    def save_chat_context(self, item):
        """Save current item context for chat mode"""
        context_file = self.base_dir / '.chat_context.md'

        with open(context_file, 'w') as f:
            f.write(f"# Study Topic: {item['title']}\n\n")
            f.write(f"**Category:** {item['category']}\n")
            f.write(f"**Difficulty:** {item['difficulty']}/5\n")
            f.write(f"**Tags:** {', '.join(item['tags'])}\n\n")
            f.write(f"## Question\n\n{item['question']}\n\n")
            f.write(f"## Answer\n\n{item['answer']}\n\n")
            if item['followup']:
                f.write(f"## Follow-up Questions\n\n{item['followup']}\n\n")
            f.write(f"## Related File\n\n")
            f.write(f"Full content: {item['filepath']}\n")

    def launch_claude_chat(self, item):
        """Launch Claude web in browser with context prompt via URL parameter"""
        # Create the initial prompt for Claude
        prompt = f"""I'm studying the following ML topic. Please respond with just "OK" so I can ask my questions.

Topic: {item['title']}
Category: {item['category']}
Difficulty: {item['difficulty']}/5

## Question
{item['question']}

## Answer
{item['answer']}"""

        if item['followup']:
            prompt += f"\n\n## Follow-up Questions\n{item['followup']}"

        # Platform-specific browser launch
        if sys.platform == 'darwin':  # macOS
            import urllib.parse
            import time

            # URL-encode the prompt
            encoded_prompt = urllib.parse.quote(prompt)

            # Open Claude.ai with the prompt pre-filled using URL parameter
            url = f'https://claude.ai/new?q={encoded_prompt}'

            subprocess.run(['open', url])

            # Wait for browser to load, then auto-submit with AppleScript
            time.sleep(3)  # Give browser time to load

            applescript = '''
            tell application "System Events"
                keystroke return
            end tell
            '''

            subprocess.run(['osascript', '-e', applescript])

            return True

        elif sys.platform == 'linux':
            # Copy to clipboard (try xclip or xsel)
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'], input=prompt.encode('utf-8'), check=True)
            except FileNotFoundError:
                try:
                    subprocess.run(['xsel', '--clipboard'], input=prompt.encode('utf-8'), check=True)
                except FileNotFoundError:
                    return False
            # Open Claude web
            subprocess.run(['xdg-open', 'https://claude.ai/new'])
            # Note: Auto-paste on Linux would require xdotool, which may not be installed
            return True

        elif sys.platform == 'win32':  # Windows
            # Copy to clipboard
            subprocess.run(['clip'], input=prompt.encode('utf-16le'), check=True)
            # Open Claude web
            subprocess.run(['start', 'https://claude.ai/new'], shell=True)
            # Note: Auto-paste on Windows would require additional setup
            return True
        else:
            return False

    def launch_claude_grading(self, item, user_answer):
        """Launch Claude web to grade user's answer in MLE interview style"""
        # Check if this is a coding question
        is_coding = item['category'] == 'coding' or 'coding' in item['tags']

        if is_coding:
            # Code review style prompt
            prompt = f"""You are conducting an MLE (Machine Learning Engineer) coding interview. Please review the following implementation.

**Coding Task:**
{item['question']}

**Candidate's Implementation:**
```python
{user_answer}
```

**Reference Implementation:**
{item['answer']}

Please provide:
1. **Score**: Rate 1-5 (1=doesn't work, 5=excellent implementation)
2. **Correctness**: Does it work? Any bugs or logical errors?
3. **Code quality**: Clean, readable, well-structured?
4. **What was good**: Highlight strengths in the implementation
5. **What could be improved**: Missing edge cases, efficiency issues, style issues
6. **Interview feedback**: Would this pass an MLE interview? What would make it stronger?

Keep the feedback constructive and specific."""
        else:
            # Conceptual question prompt
            prompt = f"""You are conducting an MLE (Machine Learning Engineer) interview. Please grade the following answer.

**Interview Question:**
{item['question']}

**Candidate's Answer:**
{user_answer}

**Reference Answer:**
{item['answer']}

Please provide:
1. **Score**: Rate 1-5 (1=insufficient, 5=excellent)
2. **What was good**: Highlight correct points and strong aspects
3. **What was missing**: Key concepts or details that should have been mentioned
4. **Interview feedback**: Would this answer pass an MLE interview? What would make it stronger?

Keep the feedback constructive and specific."""

        # Platform-specific browser launch
        if sys.platform == 'darwin':  # macOS
            import urllib.parse
            import time

            # URL-encode the prompt
            encoded_prompt = urllib.parse.quote(prompt)

            # Open Claude.ai with the prompt pre-filled using URL parameter
            url = f'https://claude.ai/new?q={encoded_prompt}'

            subprocess.run(['open', url])

            # Wait for browser to load, then auto-submit with AppleScript
            time.sleep(3)  # Give browser time to load

            applescript = '''
            tell application "System Events"
                keystroke return
            end tell
            '''

            subprocess.run(['osascript', '-e', applescript])

            return True

        elif sys.platform == 'linux':
            # Copy to clipboard (try xclip or xsel)
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'], input=prompt.encode('utf-8'), check=True)
            except FileNotFoundError:
                try:
                    subprocess.run(['xsel', '--clipboard'], input=prompt.encode('utf-8'), check=True)
                except FileNotFoundError:
                    return False
            # Open Claude web
            subprocess.run(['xdg-open', 'https://claude.ai/new'])
            return True

        elif sys.platform == 'win32':  # Windows
            # Copy to clipboard
            subprocess.run(['clip'], input=prompt.encode('utf-16le'), check=True)
            # Open Claude web
            subprocess.run(['start', 'https://claude.ai/new'], shell=True)
            return True
        else:
            return False

    def update_item_state(self, item_id, correct):
        """Update SRS state after reviewing an item"""
        if item_id not in self.state['items']:
            self.state['items'][item_id] = {
                'last_seen': datetime.now().isoformat(),
                'interval': 1,
                'ease_factor': 2.5,
                'correct_streak': 0
            }

        state = self.state['items'][item_id]

        if correct:
            state['correct_streak'] += 1
            # Increase interval based on ease factor
            state['interval'] = state['interval'] * state['ease_factor']
            # Slightly increase ease factor
            state['ease_factor'] = min(state['ease_factor'] + 0.1, 3.0)
        else:
            state['correct_streak'] = 0
            # Reset interval
            state['interval'] = 1
            # Decrease ease factor
            state['ease_factor'] = max(state['ease_factor'] - 0.2, 1.3)

        state['last_seen'] = datetime.now().isoformat()

    def run_session(self, num_items=10, tag_filter=None):
        """Run an interactive study session"""
        print("=" * 70)
        print("ML FUNDAMENTALS STUDY SESSION")
        if tag_filter:
            print(f"Filter: {tag_filter}")
        print("=" * 70)
        print()

        items = self.select_items(num_items, tag_filter=tag_filter)

        if not items:
            print("No study items found. Please add content to the directories.")
            return

        print(f"Selected {len(items)} items for this session\n")

        results = []

        # Initialize current session in history
        current_session = {
            'timestamp': datetime.now().isoformat(),
            'num_items': len(items),
            'results': []
        }
        self.state['session_history'].append(current_session)

        for i, item in enumerate(items, 1):
            print(f"\n{'=' * 70}")
            print(f"ITEM {i}/{len(items)} - {item['category'].upper()}")
            print(f"{'=' * 70}\n")
            print(f"📝 {item['title']}\n")
            print(f"Difficulty: {'⭐' * item['difficulty']}\n")
            print("-" * 70)
            print(item['question'])
            print("-" * 70)

            # Handle coding questions differently
            is_coding = item['category'] == 'coding' or 'coding' in item['tags']

            if is_coding:
                print("\n💻 CODING QUESTION")

                # Create coding directory if it doesn't exist
                coding_dir = self.base_dir / 'coding'
                coding_dir.mkdir(exist_ok=True)

                # Create a file for this coding problem
                coding_file = coding_dir / f"{item['id']}_solution.py"

                # Write the problem as comments in the file
                # Extract code skeleton vs description
                question_text = item['question']

                # Find code blocks (between ```python and ```)
                code_blocks = []
                parts = question_text.split('```python')
                description = parts[0]  # Everything before first code block

                for part in parts[1:]:
                    if '```' in part:
                        code, rest = part.split('```', 1)
                        code_blocks.append(code.strip())
                        description += rest
                    else:
                        description += part

                with open(coding_file, 'w') as f:
                    # Write description as comments
                    f.write(f"# {item['title']}\n")
                    f.write("# " + "=" * 68 + "\n#\n")

                    for line in description.strip().split('\n'):
                        f.write(f"# {line}\n")

                    f.write("#\n")
                    f.write("# " + "=" * 68 + "\n\n")

                    # Write code skeleton as actual code
                    if code_blocks:
                        for code_block in code_blocks:
                            f.write(code_block)
                            f.write("\n\n")
                    else:
                        f.write("# Your solution here:\n\n")

                # Open in VS Code (using macOS 'open' command to avoid Cursor conflict)
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', '-a', 'Visual Studio Code', str(coding_file)])
                    else:
                        subprocess.run(['code', str(coding_file)])
                    print(f"✓ Opened {coding_file.name} in VS Code")
                except Exception as e:
                    print(f"⚠ Could not open VS Code: {e}")
                    print(f"File created at: {coding_file}")

                print("\nImplement your solution, then come back here.")
                print("\nOptions:")
                print("  ENTER: Continue when ready (reads from file)")
                print("  p: Paste your solution manually")
                print("  s: Skip to see answer")

                choice = input("\nYour choice: ").strip().lower()

                if choice == 'p':
                    print("\n📝 Paste your code (type 'END' on a new line when done):")
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == 'END':
                            break
                        lines.append(line)
                    user_answer = '\n'.join(lines)
                elif choice == 's':
                    user_answer = ''
                else:
                    # Read from the file
                    try:
                        with open(coding_file, 'r') as f:
                            content = f.read()
                            # Extract only the code after the docstring
                            if '# Your solution here:' in content:
                                user_answer = content.split('# Your solution here:')[1].strip()
                            else:
                                user_answer = content

                        # Auto-test the solution if user wrote code
                        if user_answer.strip():
                            print("\n" + "=" * 70)
                            print("🧪 TESTING YOUR SOLUTION")
                            print("=" * 70 + "\n")

                            # Create test file with user's solution + test code from answer
                            test_file = coding_dir / f"{item['id']}_test.py"

                            # Extract test code from answer section (between Testing and next ##)
                            test_code_match = re.search(r'\*\*Testing[^*]*?\*\*.*?```python\s*(.*?)```', item['answer'], re.DOTALL)

                            if test_code_match:
                                test_code = test_code_match.group(1).strip()

                                # Write test file
                                with open(test_file, 'w') as f:
                                    # Standard imports with error handling
                                    f.write("from typing import *\n")
                                    f.write("try:\n")
                                    f.write("    import torch\n")
                                    f.write("    import torch.nn as nn\n")
                                    f.write("    import torch.nn.functional as F\n")
                                    f.write("except ImportError:\n")
                                    f.write("    print('ERROR: PyTorch not installed. Run: pip3 install torch')\n")
                                    f.write("    exit(1)\n")
                                    f.write("import math\n")
                                    f.write("try:\n")
                                    f.write("    import numpy as np\n")
                                    f.write("except ImportError:\n")
                                    f.write("    print('ERROR: NumPy not installed. Run: pip3 install numpy')\n")
                                    f.write("    exit(1)\n\n")
                                    f.write("# User's solution\n")
                                    f.write(user_answer)
                                    f.write("\n\n# Tests\n")
                                    f.write(test_code)

                                # Run tests
                                try:
                                    # Try python3 first, fall back to python
                                    python_cmd = 'python3' if subprocess.run(['which', 'python3'], capture_output=True).returncode == 0 else 'python'

                                    result = subprocess.run(
                                        [python_cmd, str(test_file)],
                                        capture_output=True,
                                        text=True,
                                        timeout=10,
                                        cwd=str(coding_dir)
                                    )

                                    if result.returncode == 0:
                                        print("✅ Tests passed!")
                                        print("\nOutput:")
                                        print(result.stdout)
                                    else:
                                        print("❌ Tests failed!")
                                        if result.stderr:
                                            print("\nError:")
                                            print(result.stderr[:500])  # Limit error output
                                        if result.stdout:
                                            print("\nOutput:")
                                            print(result.stdout[:500])

                                except subprocess.TimeoutExpired:
                                    print("⚠ Tests timed out (>10s)")
                                except Exception as e:
                                    print(f"⚠ Could not run tests: {e}")

                                # Clean up test file
                                try:
                                    test_file.unlink()
                                except:
                                    pass
                            else:
                                print("⚠ No tests found in problem description")

                            print("\n" + "=" * 70)
                            input("\n[Press ENTER to see reference answer...]")

                    except Exception as e:
                        print(f"⚠ Could not read file: {e}")
                        user_answer = ''
            else:
                print("\n[Type your answer below, or press ENTER to skip]")
                user_answer = input("Your answer: ").strip()

            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70 + "\n")
            print(item['answer'])

            if item['followup']:
                print("\n" + "-" * 70)
                print("FOLLOW-UP QUESTIONS")
                print("-" * 70)
                print(item['followup'])

            print("\n" + "=" * 70)

            # Get user feedback with chat option
            while True:
                print("\nOptions:")
                print("  1-3: Rate your understanding (1=no idea, 2=partial, 3=got it)")
                if user_answer:
                    print("  a: Get your answer graded by Claude (MLE interview style)")
                print("  c: Chat about this topic (deep dive with context)")
                print("  s: Skip to next item")
                response = input("\nYour choice: ").strip().lower()

                if response == 'a' and user_answer:
                    # Answer grading mode
                    print("\n" + "=" * 70)
                    print("📊 ANSWER GRADING MODE")
                    print("=" * 70)
                    print(f"\nOpening Claude to grade your answer for: {item['title']}\n")

                    success = self.launch_claude_grading(item, user_answer)

                    if success:
                        print("✓ Claude web opened with your answer for grading!")
                        print("✓ Claude will grade your answer as if this were an MLE interview.\n")
                    else:
                        print("⚠ Could not auto-launch browser.\n")

                    input("[Press ENTER when done reviewing to continue...]")
                    continue

                elif response == 'c':
                    # Chat mode - automatically launch Claude Web
                    print("\n" + "=" * 70)
                    print("💬 CHAT MODE")
                    print("=" * 70)
                    print(f"\nOpening Claude web with context for: {item['title']}\n")

                    success = self.launch_claude_chat(item)

                    if success == 'permission_error':
                        print("⚠️  PERMISSION REQUIRED FOR AUTO-PASTE")
                        print("\nTo enable auto-paste, grant accessibility permissions:")
                        print("  1. Open System Settings")
                        print("  2. Go to Privacy & Security → Accessibility")
                        print("  3. Enable access for 'Terminal' (or your terminal app)")
                        print("  4. Try again!")
                        print("\nFor now, the prompt is copied to clipboard.")
                        print("Just paste it manually with Cmd+V in Claude.ai\n")
                    elif success:
                        print("✓ Claude web opened in your browser!")
                        print("✓ Auto-pasting in 5 seconds...")
                        print("\nThe prompt will be automatically pasted and submitted.")
                        print("Claude will respond with 'OK', then you can ask your questions!\n")
                    else:
                        print("⚠ Could not auto-launch browser.")
                        print("\nManual steps:")
                        print("  1. Open https://claude.ai/new")
                        print(f"  2. Ask about: {item['title']}\n")

                    input("[Press ENTER when done chatting to continue...]")
                    continue

                elif response == 's':
                    # Skip - treat as partial knowledge
                    response = '2'
                    break

                elif response in ['1', '2', '3']:
                    break

                if user_answer:
                    print("Please enter 1, 2, 3, a, c, or s")
                else:
                    print("Please enter 1, 2, 3, c, or s")

            correct = response == '3'
            self.update_item_state(item['id'], correct)

            result = {
                'item_id': item['id'],
                'title': item['title'],
                'rating': int(response)
            }
            results.append(result)

            # Update current session results and save state after each question
            current_session['results'] = results
            current_session['num_items'] = len(results)  # Update to actual count
            current_session['avg_rating'] = sum(r['rating'] for r in results) / len(results)
            self.save_state()

        # Session summary
        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)

        avg_rating = sum(r['rating'] for r in results) / len(results)
        print(f"\nAverage rating: {avg_rating:.1f}/3")
        print(f"Items mastered (3/3): {sum(1 for r in results if r['rating'] == 3)}/{len(results)}")

        print("\n✓ Progress saved after each question\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ML Fundamentals Study Tool')
    parser.add_argument('-n', '--num-items', type=int, default=1,
                        help='Number of items to study (default: 1)')
    parser.add_argument('-t', '--type', choices=['concepts', 'quiz', 'deep', 'derive', 'coding'],
                        help='Focus on specific item type')
    parser.add_argument('--stats', action='store_true',
                        help='Show study statistics')
    parser.add_argument('tag', nargs='?', default=None,
                        help='Filter by tag (e.g., safety, agents, coding)')

    args = parser.parse_args()

    session = StudySession()

    if args.stats:
        # Show statistics
        print("Study Statistics")
        print("=" * 70)
        print(f"Total items studied: {len(session.state['items'])}")
        print(f"Total sessions: {len(session.state['session_history'])}")

        if session.state['session_history']:
            recent = session.state['session_history'][-5:]
            print(f"\nRecent sessions (last {len(recent)}):")
            for s in recent:
                dt = datetime.fromisoformat(s['timestamp'])
                print(f"  {dt.strftime('%Y-%m-%d %H:%M')} - {s['num_items']} items - avg {s['avg_rating']:.1f}/3")
    else:
        # Run study session
        weights = None
        if args.type:
            weights = {args.type: 1.0}

        if weights:
            session.run_session(args.num_items, tag_filter=args.tag)
        else:
            session.run_session(args.num_items, tag_filter=args.tag)

if __name__ == '__main__':
    main()

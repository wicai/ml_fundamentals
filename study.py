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
        # Try "Question" first, then "Prompt" for deep dive items
        question_match = re.search(r'## Question\s*\n(.*?)(?=\n## )', content, re.DOTALL)
        if not question_match:
            question_match = re.search(r'## Prompt\s*\n(.*?)(?=\n## |$)', content, re.DOTALL)

        answer_match = re.search(r'## Answer\s*\n(.*?)(?=\n## |$)', content, re.DOTALL)
        if not answer_match:
            # For deep/derive items, use "Notes" as the answer
            answer_match = re.search(r'## Notes\s*\n(.*?)$', content, re.DOTALL)

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

    def get_all_items(self, item_type=None):
        """Get all study items, optionally filtered by type"""
        items = []

        if item_type is None:
            dirs = ['concepts', 'quiz', 'deep', 'derive']
        else:
            dirs = [item_type]

        for dir_name in dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                for filepath in sorted(dir_path.glob('*.md')):
                    items.append(self.parse_item(filepath))

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

    def select_items(self, count=10, weights=None):
        """Select items for study session"""
        if weights is None:
            weights = {'concepts': 0.7, 'quiz': 0.2, 'deep': 0.05, 'derive': 0.05}

        selected = []

        for item_type, weight in weights.items():
            n = int(count * weight)
            if n == 0 and random.random() < weight * count:
                n = 1  # Probabilistically include at least one

            items = self.get_all_items(item_type)
            if not items:
                continue

            # Sort by priority (SRS)
            items.sort(key=self.calculate_priority, reverse=True)

            # Take top N, with some randomness
            pool_size = min(len(items), n * 3)
            pool = items[:pool_size]
            selected.extend(random.sample(pool, min(n, len(pool))))

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

    def run_session(self, num_items=10):
        """Run an interactive study session"""
        print("=" * 70)
        print("ML FUNDAMENTALS STUDY SESSION")
        print("=" * 70)
        print()

        items = self.select_items(num_items)

        if not items:
            print("No study items found. Please add content to the directories.")
            return

        print(f"Selected {len(items)} items for this session\n")

        results = []

        for i, item in enumerate(items, 1):
            print(f"\n{'=' * 70}")
            print(f"ITEM {i}/{len(items)} - {item['category'].upper()}")
            print(f"{'=' * 70}\n")
            print(f"📝 {item['title']}\n")
            print(f"Difficulty: {'⭐' * item['difficulty']}\n")
            print("-" * 70)
            print(item['question'])
            print("-" * 70)

            input("\n[Press ENTER to reveal answer]")

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
                print("  c: Chat about this topic (deep dive with context)")
                print("  s: Skip to next item")
                response = input("\nYour choice: ").strip().lower()

                if response == 'c':
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

                print("Please enter 1, 2, 3, c, or s")

            correct = response == '3'
            self.update_item_state(item['id'], correct)
            results.append({
                'item_id': item['id'],
                'title': item['title'],
                'rating': int(response)
            })

        # Session summary
        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)

        avg_rating = sum(r['rating'] for r in results) / len(results)
        print(f"\nAverage rating: {avg_rating:.1f}/3")
        print(f"Items mastered (3/3): {sum(1 for r in results if r['rating'] == 3)}/{len(results)}")

        # Save session to history
        self.state['session_history'].append({
            'timestamp': datetime.now().isoformat(),
            'num_items': len(items),
            'avg_rating': avg_rating,
            'results': results
        })

        self.save_state()
        print("\n✓ Progress saved\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ML Fundamentals Study Tool')
    parser.add_argument('-n', '--num-items', type=int, default=10,
                        help='Number of items to study (default: 10)')
    parser.add_argument('-t', '--type', choices=['concepts', 'quiz', 'deep', 'derive'],
                        help='Focus on specific item type')
    parser.add_argument('--stats', action='store_true',
                        help='Show study statistics')

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
            session.run_session(args.num_items)
        else:
            session.run_session(args.num_items)

if __name__ == '__main__':
    main()

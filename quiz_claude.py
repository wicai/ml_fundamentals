#!/usr/bin/env python3
"""
Pick one unseen ML question and launch Claude tutorial mode
"""
import json
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

def get_one_unseen_question(base_dir):
    """Get one question that hasn't been seen yet"""
    state_file = base_dir / '.study_state.json'

    seen_ids = set()
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            seen_ids = set(state.get('items', {}).keys())

    concepts_dir = base_dir / 'concepts'

    for filepath in sorted(concepts_dir.glob('*.md')):
        item_id = filepath.stem
        if item_id not in seen_ids:
            with open(filepath) as f:
                content = f.read()

            lines = content.split('\n')
            title = lines[0].replace('# ', '').strip() if lines else filepath.stem

            question_match = re.search(r'## Question\s*\n(.*?)(?=\n## )', content, re.DOTALL)
            if question_match:
                return {
                    'id': item_id,
                    'title': title,
                    'question': question_match.group(1).strip()
                }

    return None

def launch_tutorial(item):
    """Open Claude.ai with tutorial prompt (same as launch_claude_explain)"""

    prompt = f"""I'm learning about: {item['title']}

Please explain this to me like a patient tutor having a conversation:

IMPORTANT INSTRUCTIONS:
1. Start with ONE short paragraph (2-3 sentences) explaining the most fundamental concept
2. After that paragraph, STOP and ask "Does that make sense so far?"
3. WAIT for my response before continuing
4. When I confirm (yes/got it/makes sense), give me the NEXT piece (another 2-3 sentences)
5. Keep building understanding incrementally - never dump multiple concepts at once
6. Use simple analogies and examples
7. If I ask questions, answer them before moving on

DO NOT give me a wall of text. This should feel like a back-and-forth conversation where you check my understanding after each small piece.

---
The question I need to understand:
{item['question']}

---
Start now with just the first piece - the single most important concept I need to grasp first."""

    encoded_prompt = urllib.parse.quote(prompt)
    url = f'https://claude.ai/new?q={encoded_prompt}'

    subprocess.run(['open', url])

    time.sleep(3)

    applescript = '''
    tell application "System Events"
        keystroke return
    end tell
    '''
    subprocess.run(['osascript', '-e', applescript])

    print(f"Opened tutorial for: {item['title']}")

def mark_as_seen(base_dir, item_id):
    """Mark question as seen so we don't launch it again"""
    state_file = base_dir / '.study_state.json'

    state = {'items': {}, 'session_history': []}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    from datetime import datetime
    state['items'][item_id] = {
        'last_seen': datetime.now().isoformat(),
        'interval': 1,
        'ease_factor': 2.5,
        'correct_streak': 0,
        'explain_done': True  # Mark as explained since we launched tutorial
    }

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def main():
    base_dir = Path(__file__).parent

    item = get_one_unseen_question(base_dir)

    if not item:
        print("No unseen questions found!")
        return

    launch_tutorial(item)
    mark_as_seen(base_dir, item['id'])

if __name__ == '__main__':
    main()

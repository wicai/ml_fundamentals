#!/usr/bin/env python3
"""
Progress Visualization Server
Launch a web UI to visualize study progress
"""
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify
from study import StudySession

app = Flask(__name__)
session = StudySession()

@app.route('/')
def index():
    return render_template('progress.html')

@app.route('/api/progress')
def get_progress():
    """Get all progress data for visualization"""

    # Get all items
    all_items = session.get_all_items()

    # Organize by category
    categories = {}
    for item in all_items:
        cat = item['category']
        if cat not in categories:
            categories[cat] = {
                'total': 0,
                'studied': 0,
                'mastered': 0,
                'items': []
            }

        categories[cat]['total'] += 1

        # Check if studied
        item_state = session.state['items'].get(item['id'])
        if item_state:
            categories[cat]['studied'] += 1
            if item_state.get('correct_streak', 0) >= 3:
                categories[cat]['mastered'] += 1

        # Add item details
        item_data = {
            'id': item['id'],
            'title': item['title'],
            'difficulty': item['difficulty'],
            'tags': item['tags'],
            'studied': item_state is not None,
            'last_seen': item_state['last_seen'] if item_state else None,
            'interval': item_state.get('interval', 0) if item_state else 0,
            'streak': item_state.get('correct_streak', 0) if item_state else 0,
            'ease_factor': item_state.get('ease_factor', 2.5) if item_state else 2.5
        }
        categories[cat]['items'].append(item_data)

    # Session history
    history = session.state.get('session_history', [])

    # Stats
    total_items = len(all_items)
    studied_items = len(session.state['items'])
    mastered_items = sum(1 for s in session.state['items'].values()
                         if s.get('correct_streak', 0) >= 3)

    return jsonify({
        'categories': categories,
        'history': history,
        'stats': {
            'total': total_items,
            'studied': studied_items,
            'mastered': mastered_items,
            'progress_pct': round(studied_items / total_items * 100, 1) if total_items > 0 else 0,
            'mastery_pct': round(mastered_items / total_items * 100, 1) if total_items > 0 else 0
        }
    })

@app.route('/api/heatmap')
def get_heatmap():
    """Get study activity heatmap data"""
    history = session.state.get('session_history', [])

    # Aggregate by date
    activity = {}
    for h in history:
        date = h['timestamp'].split('T')[0]  # Get date part
        if date not in activity:
            activity[date] = {
                'sessions': 0,
                'items': 0,
                'avg_rating': 0
            }
        activity[date]['sessions'] += 1
        activity[date]['items'] += h['num_items']
        activity[date]['avg_rating'] = (
            activity[date]['avg_rating'] * (activity[date]['sessions'] - 1) + h['avg_rating']
        ) / activity[date]['sessions']

    return jsonify(activity)

if __name__ == '__main__':
    print("=" * 70)
    print("ML FUNDAMENTALS PROGRESS VISUALIZER")
    print("=" * 70)
    print("\nStarting server...")
    print("Open http://localhost:5001 in your browser")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    app.run(debug=True, port=5001, use_reloader=False)

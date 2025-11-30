#!/usr/bin/env python3
"""
Test script for URL parameter functionality
This will open Claude.ai with a pre-filled message via URL
"""
import subprocess
import urllib.parse

def test_url_parameter():
    """Test the URL parameter approach"""

    test_prompt = """TEST MESSAGE - Please respond with 'OK' to confirm you received this.

This is a test of the URL parameter functionality.
If you can read this, the URL encoding worked!"""

    print("=" * 70)
    print("URL PARAMETER TEST")
    print("=" * 70)
    print("\nTest message:")
    print("-" * 70)
    print(test_prompt)
    print("-" * 70)

    # URL-encode the prompt
    encoded_prompt = urllib.parse.quote(test_prompt)

    # Build the URL
    url = f'https://claude.ai/new?q={encoded_prompt}'

    print("\n✓ URL-encoded message")
    print(f"\nURL length: {len(url)} characters")
    print("\nOpening Claude.ai with pre-filled message...")

    # Open the URL
    subprocess.run(['open', url])

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nCheck your browser!")
    print("The message should appear in the input field.")
    print("You just need to press Enter to submit it.\n")

    return True

if __name__ == '__main__':
    test_url_parameter()

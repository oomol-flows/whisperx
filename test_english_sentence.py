#!/usr/bin/env python3
"""Test English sentence integrity"""
import sys
import os
import importlib.util

# Import the generate-srt module
task_path = os.path.join(os.path.dirname(__file__), 'tasks', 'generate-srt')
spec = importlib.util.spec_from_file_location("generate_srt", os.path.join(task_path, "__init__.py"))
generate_srt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_srt_module)

main = generate_srt_module.main

# Mock context
class MockContext:
    def preview(self, data):
        print(f"\n{data.get('data', '')}\n")

# Test parameters
params = {
    "json_file": "/tmp/test_english_real.json",
    "include_speaker": False,
    "output_path": "/tmp/test_english_sentence.srt",
    "use_spacy_segmentation": True,
    "spacy_model": "en_core_web_sm",
    "subtitle_language_preset": "english",
    "max_chars_per_subtitle": None,
    "max_duration_per_subtitle": None,
    "chars_per_second": None,
    "max_words_per_subtitle": None
}

# Run
result = main(params, MockContext())

print(f"✓ Generated: {result['srt_file']}")
print(f"✓ Count: {result['subtitle_count']} subtitles\n")

# Show output
with open(result['srt_file'], 'r', encoding='utf-8') as f:
    content = f.read()
    print("=== RESULT ===")
    print(content)
    print("==============\n")

# Analyze
print("ANALYSIS:")
for i, block in enumerate(content.strip().split('\n\n'), 1):
    lines = block.split('\n')
    if len(lines) >= 3:
        text = '\n'.join(lines[2:])
        print(f"{i}. [{len(text)} chars] {text[:80]}{'...' if len(text) > 80 else ''}")

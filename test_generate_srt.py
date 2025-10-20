#!/usr/bin/env python3
"""
Test script for generate-srt task with English content
"""
import json
import sys
import os

# Add tasks directory to path
task_path = os.path.join(os.path.dirname(__file__), 'tasks', 'generate-srt')
sys.path.insert(0, task_path)

# Import directly from __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("generate_srt", os.path.join(task_path, "__init__.py"))
generate_srt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_srt_module)

main = generate_srt_module.main
Inputs = generate_srt_module.Inputs

# Create a mock context
class MockContext:
    def preview(self, data):
        print(f"\n=== PREVIEW ===")
        print(data.get("data", ""))
        print("===============\n")

# Create test data with English content
test_data = {
    "audio_file": "/test/audio.mp3",
    "model_size": "medium",
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 3.5,
            "text": "Hello everyone.",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.95},
                {"word": "everyone.", "start": 0.6, "end": 3.5, "score": 0.92}
            ]
        },
        {
            "start": 3.5,
            "end": 8.2,
            "text": "This is a test of the sentence segmentation system.",
            "words": [
                {"word": "This", "start": 3.5, "end": 3.7, "score": 0.96},
                {"word": "is", "start": 3.8, "end": 3.9, "score": 0.98},
                {"word": "a", "start": 4.0, "end": 4.1, "score": 0.97},
                {"word": "test", "start": 4.2, "end": 4.5, "score": 0.95},
                {"word": "of", "start": 4.6, "end": 4.7, "score": 0.98},
                {"word": "the", "start": 4.8, "end": 4.9, "score": 0.97},
                {"word": "sentence", "start": 5.0, "end": 5.6, "score": 0.94},
                {"word": "segmentation", "start": 5.7, "end": 6.5, "score": 0.93},
                {"word": "system.", "start": 6.6, "end": 8.2, "score": 0.91}
            ]
        },
        {
            "start": 8.3,
            "end": 12.0,
            "text": "We want to ensure that complete sentences stay together.",
            "words": [
                {"word": "We", "start": 8.3, "end": 8.4, "score": 0.97},
                {"word": "want", "start": 8.5, "end": 8.7, "score": 0.96},
                {"word": "to", "start": 8.8, "end": 8.9, "score": 0.98},
                {"word": "ensure", "start": 9.0, "end": 9.4, "score": 0.95},
                {"word": "that", "start": 9.5, "end": 9.7, "score": 0.97},
                {"word": "complete", "start": 9.8, "end": 10.2, "score": 0.94},
                {"word": "sentences", "start": 10.3, "end": 10.8, "score": 0.93},
                {"word": "stay", "start": 10.9, "end": 11.2, "score": 0.95},
                {"word": "together.", "start": 11.3, "end": 12.0, "score": 0.92}
            ]
        },
        {
            "start": 12.1,
            "end": 16.5,
            "text": "Even if they are a bit longer than the standard character limits.",
            "words": [
                {"word": "Even", "start": 12.1, "end": 12.3, "score": 0.96},
                {"word": "if", "start": 12.4, "end": 12.5, "score": 0.98},
                {"word": "they", "start": 12.6, "end": 12.8, "score": 0.97},
                {"word": "are", "start": 12.9, "end": 13.0, "score": 0.98},
                {"word": "a", "start": 13.1, "end": 13.2, "score": 0.97},
                {"word": "bit", "start": 13.3, "end": 13.5, "score": 0.96},
                {"word": "longer", "start": 13.6, "end": 14.0, "score": 0.94},
                {"word": "than", "start": 14.1, "end": 14.3, "score": 0.97},
                {"word": "the", "start": 14.4, "end": 14.5, "score": 0.98},
                {"word": "standard", "start": 14.6, "end": 15.1, "score": 0.93},
                {"word": "character", "start": 15.2, "end": 15.7, "score": 0.92},
                {"word": "limits.", "start": 15.8, "end": 16.5, "score": 0.91}
            ]
        },
        {
            "start": 16.6,
            "end": 19.0,
            "text": "This ensures better readability and comprehension.",
            "words": [
                {"word": "This", "start": 16.6, "end": 16.8, "score": 0.96},
                {"word": "ensures", "start": 16.9, "end": 17.3, "score": 0.94},
                {"word": "better", "start": 17.4, "end": 17.7, "score": 0.95},
                {"word": "readability", "start": 17.8, "end": 18.4, "score": 0.92},
                {"word": "and", "start": 18.5, "end": 18.6, "score": 0.98},
                {"word": "comprehension.", "start": 18.7, "end": 19.0, "score": 0.90}
            ]
        }
    ]
}

# Save test JSON file
test_json_path = "/tmp/test_english.json"
with open(test_json_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"Created test JSON: {test_json_path}")

# Test parameters
params: Inputs = {
    "json_file": test_json_path,
    "include_speaker": False,
    "output_path": "/tmp/test_english_output.srt",
    "use_spacy_segmentation": True,
    "spacy_model": "en_core_web_sm",
    "subtitle_language_preset": "english",
    "max_chars_per_subtitle": None,
    "max_duration_per_subtitle": None,
    "chars_per_second": None,
    "max_words_per_subtitle": None
}

# Run the main function
context = MockContext()
result = main(params, context)

# Print results
print(f"\n✓ Generated SRT file: {result['srt_file']}")
print(f"✓ Subtitle count: {result['subtitle_count']}")

# Print the generated SRT content
print("\n=== GENERATED SRT CONTENT ===")
with open(result['srt_file'], 'r', encoding='utf-8') as f:
    print(f.read())
print("=============================\n")

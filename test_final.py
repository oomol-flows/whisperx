#!/usr/bin/env python3
"""Final test - English and Chinese subtitle generation"""
import sys, os
import importlib.util

# Import module
spec = importlib.util.spec_from_file_location('gen_srt', 'tasks/generate-srt/__init__.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class Ctx:
    def preview(self, d): pass

# Test English
print("=" * 70)
print("ENGLISH SUBTITLE TEST")
print("=" * 70)
result_en = mod.main({
    'json_file': '/oomol-driver/oomol-storage/whisperx-output/bm_audio_transcription.json',
    'include_speaker': False,
    'output_path': '/tmp/final_test_en.srt',
    'use_spacy_segmentation': True,
    'spacy_model': 'auto',
    'subtitle_language_preset': 'auto',
    'max_chars_per_subtitle': None,
    'max_duration_per_subtitle': None,
    'chars_per_second': None,
    'max_words_per_subtitle': None
}, Ctx())

with open('/tmp/final_test_en.srt') as f:
    blocks = f.read().strip().split('\n\n')[:8]
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            text = ' '.join(lines[2:])
            print(f"{lines[0]:>3}. {text}")

print(f"\n✓ Total: {result_en['subtitle_count']} subtitles")

# Test Chinese
print("\n" + "=" * 70)
print("CHINESE SUBTITLE TEST (if available)")
print("=" * 70)

chinese_file = '/oomol-driver/oomol-storage/whisperx-output/《酌见2》俞敏洪对话张朝阳#以价值观为导向的人生意义_audio_transcription.json'
if os.path.exists(chinese_file):
    result_zh = mod.main({
        'json_file': chinese_file,
        'include_speaker': False,
        'output_path': '/tmp/final_test_zh.srt',
        'use_spacy_segmentation': True,
        'spacy_model': 'auto',
        'subtitle_language_preset': 'auto',
        'max_chars_per_subtitle': None,
        'max_duration_per_subtitle': None,
        'chars_per_second': None,
        'max_words_per_subtitle': None
    }, Ctx())

    with open('/tmp/final_test_zh.srt', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')[:8]
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                text = ' '.join(lines[2:])
                print(f"{lines[0]:>3}. {text}")

    print(f"\n✓ Total: {result_zh['subtitle_count']} subtitles")
else:
    print("Chinese test file not found")

print("\n" + "=" * 70)
print("TEST COMPLETE - Sentence integrity prioritized!")
print("=" * 70)

#region generated meta
import typing
class Inputs(typing.TypedDict):
    json_file: str
    include_speaker: bool
    output_path: str | None
    use_spacy_segmentation: bool
    spacy_model: str
    max_chars_per_subtitle: int
    max_duration_per_subtitle: float
    chars_per_second: float
class Outputs(typing.TypedDict):
    srt_file: typing.NotRequired[str]
    subtitle_count: typing.NotRequired[int]
#endregion

from oocana import Context
import json
import os
import subprocess
import sys
from pathlib import Path

import spacy
from typing import List, Dict, Any


def download_spacy_model(model_name: str) -> bool:
    """
    Download a spaCy model if not already installed

    Args:
        model_name: Name of the spaCy model to download

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading spaCy model: {model_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Successfully downloaded: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {model_name}: {e}")
        return False


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def smart_join_words(words: List[Dict[str, Any]]) -> str:
    """
    Intelligently join words with proper spacing for mixed Chinese-English text

    Args:
        words: List of word objects with 'word' key

    Returns:
        Properly formatted text string
    """
    if not words:
        return ""

    result = []
    for i, word_obj in enumerate(words):
        word = word_obj.get("word", "").strip()
        if not word:
            continue

        # Add word to result
        if i == 0:
            result.append(word)
        else:
            prev_word = words[i - 1].get("word", "").strip()

            # Check if previous or current word contains Chinese characters
            def has_chinese(text):
                return any('\u4e00' <= char <= '\u9fff' for char in text)

            prev_is_chinese = has_chinese(prev_word[-1]) if prev_word else False
            curr_is_chinese = has_chinese(word[0]) if word else False

            # Add space only when both words are non-Chinese (e.g., English)
            # or when transitioning between punctuation and words
            needs_space = not (prev_is_chinese or curr_is_chinese)

            if needs_space:
                result.append(" " + word)
            else:
                result.append(word)

    return "".join(result)


def detect_language(text: str) -> str:
    """
    Detect the primary language from text and return appropriate spaCy model

    Args:
        text: Sample text to analyze

    Returns:
        Recommended spaCy model name
    """
    # Count character types
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    japanese_count = sum(1 for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff')

    # If significant Chinese characters, use Chinese model
    if chinese_count > len(text) * 0.1:
        return "zh_core_web_sm"

    # If significant Japanese characters, use Japanese model
    if japanese_count > len(text) * 0.1:
        return "ja_core_news_sm"

    # Default to English
    return "en_core_web_sm"


def merge_segments_by_sentences(
    segments: List[Dict[str, Any]],
    spacy_model: str,
    max_chars: int,
    max_duration: float = 5.0,
    chars_per_second: float = 4.0
) -> List[Dict[str, Any]]:
    """
    Merge WhisperX segments based on sentence boundaries using spaCy

    Optimized for Chinese subtitles with mixed Chinese-English support:
    - Default max_chars: 20 (15-20 characters per line for Chinese)
    - Default chars_per_second: 4 (reading speed for Chinese)
    - Default max_duration: 5 seconds per subtitle
    - Intelligent word spacing for mixed language content

    Args:
        segments: List of WhisperX segment dictionaries
        spacy_model: spaCy model name to use (or 'auto' for auto-detection)
        max_chars: Maximum characters per subtitle
        max_duration: Maximum duration per subtitle in seconds
        chars_per_second: Maximum reading speed in characters per second

    Returns:
        List of merged segments with sentence boundaries
    """
    # Auto-detect language if requested
    if spacy_model == "auto":
        sample_text = " ".join([seg.get("text", "") for seg in segments[:5]])
        spacy_model = detect_language(sample_text)
        print(f"Auto-detected language model: {spacy_model}")

    try:
        # Try to load the spaCy model
        nlp = spacy.load(spacy_model)
        print(f"spaCy model loaded: {spacy_model}")

    except OSError:
        # Model not found, try to download it
        print(f"spaCy model '{spacy_model}' not found locally")
        if download_spacy_model(spacy_model):
            try:
                # Reload after download
                nlp = spacy.load(spacy_model)
                print(f"spaCy model loaded: {spacy_model}")
            except OSError:
                print(f"Error: Failed to load model after download. Using original segments.")
                return segments
        else:
            print(f"Error: Could not download model. Using original segments.")
            return segments

    # Combine all text to detect sentence boundaries
    full_text = smart_join_words([{"word": seg.get("text", "").strip()} for seg in segments])
    doc = nlp(full_text)

    # Get sentence boundaries
    sentences = [sent.text.strip() for sent in doc.sents]

    # Build word-level timeline from segments
    words_timeline = []
    for segment in segments:
        if "words" in segment:
            words_timeline.extend(segment["words"])
        else:
            # If no word-level data, use segment-level
            words_timeline.append({
                "word": segment.get("text", ""),
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "speaker": segment.get("speaker")
            })

    # Match sentences to timeline
    merged_segments = []
    word_idx = 0

    for sentence in sentences:
        if not sentence:
            continue

        # Find words in this sentence
        sentence_words = []
        sentence_text = ""

        while word_idx < len(words_timeline) and len(sentence_text) < len(sentence):
            word_obj = words_timeline[word_idx]
            word_text = word_obj.get("word", "").strip()
            sentence_text = smart_join_words(sentence_words + [word_obj])
            sentence_words.append(word_obj)
            word_idx += 1

        if not sentence_words:
            continue

        # Split by constraints: max_chars, max_duration, and chars_per_second
        current_segment_words = []

        for word_obj in sentence_words:
            # Calculate what the new segment would look like
            test_words = current_segment_words + [word_obj]
            test_text = smart_join_words(test_words)
            test_duration = test_words[-1].get("end", 0) - test_words[0].get("start", 0)

            # Check all constraints
            text_too_long = len(test_text) > max_chars
            duration_too_long = max_duration > 0 and test_duration > max_duration
            reading_too_fast = chars_per_second > 0 and test_duration > 0 and len(test_text) / test_duration > chars_per_second * 1.2

            # If adding this word violates constraints, save current segment
            if current_segment_words and (text_too_long or duration_too_long):
                merged_segments.append({
                    "start": current_segment_words[0].get("start", 0),
                    "end": current_segment_words[-1].get("end", 0),
                    "text": smart_join_words(current_segment_words),
                    "speaker": current_segment_words[0].get("speaker"),
                    "words": current_segment_words
                })
                current_segment_words = []

            current_segment_words.append(word_obj)

        # Add remaining words as final segment
        if current_segment_words:
            merged_segments.append({
                "start": current_segment_words[0].get("start", 0),
                "end": current_segment_words[-1].get("end", 0),
                "text": smart_join_words(current_segment_words),
                "speaker": current_segment_words[0].get("speaker"),
                "words": current_segment_words
            })

    return merged_segments


def generate_srt(segments: list, include_speaker: bool = True) -> str:
    """
    Generate SRT content from WhisperX segments

    Args:
        segments: List of segment dictionaries with start, end, text, and optional speaker
        include_speaker: Whether to include speaker labels

    Returns:
        SRT formatted string
    """
    srt_content = []

    for i, segment in enumerate(segments, start=1):
        # Get segment data
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()

        if not text:
            continue

        # Format timestamps
        start_ts = format_timestamp(start_time)
        end_ts = format_timestamp(end_time)

        # Add speaker label if available and requested
        if include_speaker:
            # Check for speaker in segment or in first word
            speaker = segment.get("speaker")
            if not speaker and "words" in segment and segment["words"]:
                speaker = segment["words"][0].get("speaker")

            if speaker:
                text = f"[{speaker}] {text}"

        # Build SRT entry
        srt_entry = f"{i}\n{start_ts} --> {end_ts}\n{text}\n"
        srt_content.append(srt_entry)

    return "\n".join(srt_content)


def main(params: Inputs, context: Context) -> Outputs:
    """
    Convert WhisperX JSON transcription to SRT subtitle format with spaCy sentence segmentation

    Args:
        params: Input parameters
        context: OOMOL context object

    Returns:
        Output with SRT file path and subtitle count
    """
    # Read JSON file
    json_file = params["json_file"]

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get segments
    segments = data.get("segments", [])

    if not segments:
        raise ValueError("No segments found in the JSON file")

    # Apply spaCy sentence segmentation if enabled
    use_spacy = params.get("use_spacy_segmentation", True)
    if use_spacy:
        spacy_model = params.get("spacy_model", "en_core_web_sm")
        max_chars = params.get("max_chars_per_subtitle", 20)
        max_duration = params.get("max_duration_per_subtitle", 5.0)
        chars_per_second = params.get("chars_per_second", 4.0)
        segments = merge_segments_by_sentences(
            segments,
            spacy_model,
            max_chars,
            max_duration,
            chars_per_second
        )

    # Generate SRT content
    include_speaker = params.get("include_speaker", True)
    srt_content = generate_srt(segments, include_speaker)

    # Determine output path
    output_path = params.get("output_path")
    if not output_path:
        # Auto-generate output path based on input file
        json_path = Path(json_file)
        output_path = str(json_path.parent / f"{json_path.stem}.srt")

    # Write SRT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    # Count subtitles
    subtitle_count = len(segments)

    # Preview first few entries
    preview_lines = srt_content.split("\n\n")[:3]
    preview_text = "\n\n".join(preview_lines)

    context.preview({
        "type": "text",
        "data": f"SRT Preview (first 3 entries):\n\n{preview_text}"
    })

    return {
        "srt_file": output_path,
        "subtitle_count": subtitle_count
    }

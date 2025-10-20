#region generated meta
import typing
class Inputs(typing.TypedDict):
    json_file: str
    include_speaker: bool | None
    output_path: str | None
    use_spacy_segmentation: bool | None
    spacy_model: typing.Literal["auto", "en_core_web_sm", "zh_core_web_sm", "ja_core_news_sm", "de_core_news_sm", "fr_core_news_sm", "es_core_news_sm"] | None
    subtitle_language_preset: typing.Literal["auto", "english", "chinese", "japanese", "custom"] | None
    max_chars_per_subtitle: int | None
    max_duration_per_subtitle: float | None
    chars_per_second: float | None
    max_words_per_subtitle: int | None
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


def detect_language(text: str) -> tuple[str, str]:
    """
    Detect the primary language from text and return appropriate spaCy model and language preset

    Args:
        text: Sample text to analyze

    Returns:
        Tuple of (spaCy model name, language preset)
    """
    # Count character types
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    japanese_count = sum(1 for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff')

    # If significant Chinese characters, use Chinese model
    if chinese_count > len(text) * 0.1:
        return ("zh_core_web_sm", "chinese")

    # If significant Japanese characters, use Japanese model
    if japanese_count > len(text) * 0.1:
        return ("ja_core_news_sm", "japanese")

    # Default to English
    return ("en_core_web_sm", "english")


def apply_language_preset(preset: str) -> dict:
    """
    Apply language-specific subtitle standards

    Args:
        preset: Language preset name

    Returns:
        Dictionary with subtitle parameters
    """
    presets = {
        "english": {
            "max_chars": 84,  # Professional standard: 42-84 chars for comfortable reading
            "max_duration": 7.0,  # Allow longer duration for complete sentences
            "chars_per_second": 21.0,  # Comfortable: 17-21, max 41
            "max_words": 20,  # Increased to keep sentences together
        },
        "chinese": {
            "max_chars": 35,  # 15-20 per line, 30-40 for double line
            "max_duration": 6.0,  # Longer duration for reading (35 chars / 5 CPS ≈ 7s)
            "chars_per_second": 5.0,  # ~4-6 chars/second for Chinese
            "max_words": 0,  # Not applicable for Chinese
        },
        "japanese": {
            "max_chars": 20,  # Similar to Chinese
            "max_duration": 5.0,
            "chars_per_second": 4.0,
            "max_words": 0,
        },
    }

    return presets.get(preset, presets["chinese"])  # Default to Chinese


def merge_segments_by_sentences(
    segments: List[Dict[str, Any]],
    spacy_model: str,
    max_chars: int,
    max_duration: float = 3.0,
    chars_per_second: float = 21.0,
    max_words_per_subtitle: int = 15
) -> List[Dict[str, Any]]:
    """
    Merge WhisperX segments based on sentence boundaries using spaCy

    English subtitle standards (professional):
    - chars_per_second: 17-21 comfortable, max 41
    - max_chars: 42-84 characters (for 2-3 second display)
    - max_duration: 2-3 seconds per subtitle
    - max_words_per_subtitle: 7 words (single line) / 15 words (double line)

    Chinese subtitle standards:
    - max_chars: 15-20 characters
    - chars_per_second: ~4
    - max_duration: 5 seconds
    - Intelligent word spacing for mixed language content

    Args:
        segments: List of WhisperX segment dictionaries
        spacy_model: spaCy model name to use (or 'auto' for auto-detection)
        max_chars: Maximum characters per subtitle
        max_duration: Maximum duration per subtitle in seconds
        chars_per_second: Maximum reading speed in characters per second
        max_words_per_subtitle: Maximum words per subtitle block (0 = no limit, for English only)

    Returns:
        List of merged segments with sentence boundaries
    """
    # Auto-detect language if requested
    if spacy_model == "auto":
        sample_text = " ".join([seg.get("text", "") for seg in segments[:5]])
        spacy_model, _ = detect_language(sample_text)
        print(f"Auto-detected spaCy model: {spacy_model}")

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

        # Try to keep the entire sentence together first
        full_sentence_text = smart_join_words(sentence_words)
        full_sentence_duration = sentence_words[-1].get("end", 0) - sentence_words[0].get("start", 0)
        full_sentence_word_count = len([w for w in sentence_words if w.get("word", "").strip()])

        # Check if the entire sentence fits within constraints
        sentence_fits = (
            len(full_sentence_text) <= max_chars and
            (max_duration == 0 or full_sentence_duration <= max_duration) and
            (max_words_per_subtitle == 0 or full_sentence_word_count <= max_words_per_subtitle)
        )

        # If entire sentence fits, keep it together
        if sentence_fits:
            merged_segments.append({
                "start": sentence_words[0].get("start", 0),
                "end": sentence_words[-1].get("end", 0),
                "text": full_sentence_text,
                "speaker": sentence_words[0].get("speaker"),
                "words": sentence_words
            })
        else:
            # Only split if necessary - use word-by-word approach
            current_segment_words = []

            for word_obj in sentence_words:
                # Calculate what the new segment would look like
                test_words = current_segment_words + [word_obj]
                test_text = smart_join_words(test_words)
                test_duration = test_words[-1].get("end", 0) - test_words[0].get("start", 0)
                test_word_count = len([w for w in test_words if w.get("word", "").strip()])

                # Check critical constraints (prioritize keeping text together)
                text_too_long = len(test_text) > max_chars
                duration_too_long = max_duration > 0 and test_duration > max_duration * 1.5  # Allow 50% buffer
                too_many_words = max_words_per_subtitle > 0 and test_word_count > max_words_per_subtitle * 1.3  # Allow 30% buffer

                # If adding this word violates hard constraints, save current segment
                if current_segment_words and (text_too_long or duration_too_long or too_many_words):
                    merged_segments.append({
                        "start": current_segment_words[0].get("start", 0),
                        "end": current_segment_words[-1].get("end", 0),
                        "text": smart_join_words(current_segment_words),
                        "speaker": current_segment_words[0].get("speaker"),
                        "words": current_segment_words
                    })
                    current_segment_words = []

                current_segment_words.append(word_obj)

            # Add remaining words as final segment (only if we split the sentence)
            if current_segment_words:
                merged_segments.append({
                    "start": current_segment_words[0].get("start", 0),
                    "end": current_segment_words[-1].get("end", 0),
                    "text": smart_join_words(current_segment_words),
                    "speaker": current_segment_words[0].get("speaker"),
                    "words": current_segment_words
                })

    return merged_segments


def smart_line_break(text: str, max_chars_per_line: int = 20) -> str:
    """
    Intelligently break text into two lines for subtitle display

    Args:
        text: Text to break
        max_chars_per_line: Target characters per line (15-20 for Chinese, ~42 for English)

    Returns:
        Text with line break (\n) inserted at optimal position
    """
    # If text is short enough for single line, return as-is
    if len(text) <= max_chars_per_line:
        return text

    # Detect if text is primarily Chinese
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    is_chinese = chinese_count > len(text) * 0.3

    # Define break points (punctuation and natural breaks)
    if is_chinese:
        # Chinese punctuation marks
        break_chars = '，。、；：！？,;:!?'
        # Ideal break position: around max_chars_per_line (15-20 for Chinese)
        ideal_pos = max_chars_per_line
    else:
        # English: break at spaces or punctuation
        break_chars = ' ,.;:!?'
        # For English, target around 42 chars (half of 84)
        ideal_pos = max_chars_per_line

    # Try to find break point near the middle
    # Search window: from ideal_pos-5 to ideal_pos+5
    best_pos = -1

    # First, look for punctuation marks in the ideal range
    for offset in range(6):  # Check ±5 chars from ideal position
        # Check forward
        pos_forward = ideal_pos + offset
        if pos_forward < len(text) and text[pos_forward] in break_chars:
            best_pos = pos_forward + 1  # Break after punctuation
            break

        # Check backward
        pos_backward = ideal_pos - offset
        if pos_backward > 0 and text[pos_backward] in break_chars:
            best_pos = pos_backward + 1  # Break after punctuation
            break

    # If no punctuation found, for English try to break at nearest space
    if best_pos == -1 and not is_chinese:
        for offset in range(ideal_pos // 2):  # Expand search range
            pos_forward = ideal_pos + offset
            if pos_forward < len(text) and text[pos_forward] == ' ':
                best_pos = pos_forward + 1
                break

            pos_backward = ideal_pos - offset
            if pos_backward > 0 and text[pos_backward] == ' ':
                best_pos = pos_backward + 1
                break

    # If still no good break point, force break at ideal position
    if best_pos == -1:
        best_pos = ideal_pos

    # Insert line break
    line1 = text[:best_pos].strip()
    line2 = text[best_pos:].strip()

    return f"{line1}\n{line2}"


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

        # Apply smart line breaking
        # Detect language and apply appropriate line length
        chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        is_chinese = chinese_count > len(text) * 0.3

        if is_chinese:
            # Chinese: break at 15-20 chars
            text = smart_line_break(text, max_chars_per_line=18)
        else:
            # English: break at ~42 chars
            text = smart_line_break(text, max_chars_per_line=42)

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

    # Apply spaCy sentence segmentation if enabled (default: True)
    use_spacy = params.get("use_spacy_segmentation")
    if use_spacy is None:
        use_spacy = True  # Best practice: use intelligent segmentation by default

    if use_spacy:
        spacy_model = params.get("spacy_model")
        if spacy_model is None:
            spacy_model = "auto"  # Best practice: auto-detect language

        language_preset = params.get("subtitle_language_preset")
        if language_preset is None:
            language_preset = "auto"  # Best practice: auto-detect language standards

        # Auto-detect language preset if needed
        if language_preset == "auto" or language_preset is None:
            sample_text = " ".join([seg.get("text", "") for seg in segments[:5]])
            _, language_preset = detect_language(sample_text)
            print(f"Auto-detected language preset: {language_preset}")

        # Apply language preset if not custom
        if language_preset != "custom":
            preset_params = apply_language_preset(language_preset)

            # Use preset values, but allow user overrides
            max_chars = params.get("max_chars_per_subtitle")
            if max_chars is None:
                max_chars = preset_params["max_chars"]

            max_duration = params.get("max_duration_per_subtitle")
            if max_duration is None:
                max_duration = preset_params["max_duration"]

            chars_per_second = params.get("chars_per_second")
            if chars_per_second is None:
                chars_per_second = preset_params["chars_per_second"]

            max_words_per_subtitle = params.get("max_words_per_subtitle")
            if max_words_per_subtitle is None:
                max_words_per_subtitle = preset_params["max_words"]

            print(f"Applied {language_preset} preset: {max_chars} chars, {chars_per_second} CPS, {max_duration}s max")
        else:
            # Custom mode: use user values or fallback defaults
            max_chars = params.get("max_chars_per_subtitle") or 20
            max_duration = params.get("max_duration_per_subtitle") or 3.0
            chars_per_second = params.get("chars_per_second") or 4.0
            max_words_per_subtitle = params.get("max_words_per_subtitle") or 0

        segments = merge_segments_by_sentences(
            segments,
            spacy_model,
            max_chars,
            max_duration,
            chars_per_second,
            max_words_per_subtitle
        )

    # Generate SRT content
    include_speaker = params.get("include_speaker")
    if include_speaker is None:
        include_speaker = True  # Best practice: include speaker labels by default

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

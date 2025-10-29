#region generated meta
import typing
from oocana import LLMModelOptions
class Inputs(typing.TypedDict):
    srt_file: str
    target_language: typing.Literal["Chinese", "English", "Japanese", "Korean", "Spanish", "French", "German", "Russian"]
    source_language: str | None
    batch_size: int | None
    output_path: str | None
    llm: LLMModelOptions
class Outputs(typing.TypedDict):
    translated_srt_file: typing.NotRequired[str]
    total_entries: typing.NotRequired[int]
#endregion

from oocana import Context
from openai import OpenAI
import re
import os
from pathlib import Path
from typing import List, Dict, Any
import time


class SRTEntry:
    """Represents a single SRT subtitle entry"""
    def __init__(self, index: int, start_time: str, end_time: str, text: str):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

    def to_srt_format(self) -> str:
        """Convert entry to SRT format string"""
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"


def parse_srt_file(file_path: str) -> List[SRTEntry]:
    """
    Parse SRT file and extract all subtitle entries

    Args:
        file_path: Path to the SRT file

    Returns:
        List of SRTEntry objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newline to separate entries
    entries = []

    # Pattern to match SRT entries
    # Format:
    # 1
    # 00:00:00,000 --> 00:00:05,000
    # Subtitle text (can be multiple lines)
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n((?:.*\n?)+?)(?=\n\d+\n|\Z)'

    matches = re.finditer(pattern, content, re.MULTILINE)

    for match in matches:
        index = int(match.group(1))
        start_time = match.group(2)
        end_time = match.group(3)
        text = match.group(4).strip()

        entries.append(SRTEntry(index, start_time, end_time, text))

    return entries


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (approximately 1 token per 4 characters for English/European languages,
    1 token per 2 characters for Chinese)

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    # Simple heuristic: count Chinese characters vs others
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    # Chinese: ~2 chars per token, Other: ~4 chars per token
    return (chinese_chars // 2) + (other_chars // 4)


def translate_batch(
    entries: List[SRTEntry],
    target_language: str,
    source_language: str | None,
    client: OpenAI,
    llm_config: dict,
    context: Context,
    max_input_tokens: int = 100000
) -> List[str]:
    """
    Translate a batch of subtitle entries using LLM

    Args:
        entries: List of SRT entries to translate
        target_language: Target language for translation
        source_language: Source language (optional)
        client: OpenAI client instance
        llm_config: LLM configuration
        context: OOMOL context
        max_input_tokens: Maximum input tokens to prevent context overflow

    Returns:
        List of translated texts
    """
    # Prepare batch text for translation
    batch_text = ""
    for entry in entries:
        batch_text += f"[{entry.index}] {entry.text}\n"

    # Construct prompt
    source_lang_hint = f" from {source_language}" if source_language else ""

    prompt = f"""You are a professional subtitle translator. Translate the following subtitles{source_lang_hint} to {target_language}.

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
1. Translate EVERY subtitle entry - do not skip or omit any entries
2. Preserve the [index] numbers EXACTLY as they appear
3. Output ONLY the translated lines in format: [index] translated_text
4. Each translation must be on a SINGLE line (no line breaks within translations)
5. Do NOT add any explanations, comments, or additional content
6. Do NOT add blank lines between translations
7. Preserve speaker labels if present (e.g., [Speaker_01])

Example format:
[1] First translated subtitle
[2] Second translated subtitle
[3] Third translated subtitle

Subtitles to translate:

{batch_text}

Translated subtitles:"""

    # Estimate tokens to prevent overflow
    estimated_tokens = estimate_tokens(prompt)
    max_output_tokens = min(llm_config.get("max_tokens", 4096), 8192)  # Cap at model limit

    if estimated_tokens > max_input_tokens:
        raise ValueError(
            f"Batch too large: estimated {estimated_tokens} input tokens exceeds limit of {max_input_tokens}. "
            f"Please reduce batch_size parameter."
        )

    total_estimated = estimated_tokens + max_output_tokens
    if total_estimated > 250000:  # Leave buffer below 262144 limit
        print(f"Warning: Total estimated tokens ({total_estimated}) is high. Consider reducing batch_size or max_tokens.")

    # Call LLM API
    try:
        response = client.chat.completions.create(
            model=llm_config.get("model", "oomol-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config.get("temperature", 0.3),
            top_p=llm_config.get("top_p", 1.0),
            max_tokens=max_output_tokens
        )

        translated_text = response.choices[0].message.content.strip()

        # Parse translated text back into individual entries with strict index validation
        translated_entries = []
        index_to_translation = {}  # Map index to translation
        lines = translated_text.split('\n')

        # First pass: collect all valid translations with their indices
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match pattern [index] text (strict validation)
            match = re.match(r'\[(\d+)\]\s*(.+)', line)
            if match:
                index = int(match.group(1))
                translation = match.group(2).strip()

                # Only accept translations with indices matching our entries
                if any(entry.index == index for entry in entries):
                    if index in index_to_translation:
                        print(f"Warning: Duplicate translation for index [{index}], keeping first occurrence")
                    else:
                        index_to_translation[index] = translation
                else:
                    print(f"Warning: Ignoring translation with invalid index [{index}] (not in current batch)")
            else:
                # Log lines that don't match expected format
                if len(line) > 5:  # Ignore very short lines (likely fragments)
                    print(f"Warning: Ignoring malformed line (no index): {line[:50]}...")

        # Second pass: build ordered translation list matching input entries
        for entry in entries:
            if entry.index in index_to_translation:
                translated_entries.append(index_to_translation[entry.index])
            else:
                print(f"Warning: Missing translation for entry [{entry.index}], using original text")
                translated_entries.append(entry.text)

        # Verification summary
        found_count = len(index_to_translation)
        expected_count = len(entries)

        if found_count != expected_count:
            print(f"Translation mismatch: Expected {expected_count}, found {found_count} valid translations")
            print(f"Missing indices: {[e.index for e in entries if e.index not in index_to_translation]}")
        else:
            print(f"Successfully translated {found_count}/{expected_count} entries")

        return translated_entries

    except Exception as e:
        error_msg = str(e)
        print(f"Error during translation: {error_msg}")

        # Check if it's a token limit error
        if "token limit" in error_msg.lower() or "400" in error_msg:
            print(f"Token limit exceeded. This batch has {len(entries)} entries.")
            print(f"Estimated input tokens: {estimated_tokens}")
            print(f"Please reduce batch_size parameter and try again.")

        # Return original text as fallback
        return [entry.text for entry in entries]


def main(params: Inputs, context: Context) -> Outputs:
    """
    Translate SRT subtitle file to target language using LLM

    Args:
        params: Input parameters
        context: OOMOL context object

    Returns:
        Output with translated SRT file path and entry count
    """
    srt_file = params["srt_file"]
    target_language = params["target_language"]
    source_language = params.get("source_language")
    batch_size = params.get("batch_size") or 10
    output_path = params.get("output_path")
    llm_config = params["llm"]

    # Validate input file
    if not os.path.exists(srt_file):
        raise FileNotFoundError(f"SRT file not found: {srt_file}")

    print(f"Parsing SRT file: {srt_file}")

    # Parse SRT file
    entries = parse_srt_file(srt_file)
    total_entries = len(entries)

    if total_entries == 0:
        raise ValueError("No subtitle entries found in the SRT file")

    print(f"Found {total_entries} subtitle entries")

    # Initialize OpenAI client
    base_url = context.oomol_llm_env.get("base_url_v1")
    api_key = context.oomol_llm_env.get("api_key")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0
    )

    # Translate in batches
    translated_entries = []

    for i in range(0, total_entries, batch_size):
        batch_end = min(i + batch_size, total_entries)
        batch = entries[i:batch_end]

        print(f"Translating entries {i+1}-{batch_end} of {total_entries}...")

        # Translate batch
        translated_texts = translate_batch(
            batch,
            target_language,
            source_language,
            client,
            llm_config,
            context
        )

        # Create translated SRT entries
        for j, entry in enumerate(batch):
            translated_entry = SRTEntry(
                entry.index,
                entry.start_time,
                entry.end_time,
                translated_texts[j] if j < len(translated_texts) else entry.text
            )
            translated_entries.append(translated_entry)

        # Small delay to avoid rate limiting
        if batch_end < total_entries:
            time.sleep(0.5)

    # Generate output path if not provided
    if not output_path:
        input_path = Path(srt_file)
        output_path = str(input_path.parent / f"{input_path.stem}_{target_language.lower()}.srt")

    # Write translated SRT file
    print(f"Writing translated SRT to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in translated_entries:
            f.write(entry.to_srt_format())
            f.write('\n')

    # Preview first few entries
    preview_entries = translated_entries[:3]
    preview_text = "\n".join([entry.to_srt_format() for entry in preview_entries])

    context.preview({
        "type": "text",
        "data": f"Translation complete!\n\nTranslated {total_entries} entries to {target_language}\n\nPreview (first 3 entries):\n\n{preview_text}"
    })

    return {
        "translated_srt_file": output_path,
        "total_entries": total_entries
    }

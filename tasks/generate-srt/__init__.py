#region generated meta
import typing
class Inputs(typing.TypedDict):
    json_file: str
    include_speaker: bool
    output_path: str | None
class Outputs(typing.TypedDict):
    srt_file: typing.NotRequired[str]
    subtitle_count: typing.NotRequired[int]
#endregion

from oocana import Context
import json
import os
from pathlib import Path


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
    Convert WhisperX JSON transcription to SRT subtitle format

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

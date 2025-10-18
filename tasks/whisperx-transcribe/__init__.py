# region generated meta
import typing


class Inputs(typing.TypedDict):
    audio_file: str
    model_size: str
    language: str
    batch_size: int
    enable_alignment: bool
    enable_diarization: bool
    hf_token: typing.Optional[str]
    compute_type: str


class Outputs(typing.TypedDict):
    transcript_text: str
    segments: str
    output_file: str


# endregion

from oocana import Context
import json
import os
import sys
from pathlib import Path
import ctypes

# Preload cuDNN libraries to fix "Cannot load symbol cudnnCreateConvolutionDescriptor" error
try:
    site_packages = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    cudnn_lib_dir = site_packages / "nvidia" / "cudnn" / "lib"

    if cudnn_lib_dir.exists():
        # Preload cuDNN libraries in correct order
        cudnn_libs = [
            cudnn_lib_dir / "libcudnn_ops.so.9",
            cudnn_lib_dir / "libcudnn_cnn.so.9",
            cudnn_lib_dir / "libcudnn.so.9",
        ]
        for lib in cudnn_libs:
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    # If preloading fails, continue anyway
    pass

import whisperx
import torch
from whisperx.diarize import DiarizationPipeline


def main(params: Inputs, context: Context) -> Outputs:
    """
    Transcribe audio file using WhisperX with optional alignment and diarization

    Args:
        params: Input parameters
        context: OOMOL context object

    Returns:
        Transcription results with text, segments, and output file path
    """
    audio_file = params["audio_file"]
    model_size = params["model_size"]
    language = params.get("language") or ""
    language = language.strip() if language else None
    batch_size = params.get("batch_size", 16)
    enable_alignment = params.get("enable_alignment", True)
    enable_diarization = params.get("enable_diarization", False)
    hf_token = params.get("hf_token") or ""
    hf_token = hf_token.strip() if hf_token else None
    compute_type = params.get("compute_type", "float16")

    # Debug: Check diarization settings
    print(f"Enable diarization: {enable_diarization}")
    print(f"HF token available: {bool(hf_token)}")
    print(f"HF token length: {len(hf_token) if hf_token else 0}")

    # Validate input file
    if not os.path.exists(audio_file):
        raise ValueError(f"Audio file not found: {audio_file}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adjust compute type for CPU
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    # Load audio
    audio = whisperx.load_audio(audio_file)

    # Load model
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=language
    )

    # Transcribe
    result = model.transcribe(audio, batch_size=batch_size)

    # Extract detected language if not specified
    detected_language = result.get("language", language or "unknown")

    # Align whisper output
    if enable_alignment:
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            print(f"✓ Alignment completed successfully")
        except Exception as e:
            print(f"⚠ Alignment failed: {str(e)}")
            # Continue with unaligned results
            pass

    # Speaker diarization
    if enable_diarization and hf_token:
        try:
            print(f"Starting speaker diarization...")
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"✓ Speaker diarization completed successfully")
        except Exception as e:
            print(f"⚠ Speaker diarization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue without diarization
            pass

    # Extract full text
    segments = result.get("segments", [])
    full_text = " ".join([seg.get("text", "").strip() for seg in segments])

    # Prepare output directory
    output_dir = Path("/oomol-driver/oomol-storage/whisperx-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    input_filename = Path(audio_file).stem
    output_file = output_dir / f"{input_filename}_transcription.json"

    # Save results to JSON
    output_data = {
        "audio_file": audio_file,
        "model_size": model_size,
        "language": detected_language,
        "full_text": full_text,
        "segments": segments,
        "word_count": len(full_text.split()),
        "duration": segments[-1].get("end", 0) if segments else 0
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Preview results
    preview_text = f"""# Transcription Results

**File:** {Path(audio_file).name}
**Language:** {detected_language}
**Model:** {model_size}
**Duration:** {output_data['duration']:.2f}s
**Word Count:** {output_data['word_count']}

## Full Text

{full_text}

---

## Segments ({len(segments)} total)

"""

    # Add first few segments as preview
    for i, seg in enumerate(segments[:5]):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        speaker_info = f" **[{speaker}]**" if speaker else ""
        preview_text += f"**{start:.2f}s - {end:.2f}s**{speaker_info}: {text}\n\n"

    if len(segments) > 5:
        preview_text += f"\n... and {len(segments) - 5} more segments\n"

    context.preview({
        "type": "markdown",
        "data": preview_text
    })

    return {
        "transcript_text": full_text,
        "segments": json.dumps(segments, ensure_ascii=False),
        "output_file": str(output_file)
    }

# WhisperX Transcribe Task

Fast automatic speech recognition with word-level timestamps and optional speaker diarization.

## Features

- ✅ Fast transcription (70x realtime with GPU)
- ✅ Word-level timestamp alignment
- ✅ Speaker diarization (optional)
- ✅ Multiple language support (auto-detection or manual)
- ✅ Multiple audio format support (mp3, wav, m4a, flac, ogg, etc.)
- ✅ 6 model sizes available (tiny → large-v3)
- ✅ CPU/GPU automatic detection
- ✅ Batch processing optimization

## System Requirements

### Required System Dependencies

**FFmpeg** is required for audio processing. Ensure it's installed in the container:

```bash
apt-get update && apt-get install -y ffmpeg
```

This has been added to `package.oo.yaml` bootstrap script.

### Python Dependencies

The following packages are automatically installed via poetry:

- `whisperx` (>=3.7.4)
- `torch` (GPU acceleration support)
- `torchaudio`
- Other dependencies (automatically resolved)

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_file` | File | Yes | - | Audio file to transcribe (supports mp3, wav, m4a, flac, etc.) |
| `model_size` | Enum | Yes | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `language` | String | No | `null` | Language code (e.g., en, zh, ja). Leave empty for auto-detection |
| `batch_size` | Integer | Yes | 16 | Batch size for inference (higher = faster but more GPU memory) |
| `enable_alignment` | Boolean | Yes | `true` | Enable word-level timestamp alignment |
| `enable_diarization` | Boolean | Yes | `false` | Enable speaker diarization (requires HuggingFace token) |
| `hf_token` | Secret | No | `null` | HuggingFace token (required for diarization) |
| `compute_type` | Enum | Yes | `float16` | Computation precision: `float16` (GPU), `int8` (CPU), `float32` |

## Output

| Output | Type | Description |
|--------|------|-------------|
| `transcript_text` | String | Full transcription text |
| `segments` | JSON String | Transcription segments with timestamps and speaker info |
| `output_file` | String | Path to the output JSON file (saved in `/oomol-driver/oomol-storage/whisperx-output/`) |

## Output JSON Format

```json
{
  "audio_file": "/path/to/audio.mp3",
  "model_size": "base",
  "language": "en",
  "full_text": "Complete transcription text...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 2.5}
      ],
      "speaker": "SPEAKER_00"
    }
  ],
  "word_count": 150,
  "duration": 45.2
}
```

## Usage Example

### Basic Transcription

1. Add the `whisperx-transcribe` task to your flow
2. Connect an audio file input
3. Select model size (start with `base` for balance of speed/accuracy)
4. Run the flow

### With Speaker Diarization

1. Obtain a HuggingFace token from https://huggingface.co/settings/tokens
2. Accept the terms for pyannote models:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
3. Set `enable_diarization` to `true`
4. Provide your HuggingFace token

### Model Size Selection

| Model | Speed | Accuracy | GPU Memory | Use Case |
|-------|-------|----------|------------|----------|
| `tiny` | Fastest | Low | ~1GB | Quick drafts, testing |
| `base` | Fast | Good | ~1GB | General purpose |
| `small` | Medium | Better | ~2GB | Balanced accuracy |
| `medium` | Slower | High | ~5GB | High accuracy needed |
| `large-v2` | Slow | Very High | ~10GB | Production quality |
| `large-v3` | Slowest | Best | ~10GB | Maximum accuracy |

## Performance Tips

1. **GPU vs CPU**: GPU is 10-100x faster. CPU automatically uses `int8` precision.
2. **Batch Size**: Increase for faster processing (if you have enough GPU memory)
3. **Model Size**: Start small, upgrade only if accuracy is insufficient
4. **Alignment**: Disable if you don't need precise word-level timestamps
5. **Language**: Specify language code for faster processing (skips auto-detection)

## Supported Languages

WhisperX supports 90+ languages including:

- English (`en`)
- Chinese (`zh`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Japanese (`ja`)
- Korean (`ko`)
- Arabic (`ar`)
- Russian (`ru`)
- Portuguese (`pt`)
- And many more...

See full list: https://github.com/openai/whisper#available-models-and-languages

## Error Handling

The task handles common errors gracefully:

- **Missing audio file**: Raises `ValueError` with clear message
- **Alignment failure**: Falls back to unaligned timestamps
- **Diarization failure**: Continues without speaker labels
- **GPU memory issues**: Try reducing `batch_size` or use smaller model

## Storage

Output files are saved to:
```
/oomol-driver/oomol-storage/whisperx-output/{filename}_transcription.json
```

## Credits

Based on [WhisperX](https://github.com/m-bain/whisperX) by Max Bain et al.

# WhisperX Transcription & SRT Generation

A powerful audio processing tool that automatically transcribes audio files and generates professional subtitles. This project helps you convert spoken content from audio recordings into accurate text with timing information and speaker identification.

## What This Tool Does

Imagine you have an audio recording of a meeting, interview, lecture, or video content. This tool can:

- **Convert audio to text**: Automatically transcribe spoken words into written text
- **Identify different speakers**: Tell you who said what (useful for meetings with multiple people)
- **Create subtitles**: Generate subtitle files (.srt) that you can use with videos
- **Support multiple languages**: Works with English, Chinese, Japanese, and many other languages
- **Handle various audio formats**: Works with MP3, WAV, M4A, FLAC files and more

## Understanding the Building Blocks (Blocks)

This tool is built using two main blocks that work together:

### 1. WhisperX Transcribe Block 🎵
This is the main transcription engine that:
- Takes your audio file as input
- Uses advanced AI to recognize speech
- Can identify different speakers when multiple people are talking
- Produces accurate text with precise timing information

**Key Features:**
- **Multiple model sizes**: Choose between faster processing or higher accuracy
- **Language detection**: Automatically identifies the language spoken
- **Speaker identification**: Distinguishes between different speakers
- **Precise timing**: Creates word-level timestamps

### 2. Generate SRT Subtitles Block 📝
This block takes the transcription data and:
- Formats it into standard subtitle files (.srt)
- Can include speaker labels (like "SPEAKER_01:", "SPEAKER_02:")
- Creates files that work with video players and editing software

## How It Works: A Simple Workflow

The process works like this:

1. **Upload your audio file** → The WhisperX Transcribe block processes your audio
2. **AI transcription** → It converts speech to text with timing information
3. **Optional: Speaker identification** → It can tell you who said what
4. **Generate subtitles** → The SRT block creates a subtitle file

## When to Use This Tool

### Perfect for:
- **Content creators**: Adding subtitles to YouTube videos or podcasts
- **Business professionals**: Transcribing meetings and interviews
- **Educators**: Creating accessible lecture materials
- **Journalists**: Converting audio interviews to text
- **Researchers**: Analyzing audio recordings and focus groups
- **Video editors**: Adding professional subtitles to video projects

### Real-world examples:
- A marketing team transcribing customer feedback interviews
- A professor making lecture videos accessible with subtitles
- A podcaster creating written transcripts for SEO
- A company documenting important meeting discussions

## Getting Started

### What You'll Need:
1. **Audio file** (MP3, WAV, M4A, FLAC, etc.)
2. **HuggingFace token** (only if you want speaker identification)
   - Free to get from huggingface.co
   - Required for the speaker diarization feature

### Step-by-Step:

1. **Prepare your audio file**
   - Make sure the audio quality is clear
   - Supported formats: MP3, WAV, M4A, FLAC, and more

2. **Configure the transcription settings**
   - **Model size**: Choose accuracy vs. speed (tiny → large-v3)
   - **Language**: Specify or let it auto-detect
   - **Speaker identification**: Turn on if multiple speakers
   - **Timestamps**: Enable for precise timing

3. **Run the workflow**
   - Upload your audio file
   - Click "Start" to begin transcription
   - Wait for processing (time depends on file length)

4. **Get your results**
   - **Transcript text**: Full written transcription
   - **SRT file**: Ready-to-use subtitle file
   - **JSON data**: Detailed timing and speaker information

## Customization Options

### Transcription Settings:
- **Model sizes**: From tiny (fastest) to large-v3 (most accurate)
- **Languages**: 50+ languages supported
- **Processing speed**: Adjust batch size based on your computer's capabilities
- **Precision**: Choose GPU (float16) or CPU (int8) processing

### Subtitle Options:
- **Speaker labels**: Include or exclude speaker identification
- **File naming**: Custom output paths or automatic naming
- **Format compatibility**: Standard SRT format works with all major video players

## Technical Features (for curious users)

- **AI-powered**: Uses OpenAI's Whisper model with enhancements
- **Speaker diarization**: Advanced speaker identification technology
- **Word-level timing**: Precise synchronization for perfect subtitle timing
- **Multi-language support**: Automatic language detection
- **GPU acceleration**: Optional GPU support for faster processing

## Troubleshooting Tips

**Poor transcription quality?**
- Ensure clear audio quality
- Try a larger model size
- Specify the language manually

**Speaker identification not working?**
- Make sure you have a HuggingFace token
- Ensure speakers have distinct voices
- Check that audio has minimal background noise

**Processing taking too long?**
- Try a smaller model size
- Increase batch size if you have sufficient memory
- Use GPU acceleration if available

## Privacy and Security

- All processing happens locally on your system
- Your audio files are never sent to external servers
- API tokens are securely stored and only used for speaker identification features

---

**Ready to get started?** Simply upload your audio file and let the AI do the work of creating accurate transcriptions and professional subtitles!
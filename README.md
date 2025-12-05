# Whisper NPU Transcription for Snapdragon X Elite

This project provides fast Whisper audio transcription using the Snapdragon X Elite NPU (Hexagon HTP).

## Performance Results

Tested on Snapdragon X Elite (X1P64100 @ 3.40 GHz):

| Model | Audio Length | Time | Realtime Factor |
|-------|--------------|------|-----------------|
| large-v3-turbo (NPU) | 455s | 78.5s | **5.8x** |
| tiny (NPU) | 455s | 14.4s | **31.6x** |
| medium (CPU-only) | 455s | 1137s | 0.4x |

**Best quality/speed tradeoff**: `large-v3-turbo` with NPU encoder gives excellent transcription at 5.8x realtime.

## Scripts

### 1. `transcribe_hybrid.py` (Recommended)
Uses NPU for the encoder and CPU for the decoder with KV-cache.

```batch
transcribe_hybrid.bat "path\to\audio.m4a" --model tiny
transcribe_hybrid.bat "path\to\audio.m4a" --model small --language es
transcribe_hybrid.bat "path\to\folder" --model medium --batch
```

Options:
- `--model`: tiny, base, small, medium, large-v3-turbo (default: tiny)
- `--language`: auto-detected, or specify: en, es, etc.
- `--output-dir`: Output directory for transcriptions
- `--no-npu`: Disable NPU encoder (use CPU only)
- `--preprocess`: Audio preprocessing for noise reduction (see below)

### Audio Preprocessing

For noisy recordings, use the `--preprocess` option to clean up audio before transcription:

```batch
REM Basic noise reduction with FFmpeg
transcribe_hybrid.bat "audio.m4a" --preprocess ffmpeg

REM Stronger noise reduction
transcribe_hybrid.bat "audio.m4a" --preprocess ffmpeg --noise-reduction 25

REM Auto-select best available method
transcribe_hybrid.bat "audio.m4a" --preprocess auto
```

Preprocessing methods:
- `ffmpeg`: FFmpeg-based denoising (highpass + afftdn + normalize) - **recommended, no extra dependencies**
- `noisereduce`: Python noisereduce library (requires `pip install noisereduce`)
- `deepfilternet`: AI-powered speech enhancement (best quality, requires `pip install deepfilternet torch torchaudio`)
- `auto`: Automatically select best available method

The `--noise-reduction` option controls aggressiveness (default: 15 dB, range: 5-30).

### 2. `transcribe_ffmpeg.py` (CPU-only)
Pure CPU transcription using HuggingFace Transformers.

```batch
transcribe.bat "path\to\audio.m4a" --model medium --language es
```

### 3. `transcribe_npu.py` (Experimental)
Full NPU transcription (encoder + decoder). Currently slow due to lack of KV-cache.

## Model Quality vs Speed

| Model | Parameters | Quality | Speed (NPU hybrid) |
|-------|------------|---------|-------------------|
| tiny | 39M | Basic | ~30x realtime |
| base | 74M | Good | ~15x realtime |
| small | 244M | Very Good | ~8x realtime |
| medium | 769M | Excellent | ~3x realtime |
| large-v3-turbo | 809M | **Best** | ~5.8x realtime |

**Recommended**: `large-v3-turbo` - optimized for speed with only 4 decoder layers (vs 32 in large-v3).

For mixed Spanish/English meetings, use `large-v3-turbo` or `medium` model.

## Setup

### Prerequisites
- Windows ARM64 (Snapdragon X Elite)
- Python 3.11 ARM64

### Installation

```powershell
# Install Python packages
pip install torch transformers onnxruntime-qnn numpy

# FFmpeg is needed for audio loading
winget install Gyan.FFmpeg
```

### ONNX Models
The hybrid approach uses pre-exported ONNX models from HuggingFace.
They are automatically downloaded to `whisper-tiny-onnx/` directory.

## Architecture

```
Audio File (M4A/MP3/WAV)
         │
         ▼
   ┌─────────────┐
   │   FFmpeg    │  Load & resample to 16kHz
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   NPU       │  Encoder (mel spectrogram → hidden states)
   │  (QNN HTP)  │  Fast parallel computation
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   CPU       │  Decoder with KV-cache
   │  (PyTorch)  │  Autoregressive generation
   └─────────────┘
         │
         ▼
   Text + SRT output
```

## Files

```
whisper-npu/
├── transcribe_hybrid.py      # Hybrid NPU+CPU transcription
├── transcribe_hybrid.bat     # Batch launcher
├── transcribe_ffmpeg.py      # CPU-only transcription
├── transcribe.bat            # CPU launcher
├── transcribe_npu.py         # Full NPU (experimental)
├── whisper-tiny-onnx/        # ONNX models
│   └── onnx/
│       ├── encoder_model_quantized.onnx
│       ├── decoder_model_quantized.onnx
│       └── ...
└── README.md
```

## Troubleshooting

### "Unknown chip model name 'Snapdragon...'"
This is a harmless warning from cpuinfo. The NPU still works correctly.

### FFmpeg not found
Ensure FFmpeg is in your PATH:
```powershell
winget install Gyan.FFmpeg
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

### Low quality transcription
Use a larger model:
```batch
transcribe_hybrid.bat audio.m4a --model small
```

## Technical Details

- **NPU Backend**: Qualcomm QNN HTP (Hexagon Tensor Processor)
- **Encoder Quantization**: INT8/UINT8 for NPU acceleration
- **Decoder**: FP32 with KV-cache for efficient autoregressive generation
- **Audio Processing**: FFmpeg for format conversion, 16kHz mono
- **Chunk Size**: 30 seconds (Whisper's native segment length)

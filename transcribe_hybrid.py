#!/usr/bin/env python3
"""
Whisper Hybrid Transcription for Snapdragon X Elite
Uses NPU for encoder (via ONNX Runtime QNN) and CPU for decoder (with KV-cache)

This hybrid approach gives the best performance:
- NPU encoder: Fast parallel computation of mel spectrogram -> hidden states
- CPU decoder: Uses optimized KV-cache for autoregressive decoding

Features:
- Multilingual support (auto-detect or specify language)
- Audio preprocessing with noise reduction
- Hallucination detection and quality scoring
- Preset configurations for common scenarios
- VAD (Voice Activity Detection) for silence removal
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import timedelta

import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Import audio preprocessing
try:
    from audio_preprocessing import preprocess_audio, get_available_methods
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

# Import quality checker
try:
    from quality_check import check_transcription_quality, print_quality_report
    QUALITY_CHECK_AVAILABLE = True
except ImportError:
    QUALITY_CHECK_AVAILABLE = False

# Import presets
try:
    from presets import get_preset, list_presets, PRESETS
    PRESETS_AVAILABLE = True
except ImportError:
    PRESETS_AVAILABLE = False

# Constants
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30  # seconds

# FFmpeg path cache
_FFMPEG_CMD = None

def get_ffmpeg_cmd() -> str:
    """Get the FFmpeg command path."""
    global _FFMPEG_CMD
    if _FFMPEG_CMD is not None:
        return _FFMPEG_CMD
    
    ffmpeg_paths = [
        r"C:\Users\jimla\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg"  # Fallback to PATH
    ]
    
    for path in ffmpeg_paths:
        if path == "ffmpeg" or os.path.exists(path):
            _FFMPEG_CMD = path
            return _FFMPEG_CMD
    
    _FFMPEG_CMD = "ffmpeg"
    return _FFMPEG_CMD


def apply_vad(file_path: str, output_path: str = None, 
              silence_threshold: float = -35, min_silence_duration: float = 0.5) -> str:
    """
    Apply Voice Activity Detection to remove silence from audio.
    
    Args:
        file_path: Input audio file path
        output_path: Output file path (auto-generated if None)
        silence_threshold: Silence threshold in dB (default: -35)
        min_silence_duration: Minimum silence duration to remove in seconds
        
    Returns:
        Path to the processed audio file
    """
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='_vad.wav', delete=False).name
    
    ffmpeg_cmd = get_ffmpeg_cmd()
    
    # Use silenceremove filter to remove silence
    # stop_periods=-1 means remove all silence periods
    # stop_duration is minimum silence duration to remove
    # stop_threshold is the noise floor
    cmd = [
        ffmpeg_cmd, '-y', '-i', file_path,
        '-af', f'silenceremove=stop_periods=-1:stop_duration={min_silence_duration}:stop_threshold={silence_threshold}dB',
        '-ar', str(SAMPLE_RATE), '-ac', '1',
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        # If VAD fails, return original file
        return file_path


def load_audio_ffmpeg(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio using FFmpeg, returns mono float32 numpy array."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    
    cmd = [
        ffmpeg_cmd,
        "-i", file_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sr),
        "-ac", "1",
        "-"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise


class WhisperHybrid:
    """
    Hybrid Whisper model: NPU encoder + CPU decoder with KV-cache.
    
    This approach is optimal because:
    1. Encoder is compute-bound and parallelizable -> NPU excels
    2. Decoder is memory-bound and sequential -> CPU with KV-cache is faster
    """
    
    def __init__(self, model_name: str = "openai/whisper-tiny", use_npu_encoder: bool = True):
        """
        Initialize hybrid Whisper model.
        
        Args:
            model_name: HuggingFace model name (e.g., "openai/whisper-tiny")
            use_npu_encoder: Whether to use NPU for encoder
        """
        self.model_name = model_name
        self.use_npu_encoder = use_npu_encoder
        
        # Determine model size from name
        if "large-v3-turbo" in model_name:
            self.model_size = "large-v3-turbo"
        elif "large-v3" in model_name:
            self.model_size = "large-v3"
        elif "large" in model_name:
            self.model_size = "large"
        elif "medium" in model_name:
            self.model_size = "medium"
        elif "small" in model_name:
            self.model_size = "small"
        elif "base" in model_name:
            self.model_size = "base"
        elif "tiny" in model_name:
            self.model_size = "tiny"
        else:
            self.model_size = "tiny"
        
        # Load processor
        print("⏳ Loading Whisper processor...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Load full model for decoder (uses KV-cache)
        print("⏳ Loading Whisper model...")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        
        # Set up NPU encoder if available
        self.encoder_session = None
        if use_npu_encoder:
            self._setup_npu_encoder()
        
        # Get device info
        self.device = "cpu"  # ARM64 Windows doesn't have CUDA
        print(f"✅ Model loaded (Encoder: {'NPU' if self.encoder_session else 'CPU'}, Decoder: CPU)")
    
    def _setup_npu_encoder(self):
        """Set up ONNX encoder for NPU inference."""
        # Map model size to ONNX directory
        onnx_dirs = {
            "tiny": Path("whisper-tiny-onnx/onnx"),
            "base": Path("whisper-base-onnx/onnx"),
            "small": Path("whisper-small-onnx/onnx"),
            "medium": Path("whisper-medium-onnx/onnx"),
            "large": Path("whisper-large-onnx/onnx"),
            "large-v3": Path("whisper-large-v3-onnx/onnx"),
            "large-v3-turbo": Path("whisper-large-v3-turbo-onnx/onnx"),
        }
        
        onnx_dir = onnx_dirs.get(self.model_size, Path(f"whisper-{self.model_size}-onnx/onnx"))
        
        if not onnx_dir.exists():
            print(f"⚠️ ONNX models not found for {self.model_size}. Using CPU encoder.")
            print(f"   To use NPU encoder, download from: huggingface.co/onnx-community/whisper-{self.model_size}")
            print(f"   Expected directory: {onnx_dir}")
            return
        
        # Try quantized encoder first (best for NPU), then full precision
        encoder_paths = [
            onnx_dir / "encoder_model_quantized.onnx",
            onnx_dir / "encoder_model_uint8.onnx",
            onnx_dir / "encoder_model_int8.onnx",
            onnx_dir / "encoder_model_q4.onnx",
            onnx_dir / "encoder_model.onnx"
        ]
        
        encoder_path = None
        for path in encoder_paths:
            if path.exists():
                encoder_path = path
                break
        
        if encoder_path is None:
            print("⚠️ No encoder ONNX model found. Using CPU encoder.")
            return
        
        # Create NPU session
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            provider_options = [{
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1"
            }]
            
            print(f"⏳ Loading NPU encoder from {encoder_path.name}...")
            self.encoder_session = ort.InferenceSession(
                str(encoder_path),
                sess_options=sess_options,
                providers=['QNNExecutionProvider'],
                provider_options=provider_options
            )
            self.encoder_input_name = self.encoder_session.get_inputs()[0].name
            print("✅ NPU encoder ready")
            
        except Exception as e:
            print(f"⚠️ Failed to load NPU encoder: {e}")
            print("   Falling back to CPU encoder.")
            self.encoder_session = None
    
    def _encode_npu(self, input_features: np.ndarray) -> torch.Tensor:
        """Encode audio features using NPU."""
        encoder_output = self.encoder_session.run(
            None,
            {self.encoder_input_name: input_features}
        )[0]
        return torch.from_numpy(encoder_output)
    
    def _encode_cpu(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode audio features using CPU."""
        with torch.no_grad():
            encoder_output = self.model.model.encoder(input_features)
        return encoder_output.last_hidden_state
    
    def transcribe(self, audio: np.ndarray, language: str = "auto") -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (float32, 16kHz, mono)
            language: Language code (e.g., "en", "es", "auto" for auto-detect)
            
        Returns:
            Transcribed text
        """
        # Extract features
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        input_features = inputs.input_features
        
        # Encode and decode
        if self.encoder_session is not None:
            # NPU encoding - need to pass encoder_outputs to decoder
            input_np = input_features.numpy().astype(np.float32)
            encoder_output = self._encode_npu(input_np)  # Returns torch.Tensor
            
            # Create proper encoder outputs object
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_output)
            
            # Decode with KV-cache (CPU) - pass encoder_outputs from NPU
            with torch.no_grad():
                # For auto-detect with NPU encoder, we need to detect language first
                # using the encoder outputs, then generate with explicit language
                if language == "auto":
                    # Detect language using the encoder outputs
                    # Use the model's detect_language method with encoder_outputs only
                    detected_lang_ids = self.model.detect_language(encoder_outputs=encoder_outputs)
                    # Get the detected language token id (first one in batch)
                    detected_lang_id = detected_lang_ids[0].item()
                    # Map token id back to language code
                    lang_token = self.processor.tokenizer.decode([detected_lang_id])
                    # Extract language code from token like "<|en|>" -> "en"
                    detected_language = lang_token.strip("<|>")
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=detected_language,
                        task="transcribe"
                    )
                else:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language,
                        task="transcribe"
                    )
                
                # Generate tokens using model's generate method (uses KV-cache internally)
                # When we have encoder_outputs, don't pass input_features
                # Use parameters to suppress hallucinations
                generated_ids = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=440,  # Leave room for prompt tokens
                    do_sample=False,
                    num_beams=1,  # Greedy decoding (faster)
                    repetition_penalty=1.2,  # Penalize repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )
        else:
            # CPU-only path: let the model handle encoder+decoder internally
            with torch.no_grad():
                # Set forced decoder IDs for language (None for auto-detect)
                if language == "auto":
                    forced_decoder_ids = None  # Let model auto-detect language
                else:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language,
                        task="transcribe"
                    )
                
                # Generate tokens - model handles encoder internally
                # Use parameters to suppress hallucinations
                generated_ids = self.model.generate(
                    input_features=input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=440,
                    do_sample=False,
                    num_beams=1,  # Greedy decoding (faster)
                    repetition_penalty=1.2,  # Penalize repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )
        
        # Decode tokens to text
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    
    def transcribe_chunked(self, audio: np.ndarray, chunk_length_s: int = CHUNK_LENGTH, 
                          language: str = "en") -> list:
        """
        Transcribe long audio by chunking.
        
        Args:
            audio: Audio array (float32, 16kHz, mono)
            chunk_length_s: Chunk length in seconds
            language: Language code
            
        Returns:
            List of (start_time, end_time, text) tuples
        """
        chunk_samples = chunk_length_s * SAMPLE_RATE
        total_samples = len(audio)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        
        results = []
        
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, total_samples)
            chunk = audio[start_sample:end_sample]
            
            # Pad to chunk_length_s if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            # Transcribe chunk
            text = self.transcribe(chunk, language=language)
            
            if text:  # Only add non-empty results
                start_time = start_sample / SAMPLE_RATE
                end_time = end_sample / SAMPLE_RATE
                results.append((start_time, end_time, text))
            
            # Progress indicator
            print(f"done")
        
        return results


def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp."""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() - int(td.total_seconds())) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list, output_path: Path):
    """Write segments to SRT file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (start, end, text) in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def write_txt(segments: list, output_path: Path):
    """Write segments to plain text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, _, text in segments:
            f.write(f"{text}\n")


def transcribe_file(model: WhisperHybrid, file_path: Path, output_dir: Path = None, 
                   language: str = "en", preprocess: str = "none", noise_reduction: float = 15,
                   use_vad: bool = False, check_quality: bool = True):
    """Transcribe a single audio file."""
    print(f"\n🎤 Transcribing: {file_path.name}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = file_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio preprocessing if requested
    audio_path = str(file_path)
    temp_files = []  # Track all temp files for cleanup
    
    # Apply VAD if requested
    if use_vad:
        print("   🔇 Applying VAD (removing silence)...", end=" ")
        try:
            vad_temp = tempfile.NamedTemporaryFile(suffix='_vad.wav', delete=False)
            vad_temp.close()
            temp_files.append(vad_temp.name)
            
            vad_path = apply_vad(audio_path, vad_temp.name)
            if vad_path != audio_path:
                audio_path = vad_path
                print("done")
            else:
                print("skipped (VAD failed)")
        except Exception as e:
            print(f"failed ({e})")
    
    if preprocess != "none" and PREPROCESSING_AVAILABLE:
        print("   🔊 Preprocessing audio...", end=" ")
        try:
            # Create temp file for preprocessed audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            preprocess_audio(
                audio_path, 
                temp_file.name, 
                method=preprocess,
                noise_reduction_db=noise_reduction
            )
            audio_path = temp_file.name
            print("done")
        except Exception as e:
            print(f"failed ({e}), using original audio")
    elif preprocess != "none" and not PREPROCESSING_AVAILABLE:
        print("   ⚠️ Preprocessing not available (audio_preprocessing.py not found)")
    
    # Load audio
    print("   Loading audio...", end=" ")
    start_load = time.time()
    
    try:
        audio = load_audio_ffmpeg(audio_path)
    except Exception as e:
        print(f"\n❌ Failed to load audio: {e}")
        # Clean up all temp files
        for tf in temp_files:
            if os.path.exists(tf):
                os.unlink(tf)
        return
    
    # Clean up temp files
    for tf in temp_files:
        if os.path.exists(tf):
            os.unlink(tf)
    
    duration = len(audio) / SAMPLE_RATE
    load_time = time.time() - start_load
    print(f"done ({duration:.1f}s audio, loaded in {load_time:.1f}s)")
    
    # Transcribe
    print("   Transcribing...")
    start_transcribe = time.time()
    
    num_chunks = int(np.ceil(duration / CHUNK_LENGTH))
    for i in range(num_chunks):
        print(f"   Chunk {i+1}/{num_chunks}...", end=" ")
    
    segments = model.transcribe_chunked(audio, language=language)
    
    transcribe_time = time.time() - start_transcribe
    realtime_factor = duration / transcribe_time if transcribe_time > 0 else float('inf')
    
    print(f"✅ Done in {transcribe_time:.1f}s ({realtime_factor:.1f}x realtime)")
    
    # Get full text for quality check
    full_text = " ".join([text for _, _, text in segments])
    
    # Quality check
    if check_quality and QUALITY_CHECK_AVAILABLE:
        quality = check_transcription_quality(full_text)
        if quality['is_hallucination']:
            print(f"   ⚠️ Quality warning: Possible hallucination detected (score: {quality['quality_score']}/100)")
            for issue in quality['issues'][:3]:  # Show top 3 issues
                print(f"      - {issue}")
        elif quality['quality_score'] < 50:
            print(f"   ⚠️ Quality warning: Low quality score ({quality['quality_score']}/100)")
        else:
            print(f"   📊 Quality score: {quality['quality_score']}/100")
    
    # Save outputs
    base_name = file_path.stem
    
    # Save SRT
    srt_path = output_dir / f"{base_name}.srt"
    write_srt(segments, srt_path)
    print(f"   📄 {srt_path.name}")
    
    # Save plain text
    txt_path = output_dir / f"{base_name}.txt"
    write_txt(segments, txt_path)
    print(f"   📄 {txt_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Whisper Hybrid Transcription (NPU Encoder + CPU Decoder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python transcribe_hybrid.py audio.m4a
  
  # Use preset for bilingual meetings
  python transcribe_hybrid.py audio.m4a --preset meeting-bilingual
  
  # Custom configuration
  python transcribe_hybrid.py audio.m4a --model large-v3-turbo --preprocess ffmpeg --vad
  
  # List available presets
  python transcribe_hybrid.py --list-presets
"""
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Audio file or directory to transcribe"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for transcriptions"
    )
    parser.add_argument(
        "--model", "-m",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large-v3-turbo"],
        help="Whisper model size (default: tiny). large-v3-turbo is optimized for speed."
    )
    parser.add_argument(
        "--language", "-l",
        default="auto",
        help="Language code: 'auto' for auto-detect, 'en' for English, 'es' for Spanish, etc. (default: auto)"
    )
    parser.add_argument(
        "--no-npu",
        action="store_true",
        help="Disable NPU encoder (use CPU only)"
    )
    parser.add_argument(
        "--preprocess", "-p",
        choices=["none", "ffmpeg", "noisereduce", "deepfilternet", "auto"],
        default="none",
        help="Audio preprocessing method for noise reduction (default: none)"
    )
    parser.add_argument(
        "--noise-reduction",
        type=float,
        default=15,
        help="Noise reduction strength in dB for FFmpeg method (default: 15)"
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Apply Voice Activity Detection to remove silence before transcription"
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Disable quality checking (hallucination detection)"
    )
    parser.add_argument(
        "--preset",
        help="Use a preset configuration (e.g., meeting-bilingual, phone-call, best-quality)"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit"
    )
    
    args = parser.parse_args()
    
    # Handle list-presets
    if args.list_presets:
        if PRESETS_AVAILABLE:
            list_presets()
        else:
            print("❌ Presets not available (presets.py not found)")
        sys.exit(0)
    
    # Check for required input
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    # Apply preset if specified
    if args.preset:
        if not PRESETS_AVAILABLE:
            print(f"❌ Presets not available (presets.py not found)")
            sys.exit(1)
        try:
            preset = get_preset(args.preset)
            print(f"📋 Using preset: {args.preset}")
            print(f"   {preset['description']}")
            # Apply preset values (command-line args override preset)
            if args.model == "tiny":  # Default value, use preset
                args.model = preset['model']
            if args.language == "auto":  # Default value, use preset
                args.language = preset['language']
            if args.preprocess == "none":  # Default value, use preset
                args.preprocess = preset['preprocess']
            if args.noise_reduction == 15:  # Default value, use preset
                args.noise_reduction = preset['noise_reduction']
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    
    # Print header
    print("=" * 60)
    print("🎙️  Whisper Hybrid Transcription")
    print("   NPU Encoder + CPU Decoder (with KV-cache)")
    print("=" * 60)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {args.input}")
        sys.exit(1)
    
    # Initialize model
    if args.model == "large-v3-turbo":
        model_name = "openai/whisper-large-v3-turbo"
    else:
        model_name = f"openai/whisper-{args.model}"
    model = WhisperHybrid(model_name, use_npu_encoder=not args.no_npu)
    
    # Process files
    check_quality = not args.no_quality_check
    
    if input_path.is_file():
        transcribe_file(model, input_path, args.output_dir, args.language, 
                       args.preprocess, args.noise_reduction, args.vad, check_quality)
    elif input_path.is_dir():
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma'}
        audio_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"❌ No audio files found in: {input_path}")
            sys.exit(1)
        
        print(f"📁 Found {len(audio_files)} audio files")
        if args.preprocess != "none":
            print(f"🔊 Audio preprocessing: {args.preprocess}")
        if args.vad:
            print(f"🔇 VAD enabled (silence removal)")
        for audio_file in sorted(audio_files):
            transcribe_file(model, audio_file, args.output_dir, args.language,
                          args.preprocess, args.noise_reduction, args.vad, check_quality)
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Whisper Transcription Tool for Windows ARM64
Using HuggingFace Transformers with FFmpeg audio loading

Usage:
    python transcribe_ffmpeg.py audio.mp3
    python transcribe_ffmpeg.py audio.mp3 --model medium --language en
    python transcribe_ffmpeg.py folder/ --batch
"""

import argparse
import os
import sys
import time
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def load_audio_ffmpeg(audio_path: str, sr: int = 16000):
    """Load audio using ffmpeg and return numpy array."""
    import numpy as np
    
    # Use ffmpeg to convert to raw PCM
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "s16le",  # signed 16-bit little-endian
        "-acodec", "pcm_s16le",
        "-ar", str(sr),  # sample rate
        "-ac", "1",  # mono
        "-loglevel", "error",
        "-"  # output to stdout
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        audio_data = np.frombuffer(result.stdout, dtype=np.int16)
        # Normalize to float32 in range [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        return audio_float
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg and add it to PATH.")
        raise

def transcribe_file(audio_path: str, model, processor, output_dir: str = None, output_formats: list = None, language: str = None):
    """Transcribe a single audio file."""
    import torch
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        return None
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = audio_path.parent
    
    if output_formats is None:
        output_formats = ["txt", "srt"]
    
    base_name = audio_path.stem
    
    print(f"\n🎤 Transcribing: {audio_path.name}")
    start_time = time.time()
    
    # Load audio using ffmpeg
    print("   Loading audio...")
    audio = load_audio_ffmpeg(str(audio_path), sr=16000)
    audio_duration = len(audio) / 16000
    
    # Process audio
    print("   Processing...")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    # Generate transcription
    print("   Transcribing...")
    
    # For multilingual audio, we need to process in chunks and let the model detect language
    # Whisper can only process 30 seconds at a time
    chunk_length = 30 * 16000  # 30 seconds at 16kHz
    all_text = []
    all_segments = []
    
    # Process audio in chunks for long files
    num_chunks = max(1, len(audio) // chunk_length + (1 if len(audio) % chunk_length else 0))
    
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_length
        end_sample = min(start_sample + chunk_length, len(audio))
        chunk_audio = audio[start_sample:end_sample]
        
        # Skip very short chunks
        if len(chunk_audio) < 1600:  # Less than 0.1 second
            continue
        
        # Pad short chunks to minimum length
        if len(chunk_audio) < 16000:  # Less than 1 second
            chunk_audio = np.pad(chunk_audio, (0, 16000 - len(chunk_audio)))
        
        chunk_inputs = processor(chunk_audio, sampling_rate=16000, return_tensors="pt")
        
        # Generate with auto language detection for multilingual support
        generate_kwargs = {
            "task": "transcribe",
            "return_timestamps": True,
        }
        
        # If language specified, use it; otherwise let model auto-detect
        if language:
            generate_kwargs["language"] = language
        # Don't force language - let Whisper detect per chunk for mixed language audio
        
        with torch.no_grad():
            predicted_ids = model.generate(
                chunk_inputs.input_features,
                **generate_kwargs
            )
        
        # Decode this chunk
        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if chunk_text and chunk_text[0].strip():
            all_text.append(chunk_text[0].strip())
            
            # Calculate timestamp for this segment
            chunk_start = start_sample / 16000
            chunk_end = end_sample / 16000
            all_segments.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": chunk_text[0].strip()
            })
        
        if num_chunks > 1:
            print(f"   Chunk {chunk_idx + 1}/{num_chunks} done")
    
    text = " ".join(all_text)
    
    elapsed = time.time() - start_time
    speed = audio_duration / elapsed if elapsed > 0 else 0
    print(f"✅ Done in {elapsed:.1f}s ({speed:.1f}x realtime)")
    
    # Save outputs
    outputs = {}
    
    # Plain text
    if "txt" in output_formats:
        txt_path = output_dir / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        outputs["txt"] = txt_path
        print(f"   📄 {txt_path.name}")
    
    # SRT with proper segments
    if "srt" in output_formats:
        srt_path = output_dir / f"{base_name}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(all_segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                f.write(f"{segment['text']}\n\n")
        outputs["srt"] = srt_path
        print(f"   📄 {srt_path.name}")
    
    return {"text": text, "duration": audio_duration, "segments": all_segments}

def batch_transcribe(folder_path: str, model, processor, output_dir: str = None, output_formats: list = None, language: str = None):
    """Transcribe all audio files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return
    
    # Supported audio extensions
    audio_extensions = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.wma', '.aac', '.mp4', '.webm'}
    
    # Find all audio files
    audio_files = [f for f in folder.iterdir() if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print(f"❌ No audio files found in {folder}")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio file(s) in {folder}")
    
    if output_dir is None:
        output_dir = folder / "transcripts"
    
    total_start = time.time()
    success_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}]", end="")
        try:
            result = transcribe_file(str(audio_file), model, processor, output_dir, output_formats, language)
            if result:
                success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"✅ Batch complete: {success_count}/{len(audio_files)} files")
    print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes")
    print(f"📂 Output: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Whisper Transcription Tool for Windows ARM64",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_ffmpeg.py recording.mp3
  python transcribe_ffmpeg.py recording.mp3 --model small --language en
  python transcribe_ffmpeg.py "C:\\Audio Files\\" --batch
  python transcribe_ffmpeg.py folder/ --batch --model medium --output ./transcripts
        """
    )
    
    parser.add_argument("input", help="Audio file or folder path")
    parser.add_argument("--model", "-m", default="small", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (e.g., 'en', 'es'). Auto-detect if not specified")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process all audio files in the input folder")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for transcripts")
    parser.add_argument("--format", "-f", default="txt,srt",
                        help="Output formats: txt,srt (comma-separated, default: txt,srt)")
    
    args = parser.parse_args()
    
    # Parse output formats
    output_formats = [fmt.strip().lower() for fmt in args.format.split(",")]
    
    print("=" * 50)
    print("🎙️  Whisper Transcription Tool")
    print("   Windows ARM64 / Snapdragon X Elite")
    print("=" * 50)
    print(f"Model: openai/whisper-{args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Output formats: {', '.join(output_formats)}")
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("\n❌ FFmpeg not found! Please restart your terminal to pick up the new PATH.")
        print("   Or manually add FFmpeg to your PATH.")
        sys.exit(1)
    
    # Load model
    print(f"\n⏳ Loading whisper-{args.model} model...")
    
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    start_load = time.time()
    
    model_name = f"openai/whisper-{args.model}"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    
    print(f"✅ Model loaded in {time.time() - start_load:.1f}s")
    
    # Process
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        batch_transcribe(str(input_path), model, processor, args.output, output_formats, args.language)
    else:
        transcribe_file(str(input_path), model, processor, args.output, output_formats, args.language)
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()

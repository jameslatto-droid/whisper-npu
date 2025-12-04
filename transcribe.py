#!/usr/bin/env python3
"""
Whisper Transcription Tool for Windows ARM64
Optimized for Snapdragon X Elite

Usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --model medium --language en
    python transcribe.py folder/ --batch
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe_file(audio_path: str, model, output_dir: str = None, output_formats: list = None):
    """Transcribe a single audio file."""
    import whisper
    
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
    
    # Transcribe
    result = model.transcribe(
        str(audio_path),
        fp16=False,  # ARM64 doesn't support FP16 on CPU
        verbose=False
    )
    
    elapsed = time.time() - start_time
    
    # Get audio duration from segments
    if result["segments"]:
        audio_duration = result["segments"][-1]["end"]
        speed_ratio = audio_duration / elapsed if elapsed > 0 else 0
        print(f"✅ Done in {elapsed:.1f}s ({speed_ratio:.1f}x realtime)")
    else:
        print(f"✅ Done in {elapsed:.1f}s")
    
    # Save outputs
    outputs = {}
    
    # Plain text
    if "txt" in output_formats:
        txt_path = output_dir / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"].strip())
        outputs["txt"] = txt_path
        print(f"   📄 {txt_path.name}")
    
    # SRT subtitles
    if "srt" in output_formats:
        srt_path = output_dir / f"{base_name}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        outputs["srt"] = srt_path
        print(f"   📄 {srt_path.name}")
    
    # VTT subtitles
    if "vtt" in output_formats:
        vtt_path = output_dir / f"{base_name}.vtt"
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in result["segments"]:
                start = format_timestamp(segment['start']).replace(',', '.')
                end = format_timestamp(segment['end']).replace(',', '.')
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        outputs["vtt"] = vtt_path
        print(f"   📄 {vtt_path.name}")
    
    # JSON with full details
    if "json" in output_formats:
        import json
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        outputs["json"] = json_path
        print(f"   📄 {json_path.name}")
    
    return result

def batch_transcribe(folder_path: str, model, output_dir: str = None, output_formats: list = None):
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
            result = transcribe_file(str(audio_file), model, output_dir, output_formats)
            if result:
                success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
    
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
  python transcribe.py recording.mp3
  python transcribe.py recording.mp3 --model small --language en
  python transcribe.py "C:\\Audio Files\\" --batch
  python transcribe.py folder/ --batch --model medium --output ./transcripts
        """
    )
    
    parser.add_argument("input", help="Audio file or folder path")
    parser.add_argument("--model", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (e.g., 'en', 'es'). Auto-detect if not specified")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process all audio files in the input folder")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for transcripts")
    parser.add_argument("--format", "-f", default="txt,srt",
                        help="Output formats: txt,srt,vtt,json (comma-separated, default: txt,srt)")
    parser.add_argument("--threads", "-t", type=int, default=None,
                        help="Number of CPU threads to use")
    
    args = parser.parse_args()
    
    # Parse output formats
    output_formats = [fmt.strip().lower() for fmt in args.format.split(",")]
    
    # Set thread count for optimal ARM64 performance
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    
    print("=" * 50)
    print("🎙️  Whisper Transcription Tool")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Output formats: {', '.join(output_formats)}")
    
    # Load model
    print(f"\n⏳ Loading {args.model} model...")
    import whisper
    
    start_load = time.time()
    model = whisper.load_model(args.model)
    print(f"✅ Model loaded in {time.time() - start_load:.1f}s")
    
    # Process
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        batch_transcribe(str(input_path), model, args.output, output_formats)
    else:
        transcribe_file(str(input_path), model, args.output, output_formats)
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()

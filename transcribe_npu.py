#!/usr/bin/env python3
"""
Whisper NPU Transcription Script for Snapdragon X Elite
Uses ONNX Runtime with QNN Execution Provider for NPU acceleration.
"""

import argparse
import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path

# ONNX Runtime with QNN
import onnxruntime as ort

# For tokenizer and feature extraction
from transformers import WhisperProcessor

# Constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds


def load_audio_ffmpeg(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio using FFmpeg, returns mono float32 numpy array."""
    # Use full path to ffmpeg on Windows
    ffmpeg_paths = [
        r"C:\Users\jimla\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg"  # Fallback to PATH
    ]
    
    ffmpeg_cmd = "ffmpeg"
    for path in ffmpeg_paths:
        if os.path.exists(path):
            ffmpeg_cmd = path
            break
    
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


class WhisperNPU:
    """Whisper model running on Snapdragon NPU via ONNX Runtime QNN."""
    
    def __init__(self, model_dir: str, use_npu: bool = True, model_variant: str = "quantized"):
        """
        Initialize Whisper NPU model.
        
        Args:
            model_dir: Path to ONNX model directory (e.g., whisper-tiny-onnx/onnx)
            use_npu: Whether to use NPU (HTP) or CPU backend
            model_variant: Model variant to use (quantized, uint8, fp32)
        """
        self.model_dir = Path(model_dir)
        self.use_npu = use_npu
        self.model_variant = model_variant
        
        # Load processor for tokenization and feature extraction
        print("⏳ Loading Whisper processor...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        
        # Get model paths based on variant
        self.encoder_path, self.decoder_path = self._get_model_paths()
        
        # Create ONNX sessions
        self._create_sessions()
        
        # Special tokens
        self.sot_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot_token = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.transcribe_token = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_token = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.en_token = self.tokenizer.convert_tokens_to_ids("<|en|>")
        
        print(f"✅ Model loaded ({'NPU' if use_npu else 'CPU'} backend)")
    
    def _get_model_paths(self):
        """Get encoder and decoder model paths based on variant."""
        # Use simple decoder (not merged with KV-cache) for easier inference
        variant_map = {
            "quantized": ("encoder_model_quantized.onnx", "decoder_model_quantized.onnx"),
            "uint8": ("encoder_model_uint8.onnx", "decoder_model_uint8.onnx"),
            "int8": ("encoder_model_quantized.onnx", "decoder_model_int8.onnx"),
            "fp32": ("encoder_model.onnx", "decoder_model.onnx"),
        }
        
        if self.model_variant not in variant_map:
            print(f"⚠️ Unknown variant '{self.model_variant}', using quantized")
            self.model_variant = "quantized"
        
        encoder_name, decoder_name = variant_map[self.model_variant]
        encoder_path = self.model_dir / encoder_name
        decoder_path = self.model_dir / decoder_name
        
        # Fallback if files don't exist
        if not encoder_path.exists():
            print(f"⚠️ {encoder_name} not found, trying encoder_model.onnx")
            encoder_path = self.model_dir / "encoder_model.onnx"
        
        if not decoder_path.exists():
            print(f"⚠️ {decoder_name} not found, trying decoder_model.onnx")
            decoder_path = self.model_dir / "decoder_model.onnx"
        
        return encoder_path, decoder_path
    
    def _create_sessions(self):
        """Create ONNX Runtime sessions for encoder and decoder."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if self.use_npu:
            provider_options = [{
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1"
            }]
            providers = ['QNNExecutionProvider']
        else:
            provider_options = [{"backend_path": "QnnCpu.dll"}]
            providers = ['QNNExecutionProvider']
        
        print(f"⏳ Loading encoder from {self.encoder_path.name}...")
        try:
            self.encoder_session = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
        except Exception as e:
            print(f"⚠️ Failed to load encoder on {'NPU' if self.use_npu else 'CPU'}: {e}")
            print("   Falling back to CPU execution provider...")
            self.encoder_session = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
        
        print(f"⏳ Loading decoder from {self.decoder_path.name}...")
        try:
            self.decoder_session = ort.InferenceSession(
                str(self.decoder_path),
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
        except Exception as e:
            print(f"⚠️ Failed to load decoder on {'NPU' if self.use_npu else 'CPU'}: {e}")
            print("   Falling back to CPU execution provider...")
            self.decoder_session = ort.InferenceSession(
                str(self.decoder_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
        
        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_inputs = {inp.name: inp for inp in self.decoder_session.get_inputs()}
        self.decoder_outputs = [out.name for out in self.decoder_session.get_outputs()]
    
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode audio to hidden states using the encoder.
        
        Args:
            audio: Audio array (float32, 16kHz, mono)
            
        Returns:
            Encoder hidden states
        """
        # Extract mel spectrogram features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np"
        )
        input_features = inputs.input_features.astype(np.float32)
        
        # Run encoder
        encoder_output = self.encoder_session.run(
            None,
            {self.encoder_input_name: input_features}
        )[0]
        
        return encoder_output
    
    def decode_greedy(self, encoder_hidden_states: np.ndarray, max_length: int = 448) -> list:
        """
        Greedy decoding of encoder hidden states.
        
        Args:
            encoder_hidden_states: Output from encoder
            max_length: Maximum number of tokens to generate
            
        Returns:
            List of token IDs
        """
        # Initial decoder input: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        decoder_input_ids = np.array([[
            self.sot_token,
            self.en_token,
            self.transcribe_token,
            self.notimestamps_token
        ]], dtype=np.int64)
        
        generated_tokens = list(decoder_input_ids[0])
        
        for _ in range(max_length):
            # Prepare decoder inputs
            decoder_inputs = {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
            }
            
            # Check if model expects attention mask
            if "encoder_attention_mask" in self.decoder_inputs:
                # Create attention mask (all ones for full attention)
                seq_len = encoder_hidden_states.shape[1]
                encoder_attention_mask = np.ones((1, seq_len), dtype=np.int64)
                decoder_inputs["encoder_attention_mask"] = encoder_attention_mask
            
            # Run decoder
            try:
                outputs = self.decoder_session.run(None, decoder_inputs)
                logits = outputs[0]
            except Exception as e:
                print(f"⚠️ Decoder error: {e}")
                break
            
            # Get next token (greedy: take argmax of last position)
            next_token_logits = logits[0, -1, :]
            next_token = int(np.argmax(next_token_logits))
            
            # Check for end of transcript
            if next_token == self.eot_token:
                break
            
            generated_tokens.append(next_token)
            
            # Update decoder input for next iteration
            decoder_input_ids = np.array([generated_tokens], dtype=np.int64)
        
        return generated_tokens
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (float32, 16kHz, mono)
            
        Returns:
            Transcribed text
        """
        # Encode
        encoder_output = self.encode(audio)
        
        # Decode
        tokens = self.decode_greedy(encoder_output)
        
        # Convert tokens to text
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text.strip()
    
    def transcribe_chunked(self, audio: np.ndarray, chunk_length_s: int = CHUNK_LENGTH) -> str:
        """
        Transcribe long audio by chunking.
        
        Args:
            audio: Audio array (float32, 16kHz, mono)
            chunk_length_s: Chunk length in seconds
            
        Returns:
            Full transcription
        """
        chunk_samples = chunk_length_s * SAMPLE_RATE
        total_samples = len(audio)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        
        all_text = []
        
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, total_samples)
            chunk = audio[start:end]
            
            # Pad if necessary (Whisper expects 30s chunks)
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            print(f"   Chunk {i+1}/{num_chunks}...", end=" ", flush=True)
            text = self.transcribe(chunk)
            print("done")
            
            if text:
                all_text.append(text)
        
        return " ".join(all_text)


def transcribe_file(model: WhisperNPU, file_path: str, output_dir: str = None) -> str:
    """Transcribe a single audio file."""
    file_path = Path(file_path)
    
    if output_dir:
        output_path = Path(output_dir) / f"{file_path.stem}.txt"
    else:
        output_path = file_path.with_suffix(".txt")
    
    print(f"\n🎤 Transcribing: {file_path.name}")
    
    # Load audio
    print("   Loading audio...")
    start_time = time.time()
    audio = load_audio_ffmpeg(str(file_path))
    load_time = time.time() - start_time
    
    duration = len(audio) / SAMPLE_RATE
    print(f"   Audio duration: {duration:.1f}s (loaded in {load_time:.1f}s)")
    
    # Transcribe
    print("   Transcribing...")
    start_time = time.time()
    
    if duration > CHUNK_LENGTH:
        text = model.transcribe_chunked(audio)
    else:
        text = model.transcribe(audio)
    
    transcribe_time = time.time() - start_time
    rtf = transcribe_time / duration if duration > 0 else 0
    
    print(f"✅ Done in {transcribe_time:.1f}s ({1/rtf:.1f}x realtime)")
    
    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   📄 {output_path.name}")
    
    return text


def main():
    parser = argparse.ArgumentParser(description="Whisper NPU Transcription")
    parser.add_argument("input", help="Audio file or directory to transcribe")
    parser.add_argument("--model-dir", default=r"C:\Users\jimla\Projects\whisper-npu\whisper-tiny-onnx\onnx",
                        help="Path to ONNX model directory")
    parser.add_argument("--variant", default="quantized", choices=["quantized", "uint8", "int8", "fp32"],
                        help="Model variant to use")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of NPU")
    parser.add_argument("--output-dir", "-o", help="Output directory for transcriptions")
    parser.add_argument("--batch", action="store_true", help="Process all audio files in directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  Whisper NPU Transcription")
    print("   Snapdragon X Elite / QNN Execution Provider")
    print("=" * 60)
    
    # Initialize model
    model = WhisperNPU(
        model_dir=args.model_dir,
        use_npu=not args.cpu,
        model_variant=args.variant
    )
    
    input_path = Path(args.input)
    
    if args.batch and input_path.is_dir():
        # Batch process all audio files
        audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"}
        files = [f for f in input_path.iterdir() if f.suffix.lower() in audio_extensions]
        
        if not files:
            print(f"No audio files found in {input_path}")
            return
        
        print(f"\n📁 Found {len(files)} audio files")
        
        for file_path in sorted(files):
            try:
                transcribe_file(model, file_path, args.output_dir)
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    else:
        # Single file
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            return
        
        transcribe_file(model, input_path, args.output_dir)
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()

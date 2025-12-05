"""
Audio Preprocessing Module for Whisper Transcription
=====================================================

This module provides various audio preprocessing and noise reduction methods
to improve Whisper transcription quality.

Supported methods:
1. FFmpeg-based (afftdn, anlmdn, highpass/lowpass)
2. Python-based (noisereduce, scipy)
3. AI-based (DeepFilterNet - optional)

Author: Whisper NPU Project
Date: December 2024
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional, Literal, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FFmpeg-based Preprocessing (No additional Python dependencies)
# ============================================================================

def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ffmpeg_denoise_afftdn(
    input_file: str,
    output_file: str,
    noise_reduction_db: float = 12,
    noise_floor_db: float = -40,
    track_noise: bool = True
) -> str:
    """
    Apply FFT-based denoising using FFmpeg's afftdn filter.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        noise_reduction_db: Amount of noise reduction (0.01-97 dB)
        noise_floor_db: Estimated noise floor (-80 to -20 dB)
        track_noise: Enable automatic noise floor tracking
    
    Returns:
        Path to output file
    """
    tn = 1 if track_noise else 0
    filter_str = f'afftdn=nr={noise_reduction_db}:nf={noise_floor_db}:tn={tn}'
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', filter_str,
        '-ac', '1',  # Mono
        output_file
    ]
    
    logger.info(f"Applying FFT denoising: nr={noise_reduction_db}dB, nf={noise_floor_db}dB")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_file


def ffmpeg_denoise_anlmdn(
    input_file: str,
    output_file: str,
    strength: float = 0.0001,
    patch_ms: float = 4,
    research_ms: float = 10
) -> str:
    """
    Apply Non-Local Means denoising using FFmpeg's anlmdn filter.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        strength: Denoising strength (0.00001-10000)
        patch_ms: Patch radius in milliseconds
        research_ms: Research radius in milliseconds
    
    Returns:
        Path to output file
    """
    filter_str = f'anlmdn=s={strength}:p={patch_ms}:r={research_ms}'
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', filter_str,
        '-ac', '1',
        output_file
    ]
    
    logger.info(f"Applying Non-Local Means denoising: strength={strength}")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_file


def ffmpeg_speech_bandpass(
    input_file: str,
    output_file: str,
    highpass_hz: int = 80,
    lowpass_hz: int = 8000
) -> str:
    """
    Apply bandpass filter optimized for speech frequencies.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file  
        highpass_hz: High-pass cutoff frequency (removes low rumble)
        lowpass_hz: Low-pass cutoff frequency (removes high-freq noise)
    
    Returns:
        Path to output file
    """
    filter_str = f'highpass=f={highpass_hz},lowpass=f={lowpass_hz}'
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', filter_str,
        '-ac', '1',
        output_file
    ]
    
    logger.info(f"Applying bandpass filter: {highpass_hz}Hz - {lowpass_hz}Hz")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_file


def ffmpeg_normalize_speech(
    input_file: str,
    output_file: str,
    expansion: float = 3,
    raise_rate: float = 0.001
) -> str:
    """
    Normalize speech levels using FFmpeg's speechnorm filter.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        expansion: Maximum expansion factor (1.0-50.0)
        raise_rate: How fast expansion increases (0.0-1.0)
    
    Returns:
        Path to output file
    """
    filter_str = f'speechnorm=e={expansion}:r={raise_rate}:l=1'
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', filter_str,
        output_file
    ]
    
    logger.info(f"Applying speech normalization")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_file


def ffmpeg_full_pipeline(
    input_file: str,
    output_file: str,
    target_sr: int = 16000,
    noise_reduction_db: float = 12,
    noise_floor_db: float = -40,
    highpass_hz: int = 80,
    lowpass_hz: int = 8000,
    normalize: bool = True
) -> str:
    """
    Complete FFmpeg preprocessing pipeline for Whisper.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        target_sr: Target sample rate (16000 for Whisper)
        noise_reduction_db: Noise reduction amount
        noise_floor_db: Noise floor estimate
        highpass_hz: High-pass cutoff
        lowpass_hz: Low-pass cutoff
        normalize: Apply speech normalization
    
    Returns:
        Path to output file
    """
    filters = [
        f'highpass=f={highpass_hz}',
        f'lowpass=f={lowpass_hz}',
        f'afftdn=nr={noise_reduction_db}:nf={noise_floor_db}:tn=1',
    ]
    
    if normalize:
        filters.append('speechnorm=e=3:r=0.001:l=1')
    
    filters.append(f'aresample={target_sr}')
    
    filter_chain = ','.join(filters)
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', filter_chain,
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        output_file
    ]
    
    logger.info(f"Applying full FFmpeg pipeline to {input_file}")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_file


# ============================================================================
# Python-based Preprocessing (requires: noisereduce, librosa, scipy)
# ============================================================================

def check_noisereduce() -> bool:
    """Check if noisereduce is available."""
    try:
        import noisereduce
        return True
    except ImportError:
        return False


def check_librosa() -> bool:
    """Check if soundfile is available (replaces librosa dependency)."""
    try:
        import soundfile
        return True
    except ImportError:
        return False


def noisereduce_stationary(
    input_file: str,
    output_file: str,
    prop_decrease: float = 0.75,
    n_std_thresh: float = 1.5
) -> str:
    """
    Apply stationary noise reduction using noisereduce library.
    Best for constant background noise (fans, AC, etc.)
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        prop_decrease: Proportion of noise to remove (0.0-1.0)
        n_std_thresh: Number of std deviations for threshold
    
    Returns:
        Path to output file
    """
    import noisereduce as nr
    from scipy.io import wavfile
    
    # Load audio
    rate, data = wavfile.read(input_file)
    
    # Apply stationary noise reduction
    reduced = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=True,
        prop_decrease=prop_decrease,
        n_std_thresh_stationary=n_std_thresh
    )
    
    logger.info(f"Applied stationary noise reduction: prop_decrease={prop_decrease}")
    wavfile.write(output_file, rate, reduced)
    return output_file


def noisereduce_nonstationary(
    input_file: str,
    output_file: str,
    prop_decrease: float = 0.8,
    time_constant_s: float = 2.0,
    freq_mask_smooth_hz: int = 500,
    time_mask_smooth_ms: int = 50
) -> str:
    """
    Apply non-stationary noise reduction using noisereduce library.
    Best for varying noise (traffic, crowds, etc.)
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        prop_decrease: Proportion of noise to remove (0.0-1.0)
        time_constant_s: Time constant for noise estimation
        freq_mask_smooth_hz: Frequency smoothing width
        time_mask_smooth_ms: Time smoothing width
    
    Returns:
        Path to output file
    """
    import noisereduce as nr
    from scipy.io import wavfile
    
    # Load audio
    rate, data = wavfile.read(input_file)
    
    # Apply non-stationary noise reduction
    reduced = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=False,
        prop_decrease=prop_decrease,
        time_constant_s=time_constant_s,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms
    )
    
    logger.info(f"Applied non-stationary noise reduction: prop_decrease={prop_decrease}")
    wavfile.write(output_file, rate, reduced)
    return output_file


def librosa_preprocess(
    input_file: str,
    output_file: str,
    target_sr: int = 16000,
    trim_db: float = 30,
    normalize: bool = True
) -> str:
    """
    Preprocess audio using librosa for Whisper.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        target_sr: Target sample rate
        trim_db: Threshold for silence trimming
        normalize: Normalize audio levels
    
    Returns:
        Path to output file
    """
    import librosa
    import soundfile as sf
    
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    
    # Resample to target rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=trim_db)
    
    # Normalize
    if normalize:
        y_trimmed = librosa.util.normalize(y_trimmed)
    
    logger.info(f"Applied librosa preprocessing: sr={target_sr}, trimmed, normalized")
    sf.write(output_file, y_trimmed, target_sr)
    return output_file


def python_full_pipeline(
    input_file: str,
    output_file: str,
    target_sr: int = 16000,
    stationary: bool = False,
    prop_decrease: float = 0.8
) -> str:
    """
    Complete Python-based preprocessing pipeline.
    Uses soundfile and scipy (no librosa required).
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        target_sr: Target sample rate
        stationary: Use stationary vs non-stationary noise reduction
        prop_decrease: Noise reduction strength
    
    Returns:
        Path to output file
    """
    import noisereduce as nr
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample
    
    # Load audio using soundfile
    y, sr = sf.read(input_file)
    
    # Convert to mono if stereo
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    
    # Apply noise reduction
    y_denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
        time_constant_s=2.0
    )
    
    # Resample to target
    if sr != target_sr:
        num_samples = int(len(y_denoised) * target_sr / sr)
        y_denoised = resample(y_denoised, num_samples)
    
    # Normalize
    max_val = np.max(np.abs(y_denoised))
    if max_val > 0:
        y_normalized = y_denoised / max_val * 0.9
    else:
        y_normalized = y_denoised
    
    logger.info(f"Applied Python pipeline: denoised, resampled to {target_sr}, normalized")
    sf.write(output_file, y_normalized.astype(np.float32), target_sr)
    return output_file


# ============================================================================
# DeepFilterNet (optional, best quality)
# ============================================================================

def check_deepfilternet() -> bool:
    """Check if DeepFilterNet is available."""
    try:
        from df import enhance, init_df
        return True
    except ImportError:
        return False


def deepfilternet_denoise(
    input_file: str,
    output_file: str,
    target_sr: int = 16000
) -> str:
    """
    Apply DeepFilterNet for high-quality speech enhancement.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        target_sr: Target sample rate for output
    
    Returns:
        Path to output file
    """
    from df import enhance, init_df
    import torchaudio
    import torch
    
    # Initialize model
    model, df_state, _ = init_df()
    
    # Load audio
    audio, sr = torchaudio.load(input_file)
    
    # Resample to 48kHz if needed (DeepFilterNet expects 48kHz)
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        audio = resampler(audio)
    
    # Enhance
    with torch.no_grad():
        enhanced = enhance(model, df_state, audio)
    
    # Resample to target
    if target_sr != 48000:
        resampler = torchaudio.transforms.Resample(48000, target_sr)
        enhanced = resampler(enhanced)
    
    # Ensure mono
    if enhanced.shape[0] > 1:
        enhanced = enhanced.mean(dim=0, keepdim=True)
    
    logger.info(f"Applied DeepFilterNet enhancement")
    torchaudio.save(output_file, enhanced, target_sr)
    return output_file


# ============================================================================
# Unified Interface
# ============================================================================

def preprocess_audio(
    input_file: str,
    output_file: Optional[str] = None,
    method: Literal['ffmpeg', 'noisereduce', 'deepfilternet', 'auto'] = 'auto',
    target_sr: int = 16000,
    **kwargs
) -> str:
    """
    Unified audio preprocessing interface.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (auto-generated if None)
        method: Preprocessing method to use
        target_sr: Target sample rate
        **kwargs: Additional arguments passed to the specific method
    
    Returns:
        Path to preprocessed audio file
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_preprocessed.wav")
    
    # Auto-select best available method
    if method == 'auto':
        if check_deepfilternet():
            method = 'deepfilternet'
            logger.info("Auto-selected: DeepFilterNet (best quality)")
        elif check_noisereduce() and check_librosa():
            method = 'noisereduce'
            logger.info("Auto-selected: noisereduce (good quality)")
        elif check_ffmpeg():
            method = 'ffmpeg'
            logger.info("Auto-selected: FFmpeg (fastest)")
        else:
            raise RuntimeError("No preprocessing method available. Install FFmpeg or Python packages.")
    
    # Apply selected method
    if method == 'ffmpeg':
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
        # Filter kwargs for ffmpeg-specific params
        ffmpeg_kwargs = {k: v for k, v in kwargs.items() if k in ['noise_reduction_db', 'noise_floor_db', 'highpass_hz', 'lowpass_hz', 'normalize']}
        return ffmpeg_full_pipeline(input_file, output_file, target_sr=target_sr, **ffmpeg_kwargs)
    
    elif method == 'noisereduce':
        if not check_noisereduce() or not check_librosa():
            raise RuntimeError("noisereduce/librosa not found. Install with: pip install noisereduce librosa soundfile")
        # Filter kwargs for noisereduce-specific params
        nr_kwargs = {k: v for k, v in kwargs.items() if k in ['prop_decrease', 'stationary']}
        return python_full_pipeline(input_file, output_file, target_sr=target_sr, **nr_kwargs)
    
    elif method == 'deepfilternet':
        if not check_deepfilternet():
            raise RuntimeError("DeepFilterNet not found. Install with: pip install deepfilternet torch torchaudio")
        return deepfilternet_denoise(input_file, output_file, target_sr=target_sr)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_available_methods() -> list[str]:
    """Get list of available preprocessing methods."""
    methods = []
    
    if check_ffmpeg():
        methods.append('ffmpeg')
    if check_noisereduce() and check_librosa():
        methods.append('noisereduce')
    if check_deepfilternet():
        methods.append('deepfilternet')
    
    return methods


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Audio preprocessing for Whisper transcription',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select best method
  python audio_preprocessing.py input.wav output.wav
  
  # Use FFmpeg method
  python audio_preprocessing.py input.wav output.wav --method ffmpeg
  
  # Use noisereduce with custom settings
  python audio_preprocessing.py input.wav output.wav --method noisereduce --prop-decrease 0.9
  
  # Process all files in a directory
  python audio_preprocessing.py input_dir/ output_dir/ --method ffmpeg
        """
    )
    
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('output', nargs='?', help='Output audio file or directory')
    parser.add_argument('--method', choices=['ffmpeg', 'noisereduce', 'deepfilternet', 'auto'],
                       default='auto', help='Preprocessing method')
    parser.add_argument('--target-sr', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--noise-reduction', type=float, default=12,
                       help='Noise reduction in dB (FFmpeg method)')
    parser.add_argument('--prop-decrease', type=float, default=0.8,
                       help='Noise reduction proportion (noisereduce method)')
    parser.add_argument('--list-methods', action='store_true',
                       help='List available methods and exit')
    
    args = parser.parse_args()
    
    if args.list_methods:
        methods = get_available_methods()
        print("Available preprocessing methods:")
        for method in methods:
            print(f"  - {method}")
        if not methods:
            print("  (none available - install FFmpeg or Python packages)")
        return
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Process all audio files in directory
        output_dir = Path(args.output) if args.output else input_path / 'preprocessed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        
        for audio_file in input_path.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                output_file = output_dir / f"{audio_file.stem}_preprocessed.wav"
                print(f"Processing: {audio_file.name}")
                try:
                    preprocess_audio(
                        str(audio_file),
                        str(output_file),
                        method=args.method,
                        target_sr=args.target_sr,
                        noise_reduction_db=args.noise_reduction,
                        prop_decrease=args.prop_decrease
                    )
                    print(f"  -> {output_file}")
                except Exception as e:
                    print(f"  Error: {e}")
    else:
        # Process single file
        output_file = args.output if args.output else str(input_path.parent / f"{input_path.stem}_preprocessed.wav")
        
        result = preprocess_audio(
            args.input,
            output_file,
            method=args.method,
            target_sr=args.target_sr,
            noise_reduction_db=args.noise_reduction,
            prop_decrease=args.prop_decrease
        )
        print(f"Preprocessed audio saved to: {result}")


if __name__ == '__main__':
    main()

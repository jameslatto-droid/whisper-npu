# Audio Preprocessing & Noise Reduction Research for Whisper Transcription

This document provides a comprehensive analysis of audio preprocessing and noise reduction tools that can improve speech transcription quality before running Whisper.

---

## Table of Contents
1. [FFmpeg-based Solutions](#1-ffmpeg-based-solutions)
2. [AI/Deep Learning Audio Enhancement Tools](#2-aideep-learning-audio-enhancement-tools)
3. [Qualcomm/Snapdragon NPU Specific Options](#3-qualcommsnapdragon-npu-specific-options)
4. [Python Libraries for Audio Enhancement](#4-python-libraries-for-audio-enhancement)
5. [Recommended Pipeline](#5-recommended-pipeline)

---

## 1. FFmpeg-based Solutions

FFmpeg provides several powerful audio filters for noise reduction and speech enhancement. These are CPU-based but highly optimized and work on Windows ARM64.

### 1.1 `afftdn` - FFT-based Denoiser ⭐ RECOMMENDED

**Description**: Denoise audio samples using FFT (Fast Fourier Transform). This is FFmpeg's most powerful noise reduction filter.

**Key Parameters**:
- `nr` (noise_reduction): 0.01-97 dB (default: 12 dB)
- `nf` (noise_floor): -80 to -20 dB (default: -50 dB)
- `nt` (noise_type): white, vinyl, shellac, custom
- `tn` (track_noise): Enable automatic noise floor tracking

**Installation**: Included with FFmpeg
```powershell
# Windows ARM64 - Install via winget or download from https://ffmpeg.org
winget install FFmpeg
```

**Windows ARM64 Support**: ✅ Yes (native ARM64 builds available)

**NPU Acceleration**: ❌ No (CPU-based, but highly optimized)

**Code Example**:
```python
import subprocess

def denoise_with_afftdn(input_file, output_file, noise_reduction=12, noise_floor=-40):
    """Apply FFT-based denoising using FFmpeg afftdn filter."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', f'afftdn=nr={noise_reduction}:nf={noise_floor}:tn=1',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
    return output_file

# Example usage
denoise_with_afftdn('noisy_audio.wav', 'clean_audio.wav', noise_reduction=15, noise_floor=-40)
```

**Advanced Example with Noise Profiling**:
```python
def denoise_with_noise_profile(input_file, output_file):
    """Denoise by sampling noise from first 0.4 seconds."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', 'asendcmd=0.0 afftdn sn start,asendcmd=0.4 afftdn sn stop,afftdn=nr=20:nf=-40',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

### 1.2 `anlmdn` - Non-Local Means Denoiser

**Description**: Reduces broadband noise using Non-Local Means algorithm. Each sample is adjusted by looking for other samples with similar contexts.

**Key Parameters**:
- `s` (strength): 0.00001-10000 (default: 0.00001)
- `p` (patch): 1-100 ms (default: 2 ms)
- `r` (research): 2-300 ms (default: 6 ms)
- `m` (smooth): 1-1000 (default: 11)

**Code Example**:
```python
def denoise_with_anlmdn(input_file, output_file, strength=0.0001):
    """Apply Non-Local Means denoising."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', f'anlmdn=s={strength}:p=4:r=10',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

### 1.3 `afwtdn` - Wavelet Denoiser

**Description**: Reduces broadband noise using Wavelets. Good for preserving speech details.

**Key Parameters**:
- `sigma`: 0-1 (default: 0, use dB like -45dB)
- `levels`: 1-12 (default: 10)
- `percent`: 0-100% (default: 85%)

**Code Example**:
```python
def denoise_with_wavelet(input_file, output_file, sigma_db=-45):
    """Apply wavelet-based denoising."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', f'afwtdn=sigma={sigma_db}dB:levels=10:percent=85',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

### 1.4 `arnndn` - RNN-based Denoiser ⭐ AI-POWERED

**Description**: Reduces noise from speech using Recurrent Neural Networks. Requires a trained model file.

**Note**: Requires downloading RNNoise model files.

**Code Example**:
```python
def denoise_with_rnn(input_file, output_file, model_path):
    """Apply RNN-based denoising (requires model file)."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', f'arnndn=m={model_path}',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

### 1.5 `highpass` / `lowpass` Filters

**Description**: Basic frequency filtering to remove low-frequency rumble and high-frequency noise.

**Recommended Settings for Speech**:
- Highpass: 80-100 Hz (removes rumble, HVAC noise)
- Lowpass: 8000-12000 Hz (removes high-frequency hiss)

**Code Example**:
```python
def apply_speech_bandpass(input_file, output_file, highpass=80, lowpass=8000):
    """Apply bandpass filter optimized for speech."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', f'highpass=f={highpass},lowpass=f={lowpass}',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

### 1.6 `speechnorm` - Speech Normalizer

**Description**: Expands or compresses audio to reach target peak value. Great for normalizing quiet speech.

**Code Example**:
```python
def normalize_speech(input_file, output_file):
    """Normalize speech levels."""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', 'speechnorm=e=6:r=0.001:l=1',
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

### 1.7 Complete FFmpeg Preprocessing Pipeline

```python
def ffmpeg_preprocessing_pipeline(input_file, output_file):
    """Complete audio preprocessing pipeline for Whisper."""
    filter_chain = ','.join([
        'highpass=f=80',              # Remove low-frequency rumble
        'lowpass=f=8000',             # Remove high-frequency noise
        'afftdn=nr=12:nf=-40:tn=1',   # FFT-based denoising
        'speechnorm=e=3:r=0.001',     # Normalize speech levels
        'aresample=16000'              # Resample to 16kHz for Whisper
    ])
    
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', filter_chain,
        '-ac', '1',  # Mono
        '-y', output_file
    ]
    subprocess.run(cmd, check=True)
```

---

## 2. AI/Deep Learning Audio Enhancement Tools

### 2.1 DeepFilterNet ⭐ HIGHLY RECOMMENDED

**Description**: A low complexity speech enhancement framework for full-band audio (48kHz) using deep filtering. Real-time capable and very effective.

**Installation**:
```powershell
pip install deepfilternet
# Or for Rust-based CLI:
# Download from https://github.com/Rikorose/DeepFilterNet/releases
```

**Windows ARM64 Support**: ⚠️ Partial (Python wheel may work, Rust binary needs compilation)

**NPU Acceleration**: ❌ Not directly, but potential for ONNX export

**Code Example**:
```python
from df import enhance, init_df
import torchaudio

def denoise_with_deepfilternet(input_path, output_path):
    """Denoise audio using DeepFilterNet."""
    # Initialize model (downloads automatically)
    model, df_state, _ = init_df()
    
    # Load audio
    audio, sr = torchaudio.load(input_path)
    
    # Enhance (expects 48kHz)
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        audio = resampler(audio)
    
    # Denoise
    enhanced = enhance(model, df_state, audio)
    
    # Resample to 16kHz for Whisper
    resampler = torchaudio.transforms.Resample(48000, 16000)
    enhanced = resampler(enhanced)
    
    # Save
    torchaudio.save(output_path, enhanced, 16000)
    return output_path

# CLI usage
# deepFilter noisy_audio.wav --output-dir ./enhanced/
```

---

### 2.2 Facebook Denoiser (Demucs) ⚠️ ARCHIVED

**Description**: Real-time speech enhancement from Facebook Research. Based on encoder-decoder architecture with skip connections.

**Status**: Repository archived (Oct 2023), but still functional.

**Installation**:
```powershell
pip install denoiser
```

**Windows ARM64 Support**: ⚠️ Should work through PyTorch

**NPU Acceleration**: ❌ No (PyTorch-based, could potentially export to ONNX)

**Code Example**:
```python
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

def denoise_with_demucs(input_path, output_path):
    """Denoise using Facebook's Demucs model."""
    # Load pretrained model
    model = pretrained.dns64()
    model.eval()
    
    # Load audio
    wav, sr = torchaudio.load(input_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    
    # Denoise
    with torch.no_grad():
        denoised = model(wav[None])[0]
    
    # Save
    torchaudio.save(output_path, denoised.cpu(), model.sample_rate)
    return output_path

# CLI usage
# python -m denoiser.enhance --dns64 --noisy_dir ./noisy/ --out_dir ./clean/
```

---

### 2.3 Resemble Enhance

**Description**: AI-powered speech denoising and enhancement from Resemble AI. Includes both denoiser and enhancer modules for improving perceptual quality.

**Installation**:
```powershell
pip install resemble-enhance --upgrade
```

**Windows ARM64 Support**: ⚠️ Should work through PyTorch

**NPU Acceleration**: ❌ No (PyTorch-based)

**Code Example**:
```python
# CLI usage (simplest)
# resemble-enhance input_dir output_dir --denoise_only

# Python usage
import torch
from resemble_enhance.enhancer.inference import enhance

def denoise_with_resemble(input_path, output_path):
    """Denoise using Resemble Enhance."""
    # This requires GPU typically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and enhance
    enhanced, sr = enhance(
        input_path,
        device=device,
        denoise_only=True  # Only denoise, don't enhance
    )
    
    # Save result
    import soundfile as sf
    sf.write(output_path, enhanced, sr)
    return output_path
```

---

### 2.4 Microsoft NSNet / DNS Challenge Models

**Description**: Neural network-based noise suppression models from Microsoft's Deep Noise Suppression Challenge.

**Repository**: https://github.com/microsoft/DNS-Challenge

**Features**:
- Multiple pre-trained models available
- ONNX versions potentially available
- Designed for real-time use

**Installation**: Clone repository and use provided scripts

**NPU Potential**: ✅ High - ONNX models could run on QNN

---

### 2.5 SpeechBrain

**Description**: Comprehensive speech toolkit with denoising, enhancement, and separation capabilities.

**Installation**:
```powershell
pip install speechbrain
```

**Code Example**:
```python
from speechbrain.pretrained import SpectralMaskEnhancement

def denoise_with_speechbrain(input_path, output_path):
    """Denoise using SpeechBrain's enhancement model."""
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    
    enhanced = enhance_model.enhance_file(input_path)
    # Save the enhanced audio...
```

---

## 3. Qualcomm/Snapdragon NPU Specific Options

### 3.1 ONNX Runtime QNN Execution Provider ⭐ BEST FOR NPU

**Description**: Run ONNX models on Qualcomm NPU via QNN SDK. Supports audio processing models if properly quantized.

**Installation**:
```powershell
# For Windows ARM64
pip install onnxruntime-qnn
```

**Key Features**:
- Supports HTP (Hexagon Tensor Processor) / NPU backend
- Requires quantized models (uint8/uint16)
- Pre-built packages for Windows ARM64
- Python 3.11.x required

**Supported Audio Operations**: All standard ONNX ops including Conv1D, LSTM, GRU, LayerNorm, etc.

**Code Example**:
```python
import onnxruntime as ort
import numpy as np

def run_audio_model_on_npu(model_path, audio_input):
    """Run an ONNX audio model on Qualcomm NPU."""
    # Create session with QNN EP
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["QNNExecutionProvider"],
        provider_options=[{"backend_path": "QnnHtp.dll"}]  # Use NPU
    )
    
    # Run inference
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: audio_input})
    return result[0]
```

### 3.2 Converting Audio Models to QNN-Compatible ONNX

To run speech enhancement models on NPU:

1. **Export PyTorch model to ONNX**:
```python
import torch

def export_to_onnx(model, sample_input, output_path):
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        opset_version=13,
        input_names=['audio_input'],
        output_names=['enhanced_audio'],
        dynamic_axes=None  # QNN requires fixed shapes
    )
```

2. **Quantize for QNN**:
```python
from onnxruntime.quantization import quantize, CalibrationDataReader
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

# Preprocess and quantize
model_changed = qnn_preprocess_model(input_model, preprocessed_model)
qnn_config = get_qnn_qdq_config(
    preprocessed_model,
    data_reader,
    activation_type=QuantType.QUInt16,
    weight_type=QuantType.QUInt8
)
quantize(preprocessed_model, output_model, qnn_config)
```

### 3.3 Potential NPU-Accelerated Audio Models

| Model | ONNX Available | QNN Compatible | Notes |
|-------|---------------|----------------|-------|
| DeepFilterNet | ❌ (needs export) | ⚠️ Needs testing | Could be converted |
| RNNoise | ✅ | ⚠️ Needs quantization | Good candidate |
| DNS-Net | ⚠️ Partial | ⚠️ Needs work | Microsoft models |
| Custom CNN Denoiser | ✅ | ✅ | Best option |

### 3.4 Qualcomm AI Hub

**Description**: Cloud service for optimizing and running models on Qualcomm devices.

**Website**: https://aihub.qualcomm.com/

**Features**:
- Pre-optimized models for Snapdragon
- Supports ONNX Runtime QNN EP
- May have audio enhancement models

---

## 4. Python Libraries for Audio Enhancement

### 4.1 noisereduce ⭐ RECOMMENDED

**Description**: Noise reduction using spectral gating. Simple to use, effective for many scenarios.

**Installation**:
```powershell
pip install noisereduce
```

**Windows ARM64 Support**: ✅ Yes (pure Python + NumPy)

**NPU Acceleration**: ⚠️ TorchGate module could potentially be exported

**Features**:
- Stationary noise reduction (constant background noise)
- Non-stationary noise reduction (varying noise)
- PyTorch-based TorchGate module
- Can be used as nn.Module in neural networks

**Code Examples**:

```python
import noisereduce as nr
from scipy.io import wavfile

# Basic usage
def simple_denoise(input_path, output_path):
    """Simple noise reduction using noisereduce."""
    rate, data = wavfile.read(input_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output_path, rate, reduced_noise)

# Stationary noise reduction (for constant background noise)
def stationary_denoise(input_path, output_path):
    """Stationary noise reduction - better for constant noise."""
    rate, data = wavfile.read(input_path)
    reduced = nr.reduce_noise(
        y=data, 
        sr=rate, 
        stationary=True,
        prop_decrease=0.75,  # 75% noise reduction
        n_std_thresh_stationary=1.5
    )
    wavfile.write(output_path, rate, reduced)

# Non-stationary noise reduction (for varying noise)
def nonstationary_denoise(input_path, output_path):
    """Non-stationary noise reduction - better for varying noise."""
    rate, data = wavfile.read(input_path)
    reduced = nr.reduce_noise(
        y=data, 
        sr=rate, 
        stationary=False,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50
    )
    wavfile.write(output_path, rate, reduced)
```

**PyTorch TorchGate Usage**:
```python
import torch
from noisereduce.torchgate import TorchGate as TG

# Create TorchGating instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tg = TG(sr=16000, nonstationary=True).to(device)

# Apply to audio tensor
noisy_speech = torch.randn(1, 32000, device=device)  # [batch, samples]
enhanced_speech = tg(noisy_speech)
```

---

### 4.2 scipy.signal

**Description**: Signal processing functions including filtering, spectral analysis, and more.

**Installation**: Included with SciPy
```powershell
pip install scipy
```

**Windows ARM64 Support**: ✅ Yes

**Code Examples**:

```python
import numpy as np
from scipy import signal
from scipy.io import wavfile

def apply_butterworth_filter(input_path, output_path, lowcut=100, highcut=8000):
    """Apply Butterworth bandpass filter for speech."""
    rate, data = wavfile.read(input_path)
    
    # Normalize
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    
    # Design Butterworth bandpass filter
    nyq = 0.5 * rate
    low = lowcut / nyq
    high = highcut / nyq
    order = 5
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    # Save
    wavfile.write(output_path, rate, (filtered * 32768).astype(np.int16))

def spectral_subtraction(input_path, output_path, noise_frames=10):
    """Simple spectral subtraction noise reduction."""
    rate, data = wavfile.read(input_path)
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    
    # STFT
    f, t, Zxx = signal.stft(data, rate, nperseg=512)
    
    # Estimate noise from first few frames
    noise_estimate = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)
    
    # Spectral subtraction
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Subtract noise estimate
    magnitude_clean = np.maximum(magnitude - 2 * noise_estimate, 0.1 * magnitude)
    
    # Reconstruct
    Zxx_clean = magnitude_clean * np.exp(1j * phase)
    _, reconstructed = signal.istft(Zxx_clean, rate, nperseg=512)
    
    wavfile.write(output_path, rate, (reconstructed * 32768).astype(np.int16))
```

---

### 4.3 librosa

**Description**: Audio and music signal analysis library with many useful features.

**Installation**:
```powershell
pip install librosa
```

**Windows ARM64 Support**: ✅ Yes

**Code Examples**:

```python
import librosa
import numpy as np
import soundfile as sf

def preprocess_for_whisper(input_path, output_path):
    """Preprocess audio for optimal Whisper transcription."""
    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    
    # Resample to 16kHz (Whisper's expected rate)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Trim silence from beginning and end
    y, _ = librosa.effects.trim(y, top_db=30)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    # Save
    sf.write(output_path, y, sr)
    return output_path

def enhance_speech_harmonic(input_path, output_path):
    """Enhance speech using harmonic-percussive separation."""
    y, sr = librosa.load(input_path, sr=None)
    
    # Separate harmonic (speech) from percussive (transient noise)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Use mostly harmonic component (speech)
    y_enhanced = y_harmonic * 0.9 + y_percussive * 0.1
    
    sf.write(output_path, y_enhanced, sr)
```

---

### 4.4 pydub

**Description**: Simple audio manipulation library, good for format conversion and basic operations.

**Installation**:
```powershell
pip install pydub
```

**Windows ARM64 Support**: ✅ Yes (requires FFmpeg)

**Code Example**:
```python
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

def prepare_audio_with_pydub(input_path, output_path):
    """Basic audio preparation using pydub."""
    audio = AudioSegment.from_file(input_path)
    
    # Convert to mono
    audio = audio.set_channels(1)
    
    # Set sample rate to 16kHz
    audio = audio.set_frame_rate(16000)
    
    # Normalize volume
    audio = normalize(audio)
    
    # Optional: compress dynamic range
    audio = compress_dynamic_range(audio)
    
    # Export
    audio.export(output_path, format="wav")
```

---

### 4.5 pyAudioAnalysis

**Description**: Audio feature extraction, classification, and segmentation.

**Installation**:
```powershell
pip install pyAudioAnalysis
```

**Useful for**: Voice Activity Detection (VAD), speaker diarization, silence removal

---

## 5. Recommended Pipeline

### 5.1 Simple Pipeline (FFmpeg only) ⭐ FASTEST

For quick processing with minimal dependencies:

```python
import subprocess
import os

def simple_preprocessing_pipeline(input_file, output_file):
    """Simple FFmpeg-based preprocessing for Whisper."""
    
    filter_chain = ','.join([
        'highpass=f=80',                    # Remove low rumble
        'lowpass=f=8000',                   # Remove high noise  
        'afftdn=nr=10:nf=-40:tn=1',        # FFT denoising
        'speechnorm=e=3:r=0.001:l=1',      # Normalize levels
        'aresample=16000'                   # Whisper expects 16kHz
    ])
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', filter_chain,
        '-ac', '1',           # Mono
        '-acodec', 'pcm_s16le',
        output_file
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return output_file
```

### 5.2 Quality Pipeline (noisereduce + FFmpeg)

For better quality with Python libraries:

```python
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np

def quality_preprocessing_pipeline(input_file, output_file):
    """High-quality preprocessing using noisereduce + librosa."""
    
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    
    # Apply non-stationary noise reduction
    y_denoised = nr.reduce_noise(
        y=y, 
        sr=sr,
        stationary=False,
        prop_decrease=0.8,
        time_constant_s=2.0
    )
    
    # Resample to 16kHz for Whisper
    if sr != 16000:
        y_denoised = librosa.resample(y_denoised, orig_sr=sr, target_sr=16000)
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=25)
    
    # Normalize
    y_normalized = librosa.util.normalize(y_trimmed)
    
    # Save
    sf.write(output_file, y_normalized, 16000)
    return output_file
```

### 5.3 AI-Enhanced Pipeline (DeepFilterNet)

For maximum quality using deep learning:

```python
def ai_preprocessing_pipeline(input_file, output_file):
    """AI-enhanced preprocessing using DeepFilterNet."""
    import subprocess
    import tempfile
    import os
    
    # Step 1: Convert to 48kHz (DeepFilterNet expects this)
    temp_48k = tempfile.mktemp(suffix='.wav')
    subprocess.run([
        'ffmpeg', '-y', '-i', input_file,
        '-ar', '48000', '-ac', '1',
        temp_48k
    ], check=True, capture_output=True)
    
    # Step 2: Apply DeepFilterNet
    temp_enhanced = tempfile.mktemp(suffix='.wav')
    subprocess.run([
        'deepFilter', temp_48k, '-o', os.path.dirname(temp_enhanced)
    ], check=True, capture_output=True)
    
    # Step 3: Convert to 16kHz for Whisper
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_enhanced,
        '-ar', '16000', '-ac', '1',
        '-af', 'speechnorm=e=3:r=0.001',  # Normalize
        output_file
    ], check=True, capture_output=True)
    
    # Cleanup
    os.unlink(temp_48k)
    os.unlink(temp_enhanced)
    
    return output_file
```

### 5.4 NPU-Accelerated Pipeline (Future)

For Qualcomm NPU acceleration (requires ONNX model):

```python
def npu_preprocessing_pipeline(input_file, output_file, onnx_model_path):
    """NPU-accelerated preprocessing using ONNX Runtime QNN."""
    import onnxruntime as ort
    import librosa
    import soundfile as sf
    import numpy as np
    
    # Load audio
    y, sr = librosa.load(input_file, sr=16000, mono=True)
    
    # Prepare input (model-specific preprocessing)
    audio_input = y.reshape(1, 1, -1).astype(np.float32)
    
    # Create NPU session
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["QNNExecutionProvider"],
        provider_options=[{"backend_path": "QnnHtp.dll"}]
    )
    
    # Run enhancement on NPU
    enhanced = session.run(None, {"input": audio_input})[0]
    
    # Save
    sf.write(output_file, enhanced.squeeze(), 16000)
    return output_file
```

---

## Summary Table

| Tool | Type | Installation | ARM64 | NPU Possible | Quality | Speed |
|------|------|-------------|-------|--------------|---------|-------|
| FFmpeg afftdn | CPU | `winget install FFmpeg` | ✅ | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| FFmpeg anlmdn | CPU | `winget install FFmpeg` | ✅ | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| noisereduce | Python | `pip install noisereduce` | ✅ | ⚠️ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| DeepFilterNet | AI | `pip install deepfilternet` | ⚠️ | ⚠️ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Facebook Denoiser | AI | `pip install denoiser` | ⚠️ | ⚠️ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Resemble Enhance | AI | `pip install resemble-enhance` | ⚠️ | ⚠️ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| ONNX QNN | NPU | `pip install onnxruntime-qnn` | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Quick Start Recommendations

1. **For immediate use**: Use FFmpeg `afftdn` filter - no extra dependencies
2. **For better quality**: Add `noisereduce` Python library
3. **For best quality**: Use DeepFilterNet (requires PyTorch)
4. **For NPU acceleration**: Export model to ONNX and use `onnxruntime-qnn`

---

## Next Steps

1. Test FFmpeg filters on sample audio from your meeting recordings
2. Benchmark noisereduce vs FFmpeg quality/speed tradeoff
3. Evaluate DeepFilterNet quality on your specific audio
4. Investigate exporting DeepFilterNet to ONNX for potential NPU acceleration
5. Look into Qualcomm AI Hub for pre-optimized audio models

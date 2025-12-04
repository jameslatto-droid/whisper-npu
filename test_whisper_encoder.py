#!/usr/bin/env python3
"""
Test Whisper ONNX encoder on Snapdragon X Elite NPU.
"""

import onnxruntime as ort
import numpy as np
import time
import os

def test_whisper_encoder_npu():
    """Test Whisper encoder on NPU."""
    print("=" * 60)
    print("Testing Whisper ONNX Encoder on NPU (HTP Backend)")
    print("=" * 60)
    
    # Model paths
    model_dir = r"C:\Users\jimla\Projects\whisper-npu\whisper-tiny-onnx\onnx"
    
    # Try different encoder variants
    encoder_variants = [
        ("encoder_model_quantized.onnx", "Quantized (int8)"),
        ("encoder_model_uint8.onnx", "UInt8"),
        ("encoder_model.onnx", "FP32"),
        ("encoder_model_fp16.onnx", "FP16"),
    ]
    
    # Create dummy input (mel spectrogram: batch=1, features=80, time=3000)
    # This is the expected input shape for Whisper encoder
    dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)
    
    print(f"\nTest input shape: {dummy_input.shape}")
    print(f"Test input dtype: {dummy_input.dtype}")
    
    for model_file, description in encoder_variants:
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            print(f"\n⚠️ {model_file} not found, skipping...")
            continue
        
        print(f"\n{'='*50}")
        print(f"Testing: {model_file} ({description})")
        print(f"{'='*50}")
        
        # Test on CPU first
        try:
            print("\n🔧 CPU Backend:")
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['QNNExecutionProvider'],
                provider_options=[{"backend_path": "QnnCpu.dll"}]
            )
            
            # Get input name
            input_name = session.get_inputs()[0].name
            print(f"   Input name: {input_name}")
            
            # Run inference
            start = time.time()
            output = session.run(None, {input_name: dummy_input})
            elapsed = time.time() - start
            
            print(f"   ✅ Success!")
            print(f"   Output shape: {output[0].shape}")
            print(f"   Time: {elapsed*1000:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
        
        # Test on HTP (NPU)
        try:
            print("\n🔧 HTP (NPU) Backend:")
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['QNNExecutionProvider'],
                provider_options=[{
                    "backend_path": "QnnHtp.dll",
                    "htp_performance_mode": "burst",
                    "enable_htp_fp16_precision": "1"
                }]
            )
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Warm-up run
            _ = session.run(None, {input_name: dummy_input})
            
            # Timed run
            start = time.time()
            output = session.run(None, {input_name: dummy_input})
            elapsed = time.time() - start
            
            print(f"   ✅ Success!")
            print(f"   Output shape: {output[0].shape}")
            print(f"   Time: {elapsed*1000:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:200]}...")

if __name__ == "__main__":
    test_whisper_encoder_npu()

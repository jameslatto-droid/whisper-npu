#!/usr/bin/env python3
"""
Test QNN Execution Provider on Snapdragon X Elite NPU.
This script tests if the NPU backend is accessible.
"""

import onnxruntime as ort
import numpy as np
import time

def test_qnn_availability():
    """Test if QNN EP is available and working."""
    print("=" * 50)
    print("Testing QNN Execution Provider for NPU")
    print("=" * 50)
    
    # Check available providers
    providers = ort.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    
    if 'QNNExecutionProvider' not in providers:
        print("❌ QNNExecutionProvider not available!")
        return False
    
    print("✅ QNNExecutionProvider is available")
    
    # Test HTP backend availability
    print("\n🔍 Testing HTP (NPU) backend...")
    
    # Create a simple ONNX model to test
    import onnx
    from onnx import helper, TensorProto
    
    # Simple model: just an Add operation
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 4])
    
    add_node = helper.make_node('Add', ['X', 'Y'], ['Z'])
    
    graph = helper.make_graph([add_node], 'test_graph', [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7  # Use IR version 7 for compatibility
    
    # Save the model
    test_model_path = "test_qnn_model.onnx"
    onnx.save(model, test_model_path)
    print(f"✅ Created test model: {test_model_path}")
    
    # Try to create session with different backends
    backends = [
        ("CPU", {"backend_path": "QnnCpu.dll"}),
        ("HTP (NPU)", {"backend_path": "QnnHtp.dll"}),
    ]
    
    for name, provider_options in backends:
        try:
            print(f"\n🔧 Testing {name} backend...")
            session = ort.InferenceSession(
                test_model_path,
                providers=['QNNExecutionProvider'],
                provider_options=[provider_options]
            )
            
            # Run inference
            x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            y = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            
            start = time.time()
            result = session.run(None, {'X': x, 'Y': y})
            elapsed = time.time() - start
            
            print(f"   ✅ {name} backend works!")
            print(f"   Input X: {x}")
            print(f"   Input Y: {y}")
            print(f"   Result: {result[0]}")
            print(f"   Time: {elapsed*1000:.2f}ms")
            
        except Exception as e:
            print(f"   ❌ {name} backend failed: {e}")
    
    # Clean up
    import os
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    
    return True

if __name__ == "__main__":
    test_qnn_availability()

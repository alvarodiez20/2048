"""
Export trained DQN model to ONNX format for web deployment.

Usage:
    python export_onnx.py --model checkpoints/best_model.pt --output ../web/public/models/ai_model.onnx
"""

import argparse
import os

import torch
import numpy as np

from dqn_agent import DQN


def export_to_onnx(model_path: str, output_path: str, quantize: bool = False) -> None:
    """
    Export a trained DQN model to ONNX format.
    
    Args:
        model_path: Path to the PyTorch checkpoint
        output_path: Path for the output ONNX file
        quantize: Whether to apply dynamic quantization
    """
    print(f"Loading model from {model_path}...")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model and load weights
    model = DQN()
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['steps']} steps)")
    
    # Create dummy input
    dummy_input = torch.randn(1, 16, dtype=torch.float32)
    
    # Export to ONNX
    print(f"Exporting to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['q_values'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX export complete!")
    
    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Print model size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    # Optional: Quantize for smaller size
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = output_path.replace('.onnx', '_quantized.onnx')
            print(f"\nQuantizing to {quantized_path}...")
            
            quantize_dynamic(
                output_path,
                quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"Quantized model size: {quant_size_mb:.2f} MB")
            print(f"Size reduction: {(1 - quant_size_mb / size_mb) * 100:.1f}%")
        except ImportError:
            print("Quantization requires onnxruntime. Install with: pip install onnxruntime")
    
    # Test inference
    print("\nTesting ONNX inference...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(output_path)
    test_input = np.random.rand(1, 16).astype(np.float32)
    outputs = session.run(None, {'input': test_input})
    print(f"Test output shape: {outputs[0].shape}")
    print(f"Test Q-values: {outputs[0]}")


def main():
    parser = argparse.ArgumentParser(description='Export DQN model to ONNX')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='ai_model.onnx', help='Output ONNX file path')
    parser.add_argument('--quantize', action='store_true', help='Apply dynamic quantization')
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output, args.quantize)


if __name__ == "__main__":
    main()

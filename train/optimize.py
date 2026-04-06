"""Model optimization utilities for deployment.

Includes:
- ONNX conversion for cross-platform deployment
- Quantization (INT8, FP16) for reduced model size
- Profile models for inference speed
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


def convert_to_onnx(model_path: str, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 640, 640)):
    """Convert PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model (.pt file)
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    try:
        import torch
        import torch.onnx
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        return False
    
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location='cpu')
    if hasattr(model, 'model'):
        model = model.model  # Handle YOLOv8 models
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    print(f"Converting to ONNX with input shape {input_shape}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ ONNX model saved to {output_path}")
    
    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    return True


def quantize_onnx_int8(model_path: str, output_path: str, calibration_images: Optional[list] = None):
    """Quantize ONNX model to INT8 for faster inference.
    
    Args:
        model_path: Path to ONNX model
        output_path: Path to save quantized model
        calibration_images: List of images for calibration (optional)
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
        return False
    
    print(f"Quantizing {model_path} to INT8...")
    
    quantize_dynamic(
        model_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )
    
    print(f"✅ Quantized model saved to {output_path}")
    
    # Compare sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    
    return True


def benchmark_model(model_path: str, num_iterations: int = 100, input_shape: Tuple = (1, 3, 640, 640)):
    """Benchmark ONNX model inference speed.
    
    Args:
        model_path: Path to ONNX model
        num_iterations: Number of iterations for benchmarking
        input_shape: Input tensor shape
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed")
        return None
    
    print(f"Benchmarking {model_path}...")
    
    # Create session
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Create dummy input
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        sess.run(None, {input_name: dummy_input})
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(num_iterations):
        sess.run(None, {input_name: dummy_input})
    elapsed = time.time() - start
    
    fps = num_iterations / elapsed
    latency_ms = (elapsed / num_iterations) * 1000
    
    print(f"✅ Benchmark complete")
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency_ms:.2f} ms")
    
    return {
        'fps': fps,
        'latency_ms': latency_ms,
        'num_iterations': num_iterations
    }


def export_model_report(model_path: str, output_dir: str = '.'):
    """Generate a report about model specifications.
    
    Args:
        model_path: Path to ONNX model
        output_dir: Directory to save report
    """
    try:
        import onnx
    except ImportError:
        print("onnx not installed")
        return False
    
    print(f"Generating report for {model_path}...")
    
    model = onnx.load(model_path)
    
    report = {
        'model_path': str(model_path),
        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
        'opset_version': model.opset_import[0].version if model.opset_import else None,
        'inputs': [],
        'outputs': [],
        'initializers': len(model.graph.initializer),
    }
    
    # Collect input/output information
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        report['inputs'].append({
            'name': inp.name,
            'shape': shape,
            'dtype': str(inp.type.tensor_type.data_type)
        })
    
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        report['outputs'].append({
            'name': out.name,
            'shape': shape,
            'dtype': str(out.type.tensor_type.data_type)
        })
    
    # Save report
    report_path = os.path.join(output_dir, 'model_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report saved to {report_path}")
    print(json.dumps(report, indent=2))
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['convert', 'quantize', 'benchmark', 'report'], required=True)
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--output-path', help='Output path')
    parser.add_argument('--input-shape', default='1,3,640,640', help='Input shape (comma-separated)')
    
    args = parser.parse_args()
    
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    if args.action == 'convert':
        output = args.output_path or args.model_path.replace('.pt', '.onnx')
        convert_to_onnx(args.model_path, output, input_shape)
    
    elif args.action == 'quantize':
        output = args.output_path or args.model_path.replace('.onnx', '_quant.onnx')
        quantize_onnx_int8(args.model_path, output)
    
    elif args.action == 'benchmark':
        benchmark_model(args.model_path, input_shape=input_shape)
    
    elif args.action == 'report':
        export_model_report(args.model_path)

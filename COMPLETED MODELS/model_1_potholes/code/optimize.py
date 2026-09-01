import os
import time
import numpy as np
import cv2
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Import our detector classes
from cv_model import PotholeCVDetector

def run_benchmark(detector_pytorch, detector_onnx, detector_quant, num_iters=30):
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n--- Starting Benchmarking (Average Latency on CPU) ---")
    
    # 1. PyTorch Warmup & Benchmark
    print("Benchmarking PyTorch model...")
    for _ in range(5):
        _ = detector_pytorch.predict(dummy_frame)
    t0 = time.time()
    for _ in range(num_iters):
        _ = detector_pytorch.predict(dummy_frame)
    pytorch_time = (time.time() - t0) / num_iters * 1000
    print(f"PyTorch Average Latency: {pytorch_time:.2f} ms")
    
    # 2. ONNX Warmup & Benchmark
    print("Benchmarking Standard ONNX model...")
    for _ in range(5):
        _ = detector_onnx.predict(dummy_frame)
    t0 = time.time()
    for _ in range(num_iters):
        _ = detector_onnx.predict(dummy_frame)
    onnx_time = (time.time() - t0) / num_iters * 1000
    print(f"Standard ONNX Average Latency: {onnx_time:.2f} ms")
    
    # 3. Quantized ONNX Warmup & Benchmark
    print("Benchmarking Quantized ONNX model...")
    for _ in range(5):
        _ = detector_quant.predict(dummy_frame)
    t0 = time.time()
    for _ in range(num_iters):
        _ = detector_quant.predict(dummy_frame)
    quant_time = (time.time() - t0) / num_iters * 1000
    print(f"Quantized INT8 ONNX Average Latency: {quant_time:.2f} ms")
    
    # Summary of improvements
    speedup_onnx = pytorch_time / onnx_time
    speedup_quant = pytorch_time / quant_time
    print("\n--- Optimization Summary ---")
    print(f"Standard ONNX Speedup: {speedup_onnx:.2f}x")
    print(f"Quantized INT8 ONNX Speedup: {speedup_quant:.2f}x")
    print("-----------------------------\n")

def optimize_pipeline():
    pt_model_path = "yolov8n_pothole.pt"
    if not os.path.exists(pt_model_path):
        print(f"[Optimize] Trained model '{pt_model_path}' not found, falling back to 'yolov8n.pt'")
        pt_model_path = "yolov8n.pt"
        
    onnx_model_path = pt_model_path.replace(".pt", ".onnx")
    quant_model_path = pt_model_path.replace(".pt", "_quantized.onnx")
    
    # 1. Download/Load baseline PyTorch model
    print(f"[Optimize] Checking baseline model: {pt_model_path}")
    if not os.path.exists(pt_model_path) and pt_model_path == "yolov8n.pt":
        print(f"[Optimize] Downloading PyTorch YOLOv8n model...")
        model = YOLO(pt_model_path) # Instantiating downloads the weights
        
    # 2. Export PyTorch to ONNX
    print(f"[Optimize] Exporting {pt_model_path} to {onnx_model_path}...")
    model = YOLO(pt_model_path)
    # ultralytics export saves it to yolov8n.onnx
    success_path = model.export(format="onnx", imgsz=640)
    if os.path.exists(success_path) and success_path != onnx_model_path:
        os.rename(success_path, onnx_model_path)
        
    print(f"[Optimize] Raw ONNX size: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")
    
    # 3. Quantize ONNX to INT8
    print(f"[Optimize] Quantizing {onnx_model_path} to {quant_model_path} (INT8 Dynamic)...")
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quant_model_path,
        weight_type=QuantType.QUInt8
    )
    print(f"[Optimize] Quantized ONNX size: {os.path.getsize(quant_model_path) / (1024*1024):.2f} MB")
    
    # 4. Instantiate all detectors for benchmarking
    print("[Optimize] Initializing detector objects for benchmarking...")
    detector_pytorch = PotholeCVDetector(model_path=pt_model_path, use_onnx=False)
    detector_onnx = PotholeCVDetector(model_path=onnx_model_path, use_onnx=True)
    detector_quant = PotholeCVDetector(model_path=quant_model_path, use_onnx=True)
    
    # Run the benchmarks
    run_benchmark(detector_pytorch, detector_onnx, detector_quant)

if __name__ == "__main__":
    optimize_pipeline()

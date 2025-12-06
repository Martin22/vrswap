#!/usr/bin/env python3
"""
RTX 4060 Ti Benchmark Script
Měří výkon VRSwap na RTX 4060 Ti 16GB
"""

import os
import sys
import time
import cv2
import torch
import psutil
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import core.globals
from core.swapper import get_face_swapper
from core.analyser import get_face, get_faces
from core.border_blur import apply_border_blur, apply_border_blur_gpu


def print_gpu_info():
    """Vypíše info o GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA není dostupná!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("\n" + "="*60)
    print("🚀 GPU Information")
    print("="*60)
    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"FP16 Enabled: {core.globals.use_fp16}")
    print(f"Device: {core.globals.device}")
    print("="*60 + "\n")
    
    return True


def benchmark_face_detection(test_image_path, iterations=100):
    """Benchmark face detection"""
    print("📊 Benchmarking Face Detection...")
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"❌ Cannot load test image: {test_image_path}")
        return
    
    # Warm-up
    for _ in range(5):
        get_faces(img)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        faces = get_faces(img)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = iterations / elapsed
    ms_per_frame = (elapsed / iterations) * 1000
    
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Time per frame: {ms_per_frame:.1f}ms")
    print(f"  Faces detected: {len(faces) if faces else 0}")
    print()


def benchmark_face_swap(source_face_path, target_image_path, iterations=50):
    """Benchmark face swap"""
    print("📊 Benchmarking Face Swap...")
    
    # Load images
    source_img = cv2.imread(source_face_path)
    target_img = cv2.imread(target_image_path)
    
    if source_img is None or target_img is None:
        print("❌ Cannot load images")
        return
    
    # Get faces
    source_face = get_face(source_img)
    target_faces = get_faces(target_img)
    
    if source_face is None or not target_faces:
        print("❌ No faces detected")
        return
    
    swapper = get_face_swapper()
    
    # Warm-up
    for _ in range(3):
        if core.globals.use_fp16:
            with torch.autocast('cuda', dtype=torch.float16):
                swapper.get(target_img, target_faces[0], source_face, paste_back=True)
        else:
            swapper.get(target_img, target_faces[0], source_face, paste_back=True)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        result = target_img.copy()
        if core.globals.use_fp16:
            with torch.autocast('cuda', dtype=torch.float16):
                result = swapper.get(result, target_faces[0], source_face, paste_back=True)
        else:
            result = swapper.get(result, target_faces[0], source_face, paste_back=True)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = iterations / elapsed
    ms_per_frame = (elapsed / iterations) * 1000
    
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Time per swap: {ms_per_frame:.1f}ms")
    print()


def benchmark_border_blur(test_image_path, iterations=100):
    """Benchmark border blur - CPU vs GPU"""
    print("📊 Benchmarking Border Blur (CPU vs GPU)...")
    
    img = cv2.imread(test_image_path)
    if img is None:
        print("❌ Cannot load image")
        return
    
    # Get face bbox
    faces = get_faces(img)
    if not faces:
        print("❌ No face detected")
        return
    
    bbox = faces[0].bbox
    
    # === CPU Benchmark ===
    print("  Testing CPU version...")
    start = time.time()
    for _ in range(iterations):
        apply_border_blur(img.copy(), bbox, blur_strength=15)
    elapsed_cpu = time.time() - start
    fps_cpu = iterations / elapsed_cpu
    
    # === GPU Benchmark ===
    print("  Testing GPU version...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        apply_border_blur_gpu(img.copy(), bbox, blur_strength=15, device='cuda')
    torch.cuda.synchronize()
    elapsed_gpu = time.time() - start
    fps_gpu = iterations / elapsed_gpu
    
    speedup = elapsed_cpu / elapsed_gpu
    
    print(f"\n  CPU:")
    print(f"    Time: {elapsed_cpu:.2f}s")
    print(f"    FPS: {fps_cpu:.1f}")
    print(f"    Per frame: {(elapsed_cpu/iterations)*1000:.1f}ms")
    
    print(f"\n  GPU:")
    print(f"    Time: {elapsed_gpu:.2f}s")
    print(f"    FPS: {fps_gpu:.1f}")
    print(f"    Per frame: {(elapsed_gpu/iterations)*1000:.1f}ms")
    
    print(f"\n  🚀 Speedup: {speedup:.2f}x")
    print()


def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("📊 Memory Usage:")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"  GPU Memory:")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Reserved: {reserved:.2f} GB")
        print(f"    Total: {total:.2f} GB")
        print(f"    Free: {total - reserved:.2f} GB")
    
    ram = psutil.virtual_memory()
    print(f"\n  System RAM:")
    print(f"    Used: {ram.used / 1e9:.2f} GB")
    print(f"    Total: {ram.total / 1e9:.2f} GB")
    print()


def main():
    """Main benchmark"""
    print("\n" + "="*60)
    print("🎯 RTX 4060 Ti VRSwap Benchmark")
    print("="*60 + "\n")
    
    # Check GPU
    if not print_gpu_info():
        return
    
    # Create test images if they don't exist
    test_dir = Path("benchmark_test")
    test_dir.mkdir(exist_ok=True)
    
    test_image = test_dir / "test_1080p.jpg"
    
    if not test_image.exists():
        print("⚠️  Creating test image (1920x1080)...")
        # Create synthetic test image
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(test_image), img)
    
    # Run benchmarks
    try:
        # Note: Pro skutečný benchmark potřebujete skutečné obrázky s tvářemi
        print("⚠️  Pro kompletní benchmark nahrajte:")
        print("    - benchmark_test/source_face.jpg")
        print("    - benchmark_test/target_image.jpg")
        print()
        
        if (test_dir / "target_image.jpg").exists():
            benchmark_face_detection(str(test_dir / "target_image.jpg"))
        
        if (test_dir / "source_face.jpg").exists() and (test_dir / "target_image.jpg").exists():
            benchmark_face_swap(
                str(test_dir / "source_face.jpg"),
                str(test_dir / "target_image.jpg")
            )
            benchmark_border_blur(str(test_dir / "target_image.jpg"))
        
        benchmark_memory_usage()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Benchmark Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

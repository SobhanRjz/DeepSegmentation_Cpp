#!/usr/bin/env python3
"""
YOLO Batch Processing Speed Test
Simple test script to measure YOLO's native batch processing performance.
"""

from ultralytics import YOLO
import time
import os
import sys
from typing import List
import torch
import cv2

def create_test_image_list(base_image: str, count: int = 50) -> List[str]:
    """Create a list of test images by repeating the base image."""
    if not os.path.exists(base_image):
        raise FileNotFoundError(f"Base test image not found: {base_image}")
    
    # Create list by repeating the same image
    image_list = [base_image] * count
    print(f"Created test image list with {count} copies of: {base_image}")
    return image_list

def test_yolo_batch_speed():
    """Test YOLO batch processing speed with different batch sizes."""
    
    # Configuration
    MODEL_PATH = "../YoloModel/yolov8m_SegmentationDetection_dynamic.pt"
    TEST_IMAGE = "../Dataset/input_SegmentDetection/TestImage.png"
    
    # Try alternative model paths
    model_paths = [
        MODEL_PATH,
        "YoloModel/yolov8n-seg.pt",
        "yolov8n-seg.pt"  # Will download if not found
    ]
    
    actual_model_path = None
    for path in model_paths:
        if os.path.exists(path):
            actual_model_path = path
            print(f"✅ Found model: {path}")
            break
    
    if actual_model_path is None:
        print("📥 Downloading yolov8n-seg.pt model...")
        actual_model_path = "yolov8n-seg.pt"
    
    # Try alternative image paths
    test_images = [
        TEST_IMAGE,
        "Dataset/input_SegmentDetection/TestImage.png",
        "Dataset/input_SegmentationDetection/TestImage.png",
    ]
    
    actual_image_path = None
    for path in test_images:
        if os.path.exists(path):
            actual_image_path = path
            print(f"✅ Found test image: {path}")
            break
    
    if actual_image_path is None:
        print("❌ No test image found. Please provide a valid image path.")
        return
    
    print("=" * 80)
    print("🚀 YOLO BATCH PROCESSING SPEED TEST")
    print("=" * 80)
    
    # Load model
    print(f"📥 Loading YOLO model: {actual_model_path}")
    start_load = time.time()
    model = YOLO(actual_model_path)
    end_load = time.time()
    print(f"✅ Model loaded in {end_load - start_load:.4f} seconds")
    model.to("cuda")

    # Now, proper timing
    torch.cuda.synchronize()
    start = time.time()
    _ = model.predict(actual_image_path, conf=0.2, verbose=False)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Inference time: {(end - start)*1000:.2f} ms")


    t = time.time()
    model.predict(actual_image_path, conf=0.2, verbose=False)
    end_predict = time.time()
    RealTime = end_predict - t
    print(f"✅ Model predicted in {RealTime:.4f} seconds")




    # Test different batch sizes
    batch_sizes = [1, 5, 10, 50]
    results_summary = []
    
    for batch_size in batch_sizes:
        print(f"\n" + "-" * 60)
        print(f"🧪 Testing batch size: {batch_size}")
        print(f"-" * 60)
        
        # Create image list
        imgs = create_test_image_list(actual_image_path, batch_size)
        
        # Warm-up run (exclude from timing)
        print("🔥 Warm-up run...")
        _ = model(imgs[:min(4, len(imgs))], conf=0.2, verbose=False)
        
        # Test individual image processing for comparison (only for smaller batches)
        individual_times = []
        if batch_size <= 10:
            print(f"🔍 Testing individual image processing for comparison...")
            for i in range(min(5, batch_size)):  # Test first 5 images individually
                frame = cv2.imread(imgs[i])
                start_single = time.time()
                _ = model.predict(frame, conf=0.2, verbose=False)
                end_single = time.time()
                print(f"✅ Model predicted in {end_single - start_single:.4f} seconds")


            # while True:  # Test first 5 images individually
            #     frame = cv2.imread(imgs[i])
            #     start_single = time.time()
            #     _ = model.predict(frame, conf=0.2, verbose=False)
            #     end_single = time.time()
            #     print(f"✅ Model predicted in {end_single - start_single:.4f} seconds")
                #individual_times.append(end_single - start_single)
            
            # avg_individual_time = sum(individual_times) / len(individual_times)
            # print(f"📊 Individual processing times (first {len(individual_times)} images):")
            # for i, t in enumerate(individual_times):
            #     print(f"   Image {i+1}: {t:.4f} seconds")
            # print(f"   Average individual time: {avg_individual_time:.4f} seconds")
        
        # Actual timed run
        print(f"⏱️  Running batch inference on {batch_size} images...")
        start_time = time.time()
        
        # YOLO batch inference - the main test
        results = model.predict(imgs, conf=0.2, verbose=False)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        fps = batch_size / total_time
        avg_per_image = total_time / batch_size
        
        # Count detections and print per-image details
        total_detections = 0
        print(f"\n📋 Per-image processing details:")
        print(f"{'Image #':<8} {'Time (sec)':<12} {'Detections':<12} {'Status':<10}")
        print("-" * 50)
        
        for i, r in enumerate(results):
            # Calculate estimated time per image (since batch processing doesn't give individual times)
            estimated_time = avg_per_image
            detections_count = 0
            
            if hasattr(r, 'boxes') and r.boxes is not None:
                detections_count = len(r.boxes)
                total_detections += detections_count
            
            status = "✅ OK" if detections_count > 0 else "⚪ None"
            print(f"{i+1:<8} {estimated_time:<12.4f} {detections_count:<12} {status:<10}")
        
        print("-" * 50)
        print(f"{'Total:':<8} {total_time:<12.4f} {total_detections:<12}")
        
        # Store results
        result = {
            'batch_size': batch_size,
            'total_time': total_time,
            'fps': fps,
            'avg_per_image': avg_per_image,
            'total_detections': total_detections,
            'individual_times': individual_times if individual_times else None
        }
        results_summary.append(result)
        
        # Print batch results
        print(f"📊 Batch {batch_size} Results:")
        print(f"   Total time: {total_time:.4f} seconds")
        print(f"   FPS: {fps:.2f} images/second")
        print(f"   Average per image: {avg_per_image:.4f} seconds")
        if individual_times:
            print(f"   Individual avg: {avg_individual_time:.4f} seconds")
            efficiency_gain = (avg_individual_time - avg_per_image) / avg_individual_time * 100
            print(f"   Batch efficiency: {efficiency_gain:.1f}% faster than individual")
        print(f"   Total detections: {total_detections}")
        print(f"   Throughput: {fps * 60:.1f} images/minute")
    
    # Print final comparison
    print("\n" + "=" * 80)
    print("📈 BATCH PROCESSING PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Total Time':<12} {'FPS':<10} {'Avg/Image':<12} {'Detections':<12}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['batch_size']:<12} "
              f"{result['total_time']:<12.4f} "
              f"{result['fps']:<10.2f} "
              f"{result['avg_per_image']:<12.4f} "
              f"{result['total_detections']:<12}")
    
    # Find best performance
    best_fps = max(results_summary, key=lambda x: x['fps'])
    best_efficiency = min(results_summary, key=lambda x: x['avg_per_image'])
    
    print("\n" + "=" * 80)
    print("🏆 PERFORMANCE HIGHLIGHTS")
    print("=" * 80)
    print(f"🚀 Best FPS: {best_fps['fps']:.2f} images/sec (batch size {best_fps['batch_size']})")
    print(f"⚡ Best efficiency: {best_efficiency['avg_per_image']:.4f} sec/image (batch size {best_efficiency['batch_size']})")
    
    # Calculate speedup vs single image
    single_image_time = results_summary[0]['avg_per_image']
    print(f"\n📊 Speedup compared to single image processing:")
    for result in results_summary[1:]:  # Skip batch_size=1
        speedup = single_image_time / result['avg_per_image']
        print(f"   Batch {result['batch_size']:2d}: {speedup:.2f}x faster")
    
    print("\n" + "=" * 80)
    print("✅ BATCH SPEED TEST COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_yolo_batch_speed()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1) 
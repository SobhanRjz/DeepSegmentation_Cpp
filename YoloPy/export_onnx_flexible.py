#!/usr/bin/env python3
"""
Flexible ONNX Export Script for YOLO Segmentation Models
This script provides multiple options for exporting ONNX models with different input dimension strategies.
"""

from ultralytics import YOLO
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from typing import Optional, Tuple, List


def export_torchscript(model_path: str, output_path: Optional[str] = None) -> str:
    """
    Export ONNX model with fully dynamic input dimensions.
    Input shape: [batch, 3, height, width] where batch, height, width can vary.
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        output_path: Output path for ONNX file
        
    Returns:
        Path to exported ONNX file
    """
    model = YOLO(model_path)
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"output/{base_name}_dynamic.torchscript.pt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🔄 Exporting ONNX with DYNAMIC input dimensions...")
    print(f"   Input shape: [batch, 3, height, width] - ALL DYNAMIC")
    print(f"   Output path: {output_path}")
    
    # Export with dynamic=True for fully flexible input dimensions
    success = model.export(
        format="torchscript",
        imgsz=(1920, 1080),  # Default size, but will be dynamic
        simplify=False,
        opset=12,
        verbose=True, 
        half=False
    )
    
    # Move the exported file to the specified output path if needed
    default_export_path = model_path.replace('.pt', '.torchscript.pt')
    if os.path.exists(default_export_path) and default_export_path != output_path:
        import shutil
        shutil.move(default_export_path, output_path)
    
    if success:
        print(f"✅ Dynamic ONNX model exported: {output_path}")
        return output_path
    else:
        raise RuntimeError("Failed to export dynamic ONNX model")




def export_onnx_dynamic(model_path: str, output_path: Optional[str] = None) -> str:
    """
    Export ONNX model with fully dynamic input dimensions.
    Input shape: [batch, 3, height, width] where batch, height, width can vary.
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        output_path: Output path for ONNX file
        
    Returns:
        Path to exported ONNX file
    """
    model = YOLO(model_path)
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"output/{base_name}_dynamic.onnx"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🔄 Exporting ONNX with DYNAMIC input dimensions...")
    print(f"   Input shape: [batch, 3, height, width] - ALL DYNAMIC")
    print(f"   Output path: {output_path}")
    
    # Export with dynamic=True for fully flexible input dimensions
    success = model.export(
        format="onnx",
        imgsz=(1920, 1080),  # Default size, but will be dynamic
        dynamic=True,      # Enable dynamic dimensions
        simplify=True,
        opset=12,
        verbose=True, 
        half=False
    )
    
    # Move the exported file to the specified output path if needed
    default_export_path = model_path.replace('.pt', '.onnx')
    if os.path.exists(default_export_path) and default_export_path != output_path:
        import shutil
        shutil.move(default_export_path, output_path)
    
    if success:
        print(f"✅ Dynamic ONNX model exported: {output_path}")
        return output_path
    else:
        raise RuntimeError("Failed to export dynamic ONNX model")


def export_onnx_fixed_large(model_path: str, size: Tuple[int, int] = (1920, 1920), 
                           output_path: Optional[str] = None) -> str:
    """
    Export ONNX model with larger fixed input dimensions.
    Input shape: [1, 3, height, width] with specified large dimensions.
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        size: Fixed input size (width, height)
        output_path: Output path for ONNX file
        
    Returns:
        Path to exported ONNX file
    """
    model = YOLO(model_path)
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"output/{base_name}_fixed_{size[0]}x{size[1]}.onnx"
    
    print(f"🔄 Exporting ONNX with LARGE FIXED input dimensions...")
    print(f"   Input shape: [1, 3, {size[1]}, {size[0]}] - FIXED LARGE SIZE")
    
    # Export with larger fixed dimensions
    success = model.export(
        format="onnx",
        # Large fixed size
        dynamic=False,     # Fixed dimensions
        simplify=False,
        opset=12,
        verbose=True
    )
    
    # Move the exported file to the specified output path if needed
    default_export_path = model_path.replace('.pt', '.onnx')
    if os.path.exists(default_export_path) and default_export_path != output_path:
        import shutil
        shutil.move(default_export_path, output_path)
    
    if success:
        print(f"✅ Dynamic ONNX model exported: {output_path}")
        return output_path
    else:
        raise RuntimeError("Failed to export dynamic ONNX model")



def export_onnx_batch_dynamic(model_path: str, fixed_size: Tuple[int, int] = (1280, 1280),
                             output_path: Optional[str] = None) -> str:
    """
    Export ONNX model with dynamic batch size but fixed spatial dimensions.
    Input shape: [batch, 3, height, width] where only batch is dynamic.
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        fixed_size: Fixed spatial dimensions (width, height)
        output_path: Output path for ONNX file
        
    Returns:
        Path to exported ONNX file
    """
    model = YOLO(model_path)
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"output/{base_name}_batch_dynamic_{fixed_size[0]}x{fixed_size[1]}.onnx"
    
    print(f"🔄 Exporting ONNX with BATCH DYNAMIC input dimensions...")
    print(f"   Input shape: [batch, 3, {fixed_size[1]}, {fixed_size[0]}] - BATCH DYNAMIC")
    
    # This requires manual ONNX modification after export
    # First export with dynamic=True, then modify the input shape
    success = model.export(
        format="onnx",
        imgsz=fixed_size,
        dynamic=True,      # Enable dynamic, then we'll modify
        simplify=True,
        opset=12,
        verbose=True
    )
        # Move the exported file to the specified output path if needed
    default_export_path = model_path.replace('.pt', '.onnx')
    if os.path.exists(default_export_path) and default_export_path != output_path:
        import shutil
        shutil.move(default_export_path, output_path)
    
    if success:
        print(f"✅ Dynamic ONNX model exported: {output_path}")
        return output_path
    else:
        raise RuntimeError("Failed to export dynamic ONNX model")
    


def test_onnx_model(onnx_path: str, test_sizes: List[Tuple[int, int]] = None) -> None:
    """
    Test the exported ONNX model withnon_max_suppression"""
    print(f"\n🧪 Testing ONNX model: {onnx_path}")
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"📊 Model Information:")
        print(f"   Input name: {input_info.name}")
        print(f"   Input shape: {input_info.shape}")
        print(f"   Input type: {input_info.type}")
        
        for i, output in enumerate(output_info):
            print(f"   Output {i}: {output.name} - {output.shape}")
        
        # Test with different input sizes
        for width, height in test_sizes:
            try:
                print(f"\n🔍 Testing with input size: {width}x{height}")
                
                # Create dummy input
                dummy_input = np.random.randn(1, 3, height, width).astype(np.float32)
                
                # Run inference
                outputs = session.run(None, {input_info.name: dummy_input})
                
                print(f"   ✅ Success! Output shapes:")
                for i, output in enumerate(outputs):
                    print(f"      Output {i}: {output.shape}")
                    
            except Exception as e:
                print(f"   ❌ Failed with {width}x{height}: {str(e)}")
                
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {str(e)}")


def main():
    """Main function to demonstrate different ONNX export options."""
    
    # Configuration
    MODEL_PATH = "YoloModel/yolov8m_PointDetection_dynamic.pt"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please ensure the model file exists.")
        return 1
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    print("=== YOLO ONNX Export Options ===")
    print(f"Model: {MODEL_PATH}")
    print()
    
    try:

        # Option 0: Fully Dynamic Input Dimensions
        print("1️⃣  OPTION 0: Fully Dynamic Input Dimensions (TorchScript)")
        print("   Pros: Can handle any input size")
        print("   Cons: May be slower, requires dynamic ONNX Runtime support")
        dynamic_path = export_torchscript(MODEL_PATH)
        #test_onnx_model(dynamic_path, [(640, 640), (1280, 736), (800, 608)])
        print()


        # Option 1: Fully Dynamic Input Dimensions
        print("1️⃣  OPTION 1: Fully Dynamic Input Dimensions")
        print("   Pros: Can handle any input size")
        print("   Cons: May be slower, requires dynamic ONNX Runtime support")
        dynamic_path = export_onnx_dynamic(MODEL_PATH)
        test_onnx_model(dynamic_path, [(640, 640), (1280, 736), (800, 608)])
        print()
        
        # Option 2: Large Fixed Input Dimensions
        print("2️⃣  OPTION 2: Large Fixed Input Dimensions")
        print("   Pros: Fast inference, handles most image sizes with padding")
        print("   Cons: Uses more memory, requires padding for smaller images")
        large_path = export_onnx_fixed_large(MODEL_PATH, size=(1280, 1280))
        test_onnx_model(large_path, [(640, 640), (1280, 1280), (800, 600)])
        print()
        
        # Option 3: Medium Fixed Input Dimensions
        print("3️⃣  OPTION 3: Medium Fixed Input Dimensions")
        print("   Pros: Good balance of speed and memory usage")
        print("   Cons: May need padding for larger images")
        medium_path = export_onnx_fixed_large(MODEL_PATH, size=(1280, 1280))
        test_onnx_model(medium_path, [(640, 640), (1280, 1280), (800, 600)])
        print()
        
        print("✅ All ONNX export options completed!")
        print("\n📋 Summary of exported models:")
        print(f"   Dynamic: {dynamic_path}")
        print(f"   Large Fixed (1920x1920): {large_path}")
        print(f"   Medium Fixed (1280x1280): {medium_path}")
        
        print("\n💡 Recommendations:")
        print("   • Use DYNAMIC for maximum flexibility (if your C++ supports it)")
        print("   • Use LARGE FIXED for high-resolution images")
        print("   • Use MEDIUM FIXED for balanced performance")
        
    except Exception as e:
        print(f"❌ Error during export: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    main() 
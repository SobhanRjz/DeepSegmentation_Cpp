from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import os
import sys
import logging
from dataclasses import dataclass
import json
import onnx
import onnxruntime as ort
from pathlib import Path

# Add parent directory to path to import config_reader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_reader import get_config
    # Change to parent directory for config file
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = get_config()
    CONF_THRESHOLD = config.confidence_threshold
    INPUT_WIDTH = config.input_width
    INPUT_HEIGHT = config.input_height
    # Change back to original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except ImportError:
    # Fallback values if config_reader is not available
    CONF_THRESHOLD = 0.5
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
except Exception:
    # Fallback values if config loading fails
    CONF_THRESHOLD = 0.5
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640


@dataclass
class Point:
    """Represents a detected point with coordinates and metadata."""
    x: float
    y: float
    confidence: float
    class_id: int
    class_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert point to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }


class YOLOPointDetector:
    """
    Advanced YOLO Point Detection System
    
    This class provides comprehensive point detection capabilities including:
    - Keypoint detection
    - Custom point detection
    - Point tracking across frames
    - Advanced visualization
    - Statistical analysis
    """
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", script_dir: Optional[str] = None):
        """
        Initialize the YOLO Point Detector.
        
        Args:
            model_path (str): Path to YOLO model (pose model for keypoints)
            script_dir (str, optional): Directory path
        """
        self.model_path = model_path
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self.model = None
        self.output_dir = os.path.join(self.script_dir, "output")
        self.detected_points: List[Point] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Keypoint names for pose detection (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Color palette for visualization
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 0), (0, 128, 128), (192, 192, 192), (255, 20, 147),
            (75, 0, 130)
        ]
        
        self._setup_directories()
        self._load_model()
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "points"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        self.logger.info(f"Output directories created: {self.output_dir}")
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            # Check if model exists locally
            if not os.path.exists(self.model_path):
                # Try to find it in the script directory
                local_model_path = os.path.join(self.script_dir, self.model_path)
                if os.path.exists(local_model_path):
                    self.model_path = local_model_path
                else:
                    self.logger.info(f"Model {self.model_path} not found locally, will download from Ultralytics")
            
            self.model = YOLO(self.model_path)
            self.logger.info(f"Successfully loaded YOLO model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_path}: {str(e)}")
            raise
    
    def convert_to_onnx(self, 
                       onnx_path: Optional[str] = None,
                       dynamic_batch: bool = True,
                       input_shape: Tuple[int, int] = None,
                       opset_version: int = 11,
                       simplify: bool = False,
                       verify: bool = True) -> str:
        """
        Convert YOLO model to ONNX format with dynamic batch processing support.
        
        Args:
            onnx_path (str, optional): Output path for ONNX model. If None, auto-generated
            dynamic_batch (bool): Enable dynamic batch size for batch processing
            input_shape (Tuple[int, int], optional): Input image shape (height, width). 
                                                   Uses config values if None
            opset_version (int): ONNX opset version
            simplify (bool): Simplify the ONNX model using onnx-simplifier
            verify (bool): Verify the converted model
            
        Returns:
            str: Path to the converted ONNX model
        """
        if not self.model:
            raise ValueError("Model not loaded. Call _load_model() first.")
        
        # Set default input shape
        if input_shape is None:
            input_shape = (INPUT_HEIGHT, INPUT_WIDTH)
        
        # Generate ONNX path if not provided
        if onnx_path is None:
            model_name = Path(self.model_path).stem
            onnx_path = os.path.join(self.output_dir, f"{model_name}_dynamic.onnx")
        
        try:
            self.logger.info(f"Converting YOLO model to ONNX format...")
            self.logger.info(f"Input shape: {input_shape}")
            self.logger.info(f"Dynamic batch: {dynamic_batch}")
            self.logger.info(f"Output path: {onnx_path}")
            
            # Prepare export arguments
            export_args = {
                'format': 'onnx',
                'imgsz': input_shape,
                'opset': opset_version,
                'simplify': simplify,
                'dynamic': dynamic_batch,
            }
            
            # Export to ONNX
            onnx_model_path = self.model.export(**export_args)
            
            # Move to desired location if different
            if onnx_model_path != onnx_path:
                import shutil
                shutil.move(onnx_model_path, onnx_path)
            
            self.logger.info(f"Model successfully converted to ONNX: {onnx_path}")
            
            # Verify the model if requested
            if verify:
                self._verify_onnx_model(onnx_path, input_shape, dynamic_batch)
            
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"ONNX conversion failed: {str(e)}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str, input_shape: Tuple[int, int], 
                          dynamic_batch: bool) -> None:
        """
        Verify the converted ONNX model.
        
        Args:
            onnx_path (str): Path to ONNX model
            input_shape (Tuple[int, int]): Input shape used for conversion
            dynamic_batch (bool): Whether dynamic batch was enabled
        """
        try:
            self.logger.info("Verifying ONNX model...")
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()
            
            self.logger.info(f"ONNX model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            self.logger.info(f"ONNX model outputs: {len(output_info)} outputs")
            
            for i, output in enumerate(output_info):
                self.logger.info(f"  Output {i}: {output.name}, shape: {output.shape}, type: {output.type}")
            
            # Test inference with dummy data
            if dynamic_batch:
                # Test with different batch sizes
                batch_sizes = [1, 2, 4]
                for batch_size in batch_sizes:
                    dummy_input = np.random.randn(batch_size, 3, input_shape[0], input_shape[1]).astype(np.float32)
                    try:
                        outputs = ort_session.run(None, {input_info.name: dummy_input})
                        self.logger.info(f"✓ Batch size {batch_size} test passed. Output shapes: {[out.shape for out in outputs]}")
                    except Exception as e:
                        self.logger.warning(f"✗ Batch size {batch_size} test failed: {str(e)}")
            else:
                # Test with single batch
                dummy_input = np.random.randn(1, 3, input_shape[0], input_shape[1]).astype(np.float32)
                outputs = ort_session.run(None, {input_info.name: dummy_input})
                self.logger.info(f"✓ Single batch test passed. Output shapes: {[out.shape for out in outputs]}")
            
            self.logger.info("ONNX model verification completed successfully!")
            
        except Exception as e:
            self.logger.error(f"ONNX model verification failed: {str(e)}")
            raise
    
    def load_onnx_model(self, onnx_path: str) -> ort.InferenceSession:
        """
        Load ONNX model for inference.
        
        Args:
            onnx_path (str): Path to ONNX model
            
        Returns:
            ort.InferenceSession: ONNX Runtime session
        """
        try:
            # Configure ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ort_session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
            
            self.logger.info(f"ONNX model loaded successfully: {onnx_path}")
            self.logger.info(f"Available providers: {ort_session.get_providers()}")
            
            return ort_session
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {str(e)}")
            raise
    
    def detect_points_onnx(self, images: List[str], onnx_session: ort.InferenceSession,
                          conf: float = CONF_THRESHOLD) -> List[List[Point]]:
        """
        Detect points using ONNX model with batch processing support.
        
        Args:
            images (List[str]): List of image paths
            onnx_session (ort.InferenceSession): ONNX Runtime session
            conf (float): Confidence threshold
            
        Returns:
            List[List[Point]]: List of point lists for each image
        """
        if not images:
            return []
        
        try:
            # Prepare batch input
            batch_input = self._prepare_batch_input(images)
            
            # Get input/output names
            input_name = onnx_session.get_inputs()[0].name
            output_names = [output.name for output in onnx_session.get_outputs()]
            
            # Run inference
            outputs = onnx_session.run(output_names, {input_name: batch_input})
            
            # Process outputs for each image in batch
            batch_points = []
            for i in range(len(images)):
                # Extract outputs for this image
                image_outputs = [output[i:i+1] for output in outputs]  # Keep batch dimension
                
                # Convert to YOLO-like results format and extract points
                points = self._extract_points_from_onnx_outputs(image_outputs, conf)
                batch_points.append(points)
            
            self.logger.info(f"Processed batch of {len(images)} images with ONNX model")
            return batch_points
            
        except Exception as e:
            self.logger.error(f"ONNX batch inference failed: {str(e)}")
            raise
    
    def _prepare_batch_input(self, image_paths: List[str]) -> np.ndarray:
        """
        Prepare batch input for ONNX inference.
        
        Args:
            image_paths (List[str]): List of image paths
            
        Returns:
            np.ndarray: Batch input tensor
        """
        batch_images = []
        
        for image_path in image_paths:
            # Load and preprocess image
            if not os.path.isabs(image_path):
                full_path = os.path.join(self.script_dir, image_path)
            else:
                full_path = image_path
            
            image = cv2.imread(full_path)
            if image is None:
                raise ValueError(f"Could not load image: {full_path}")
            
            # Resize image
            image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
            
            # Convert BGR to RGB and normalize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            image = np.transpose(image, (2, 0, 1))
            
            batch_images.append(image)
        
        # Stack into batch
        batch_input = np.stack(batch_images, axis=0)
        return batch_input
    
    def _extract_points_from_onnx_outputs(self, outputs: List[np.ndarray], 
                                         conf_threshold: float) -> List[Point]:
        """
        Extract points from ONNX model outputs.
        
        Args:
            outputs (List[np.ndarray]): ONNX model outputs
            conf_threshold (float): Confidence threshold
            
        Returns:
            List[Point]: Extracted points
        """
        points = []
        
        # This is a simplified extraction - you may need to adapt based on your specific model output format
        # The exact implementation depends on your YOLO model's output structure
        
        try:
            # Assuming the first output contains detection results
            # Format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class_id]
            if len(outputs) > 0:
                detections = outputs[0][0]  # Remove batch dimension
                
                for detection in detections:
                    if len(detection) >= 6:
                        x1, y1, x2, y2, conf, class_id = detection[:6]
                        
                        if conf > conf_threshold:
                            # Calculate center point
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            point = Point(
                                x=float(center_x),
                                y=float(center_y),
                                confidence=float(conf),
                                class_id=int(class_id),
                                class_name=f"object_{int(class_id)}"
                            )
                            points.append(point)
            
            # Handle keypoint outputs if available (for pose models)
            if len(outputs) > 1:
                # Keypoint processing logic would go here
                # This depends on your specific model architecture
                pass
                
        except Exception as e:
            self.logger.warning(f"Error extracting points from ONNX outputs: {str(e)}")
        
        return points
    
    def detect_points(self, image_path: str, conf: float = CONF_THRESHOLD) -> List[Point]:
        """
        Detect points in an image.
        
        Args:
            image_path (str): Path to the image
            conf (float): Confidence threshold
            
        Returns:
            List[Point]: List of detected points
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Construct full image path
        if not os.path.isabs(image_path):
            full_image_path = os.path.join(self.script_dir, image_path)
        else:
            full_image_path = image_path
        
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image not found: {full_image_path}")
        
        try:
            # Run inference
            for i in range(10): 
                results = self.model(full_image_path, conf=conf, imgsz=(INPUT_HEIGHT, INPUT_WIDTH))
            
            # Extract points
            points = self._extract_points_from_results(results)
            self.detected_points = points
            
            self.logger.info(f"Detected {len(points)} points in {full_image_path}")
            return points
            
        except Exception as e:
            self.logger.error(f"Point detection failed: {str(e)}")
            raise
    
    def _extract_points_from_results(self, results) -> List[Point]:
        """Extract points from YOLO results."""
        points = []
        
        for result in results:
            # Handle pose keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                
                for person_idx, person_keypoints in enumerate(keypoints):
                    for kp_idx, (x, y, conf) in enumerate(person_keypoints):
                        if conf > 0.1:  # Minimum confidence for keypoints
                            point = Point(
                                x=float(x),
                                y=float(y),
                                confidence=float(conf),
                                class_id=kp_idx,
                                class_name=self.keypoint_names[kp_idx] if kp_idx < len(self.keypoint_names) else f"keypoint_{kp_idx}"
                            )
                            points.append(point)
            
            # Handle regular object detection as points (center of bounding boxes)
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, class_id = box
                    # Calculate center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    point = Point(
                        x=float(center_x),
                        y=float(center_y),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=f"object_{int(class_id)}"
                    )
                    points.append(point)
        
        return points
    
    def visualize_points(self, image_path: str, points: Optional[List[Point]] = None, 
                        save_path: Optional[str] = None, show_connections: bool = True) -> str:
        """
        Visualize detected points on the image.
        
        Args:
            image_path (str): Path to the original image
            points (List[Point], optional): Points to visualize. Uses self.detected_points if None
            save_path (str, optional): Path to save the visualization
            show_connections (bool): Whether to show connections between keypoints
            
        Returns:
            str: Path to the saved visualization
        """
        if points is None:
            points = self.detected_points
        
        if not points:
            raise ValueError("No points to visualize")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw points
        for i, point in enumerate(points):
            color = self.colors[i % len(self.colors)]
            
            # Draw point
            cv2.circle(vis_image, (int(point.x), int(point.y)), 5, color, -1)
            
            # Add label
            label = f"{point.class_name}: {point.confidence:.2f}"
            cv2.putText(vis_image, label, (int(point.x) + 10, int(point.y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw connections for pose keypoints
        if show_connections:
            self._draw_pose_connections(vis_image, points)
        
        # Save visualization
        if save_path is None:
            save_path = os.path.join(self.output_dir, "visualizations", "points_visualization.jpg")
        
        cv2.imwrite(save_path, vis_image)
        self.logger.info(f"Visualization saved to: {save_path}")
        
        return save_path
    
    def _draw_pose_connections(self, image: np.ndarray, points: List[Point]) -> None:
        """Draw connections between pose keypoints."""
        # Define pose connections (COCO format)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Group points by person (assuming sequential keypoints)
        keypoint_groups = {}
        for point in points:
            if point.class_name in self.keypoint_names:
                person_id = 0  # Simplified: assume single person
                if person_id not in keypoint_groups:
                    keypoint_groups[person_id] = {}
                keypoint_groups[person_id][point.class_id] = point
        
        # Draw connections
        for person_id, keypoints in keypoint_groups.items():
            for start_idx, end_idx in connections:
                if start_idx in keypoints and end_idx in keypoints:
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]
                    
                    cv2.line(image, 
                            (int(start_point.x), int(start_point.y)),
                            (int(end_point.x), int(end_point.y)),
                            (0, 255, 0), 2)
    
    def save_points_to_csv(self, points: Optional[List[Point]] = None, 
                          filename: str = "detected_point_py.csv") -> str:
        """
        Save detected points to CSV file with format: x,y,confidence,class_id,class_name
        
        Args:
            points (List[Point], optional): Points to save
            filename (str): Output filename
            
        Returns:
            str: Path to saved CSV file
        """
        if points is None:
            points = self.detected_points
        
        if not points:
            raise ValueError("No points to save")
        
        # Convert points to DataFrame with specific column order
        data = []
        for point in points:
            data.append({
                'x': point.x,
                'y': point.y,
                'confidence': point.confidence,
                'class_id': point.class_id,
                'class_name': point.class_name
            })
        
        # Create DataFrame with explicit column order
        df = pd.DataFrame(data, columns=['x', 'y', 'confidence', 'class_id', 'class_name'])
        
        # Save to the specified absolute path
        output_dir = "/home/rajabzade/DeepNetC++/output"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, filename)
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Points saved to CSV: {csv_path}")
        return csv_path
    
    def save_points_to_json(self, points: Optional[List[Point]] = None, 
                           filename: str = "detected_points.json") -> str:
        """
        Save detected points to JSON file.
        
        Args:
            points (List[Point], optional): Points to save
            filename (str): Output filename
            
        Returns:
            str: Path to saved JSON file
        """
        if points is None:
            points = self.detected_points
        
        if not points:
            raise ValueError("No points to save")
        
        # Convert points to dictionary
        data = {
            "points": [point.to_dict() for point in points],
            "total_points": len(points),
            "detection_info": {
                "model": self.model_path,
                "confidence_threshold": CONF_THRESHOLD
            }
        }
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, "points", filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Points saved to JSON: {json_path}")
        return json_path
    
    def analyze_points(self, points: Optional[List[Point]] = None) -> Dict[str, Any]:
        """
        Perform statistical analysis on detected points.
        
        Args:
            points (List[Point], optional): Points to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if points is None:
            points = self.detected_points
        
        if not points:
            raise ValueError("No points to analyze")
        
        # Extract data
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        confidences = [p.confidence for p in points]
        class_names = [p.class_name for p in points]
        
        # Calculate statistics
        analysis = {
            "total_points": len(points),
            "coordinates": {
                "x_mean": np.mean(x_coords),
                "x_std": np.std(x_coords),
                "x_min": np.min(x_coords),
                "x_max": np.max(x_coords),
                "y_mean": np.mean(y_coords),
                "y_std": np.std(y_coords),
                "y_min": np.min(y_coords),
                "y_max": np.max(y_coords)
            },
            "confidence": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            },
            "class_distribution": {}
        }
        
        # Class distribution
        unique_classes, counts = np.unique(class_names, return_counts=True)
        for class_name, count in zip(unique_classes, counts):
            analysis["class_distribution"][class_name] = int(count)
        
        return analysis
    
    def create_analysis_plots(self, points: Optional[List[Point]] = None, 
                             save_dir: Optional[str] = None) -> List[str]:
        """
        Create analysis plots for detected points.
        
        Args:
            points (List[Point], optional): Points to analyze
            save_dir (str, optional): Directory to save plots
            
        Returns:
            List[str]: Paths to saved plots
        """
        if points is None:
            points = self.detected_points
        
        if not points:
            raise ValueError("No points to analyze")
        
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "visualizations")
        
        saved_plots = []
        
        # Extract data
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        confidences = [p.confidence for p in points]
        class_names = [p.class_name for p in points]
        
        # 1. Scatter plot of point locations
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_coords, y_coords, c=confidences, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Confidence')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Point Locations with Confidence')
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        
        plot_path = os.path.join(save_dir, "point_locations.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_path)
        
        # 2. Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(save_dir, "confidence_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_path)
        
        # 3. Class distribution
        if len(set(class_names)) > 1:
            plt.figure(figsize=(12, 6))
            class_counts = pd.Series(class_names).value_counts()
            class_counts.plot(kind='bar')
            plt.xlabel('Class Name')
            plt.ylabel('Count')
            plt.title('Point Class Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(save_dir, "class_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)
        
        self.logger.info(f"Analysis plots saved: {saved_plots}")
        return saved_plots
    
    def process_image_complete(self, image_path: str, conf: float = CONF_THRESHOLD,
                              save_csv: bool = True, save_json: bool = True,
                              save_visualization: bool = True, save_analysis: bool = True) -> Dict[str, Any]:
        """
        Complete point detection pipeline.
        
        Args:
            image_path (str): Path to input image
            conf (float): Confidence threshold
            save_csv (bool): Save points to CSV
            save_json (bool): Save points to JSON
            save_visualization (bool): Save visualization
            save_analysis (bool): Save analysis plots
            
        Returns:
            Dict[str, Any]: Complete results
        """
        results = {
            "points": [],
            "analysis": {},
            "files": {
                "csv": None,
                "json": None,
                "visualization": None,
                "analysis_plots": []
            }
        }
        
        try:
            # Detect points
            points = self.detect_points(image_path, conf=conf)
            results["points"] = [point.to_dict() for point in points]
            
            if not points:
                self.logger.warning("No points detected")
                return results
            
            # Save CSV
            if save_csv:
                results["files"]["csv"] = self.save_points_to_csv(points)
            
            # Save JSON
            if save_json:
                results["files"]["json"] = self.save_points_to_json(points)
            
            # Save visualization
            if save_visualization:
                results["files"]["visualization"] = self.visualize_points(image_path, points)
            
            # Perform analysis
            results["analysis"] = self.analyze_points(points)
            
            # Save analysis plots
            if save_analysis:
                results["files"]["analysis_plots"] = self.create_analysis_plots(points)
            
            self.logger.info("Complete point detection pipeline finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Point detection pipeline failed: {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOPointDetector(model_path="../YoloModel/yolov8m_PointDetection_dynamic.pt")
    
    # Example image path (adjust as needed)
    image_path = "../Dataset/input_PointDetection/TestImage.png"
    
    try:
        # Run complete pipeline
        results = detector.process_image_complete(image_path)
        
        print(f"Detected {len(results['points'])} points")
        print(f"Analysis: {results['analysis']}")
        print(f"Files saved: {results['files']}")
        
        # ONNX Conversion Example
        print("\n" + "="*50)
        print("ONNX CONVERSION EXAMPLE")
        print("="*50)
        
        # Convert model to ONNX with dynamic batch support
        onnx_path = detector.convert_to_onnx(
            dynamic_batch=True,
            input_shape=(INPUT_HEIGHT, INPUT_WIDTH),
            verify=True
        )
        print(f"ONNX model saved to: {onnx_path}")
        
        # Load ONNX model for inference
        onnx_session = detector.load_onnx_model(onnx_path)
        
        # Batch processing example
        print("\n" + "="*50)
        print("BATCH PROCESSING EXAMPLE")
        print("="*50)
        
        # Example with multiple images (adjust paths as needed)
        image_batch = [
            "../Dataset/input_PointDetection/TestImage.png",
            # Add more image paths here for batch processing
            # "../Dataset/input_PointDetection/TestImage2.png",
            # "../Dataset/input_PointDetection/TestImage3.png",
        ]
        
        # Filter existing images
        existing_images = [img for img in image_batch if os.path.exists(img)]
        
        if existing_images:
            # Process batch with ONNX
            batch_results = detector.detect_points_onnx(existing_images, onnx_session)
            
            print(f"Processed {len(existing_images)} images in batch")
            for i, points in enumerate(batch_results):
                print(f"Image {i+1}: {len(points)} points detected")
        else:
            print("No valid images found for batch processing")
            print("Please ensure you have images in the specified paths")
        
        # Performance comparison example
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        if existing_images:
            import time
            
            # Time PyTorch inference
            start_time = time.time()
            for img_path in existing_images:
                detector.detect_points(img_path)
            pytorch_time = time.time() - start_time
            
            # Time ONNX inference
            start_time = time.time()
            detector.detect_points_onnx(existing_images, onnx_session)
            onnx_time = time.time() - start_time
            
            print(f"PyTorch inference time: {pytorch_time:.3f}s")
            print(f"ONNX batch inference time: {onnx_time:.3f}s")
            print(f"Speedup: {pytorch_time/onnx_time:.2f}x")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. An image at the specified path")
        print("2. ONNX and ONNXRuntime installed: pip install onnx onnxruntime")
        print("3. For GPU support: pip install onnxruntime-gpu") 
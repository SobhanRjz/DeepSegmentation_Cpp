from ultralytics import YOLO
import pandas as pd
import os
import sys
from typing import Optional, List, Tuple
import logging

# Add parent directory to path to import config_reader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_reader import get_config

# Load configuration
config = get_config()

# Constants from configuration for backward compatibility
CONF_THRESHOLD = config.confidence_threshold
SCORE_THRESHOLD = config.score_threshold
NMS_IOU_THRESHOLD = config.nms_threshold
INPUT_WIDTH = config.input_width
INPUT_HEIGHT = config.input_height


class YOLODetector:
    """
    Professional YOLO object detection class with comprehensive functionality.
    
    This class provides a clean interface for YOLO model operations including
    loading models, running inference, and saving results.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", script_dir: Optional[str] = None):
        """
        Initialize the YOLO detector.
        
        Args:
            model_name (str): Name of the YOLO model to load
            script_dir (str, optional): Directory path. If None, uses current script directory
        """
        self.model_name = model_name
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self.model = None
        self.output_dir = os.path.join(self.script_dir, "output")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the detector
        self._setup_directories()
        self._load_model()
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory created/verified: {self.output_dir}")
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_name)
            self.logger.info(f"Successfully loaded YOLO model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the model to the output directory.
        
        Args:
            filename (str, optional): Custom filename. If None, uses model_name
            
        Returns:
            str: Path where the model was saved
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        filename = filename or self.model_name
        model_save_path = os.path.join(self.output_dir, filename)
        
        try:
            self.model.save(model_save_path)
            self.logger.info(f"Model saved to: {model_save_path}")
            return model_save_path
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def convert_to_onnx(self, filename: Optional[str] = None, imgsz: int = INPUT_WIDTH) -> str:
        """
        Convert the YOLO model to ONNX format.
        
        Args:
            filename (str, optional): Custom filename for ONNX model. If None, uses model_name with .onnx extension
            imgsz (int): Input image size for the ONNX model
            
        Returns:
            str: Path where the ONNX model was saved
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Generate filename if not provided
        if filename is None:
            base_name = os.path.splitext(self.model_name)[0]
            filename = f"{base_name}.onnx"
        
        onnx_save_path = os.path.join(self.output_dir, filename)
        
        try:
            # Export to ONNX format
            self.model.export(format="onnx", imgsz=(INPUT_HEIGHT, INPUT_WIDTH), dynamic=True)
            
            # Move the exported file to output directory if it's not already there
            exported_file = os.path.splitext(self.model_name)[0] + ".onnx"
            if os.path.exists(exported_file) and exported_file != onnx_save_path:
                os.rename(exported_file, onnx_save_path)
            elif not os.path.exists(onnx_save_path):
                # If the file was exported to a different location, find it
                possible_paths = [
                    exported_file,
                    os.path.join(os.getcwd(), exported_file),
                    os.path.join(self.script_dir, exported_file)
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        os.rename(path, onnx_save_path)
                        break
            
            self.logger.info(f"Model converted to ONNX and saved to: {onnx_save_path}")
            return onnx_save_path
        except Exception as e:
            self.logger.error(f"Failed to convert model to ONNX: {str(e)}")
            raise
    
    def detect(self, image_path: str, conf: float = CONF_THRESHOLD) -> List:
        """
        Run object detection on an image.
        
        Args:
            image_path (str): Path to the image file
            conf (float): Confidence threshold for detections
            
        Returns:
            List: Detection results from YOLO
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Construct full image path if relative
        if not os.path.isabs(image_path):
            full_image_path = os.path.join(self.script_dir, image_path)
        else:
            full_image_path = image_path
        
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image not found: {full_image_path}")
        
        try:
            results = self.model(full_image_path, conf=conf, iou=NMS_IOU_THRESHOLD, imgsz=(INPUT_HEIGHT, INPUT_WIDTH))
            self.logger.info(f"Detection completed for: {full_image_path}")
            return results
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            raise
    
    def save_results_to_csv(self, results: List, output_filename: str = "output_py.csv") -> str:
        """
        Save detection results to CSV file.
        
        Args:
            results (List): YOLO detection results
            output_filename (str): Name of the output CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        if not results or len(results) == 0:
            raise ValueError("No results to save")
        
        try:
            # Extract detection data
            df_data = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class]
            
            # Create DataFrame with proper column names
            df = pd.DataFrame(df_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, output_filename)
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Results saved to CSV: {csv_path}")
            return csv_path
        except Exception as e:
            self.logger.error(f"Failed to save results to CSV: {str(e)}")
            raise
    
    def save_annotated_image(self, results: List, output_filename: str = "output_annotated.jpg") -> str:
        """
        Save the image with bounding boxes drawn on it.
        
        Args:
            results (List): YOLO detection results
            output_filename (str): Name of the output image file
            
        Returns:
            str: Path to the saved annotated image
        """
        if not results or len(results) == 0:
            raise ValueError("No results to save")
        
        try:
            # Get the annotated image from results
            annotated_img = results[0].plot()
            
            # Save the annotated image
            image_path = os.path.join(self.output_dir, output_filename)
            
            # Convert from RGB to BGR for OpenCV saving
            import cv2
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, annotated_img_bgr)
            
            self.logger.info(f"Annotated image saved to: {image_path}")
            return image_path
        except Exception as e:
            self.logger.error(f"Failed to save annotated image: {str(e)}")
            raise
    
    def process_image(self, image_name: str = "input/TestImage.jpeg", 
                     save_model: bool = True, save_csv: bool = True, save_image: bool = True,
                     conf: float = CONF_THRESHOLD) -> Tuple[List, Optional[str], Optional[str], Optional[str]]:
        """
        Complete image processing pipeline.
        
        Args:
            image_name (str): Name/path of the image to process
            save_model (bool): Whether to save the model
            save_csv (bool): Whether to save results to CSV
            save_image (bool): Whether to save annotated image with bounding boxes
            conf (float): Confidence threshold for detections
            
        Returns:
            Tuple[List, Optional[str], Optional[str], Optional[str]]: (results, model_path, csv_path, image_path)
        """
        model_path = None
        csv_path = None
        image_path = None
        
        try:
            # Save model if requested
            if save_model:
                model_path = self.save_model()
            
            # Run detection
            results = self.detect(image_name, conf=conf)
            
            # Save results to CSV if requested
            if save_csv:
                csv_path = self.save_results_to_csv(results)
            
            # Save annotated image if requested
            if save_image:
                image_path = self.save_annotated_image(results)
            
            self.logger.info("Image processing completed successfully")
            return results, model_path, csv_path, image_path
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLODetector()
    
    # Process image with default settings
    results, model_path, csv_path, image_path = detector.process_image()
    
    # Convert model to ONNX format
    onnx_path = detector.convert_to_onnx()
    
    print(f"Detection completed. Results saved to: {csv_path}")
    print(f"Annotated image saved to: {image_path}")
    print(f"Model converted to ONNX: {onnx_path}") 
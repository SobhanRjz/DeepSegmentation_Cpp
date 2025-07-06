import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
import sys

# Add parent directory to path to import config_reader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_reader import get_config

# Load configuration
config = get_config()

# Constants from configuration for backward compatibility
CONF_THRESHOLD = config.confidence_threshold
SCORE_THRESHOLD = config.score_threshold
NMS_IOU_THRESHOLD = config.nms_threshold

def draw_detections_from_csv(image_path, csv_path, output_path):
    """Draw bounding boxes from CSV file on image and save it"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Load detections from CSV
    try:
        df = pd.read_csv(csv_path)
    except:
        print(f"Failed to load CSV: {csv_path}")
        return
    
    # COCO class names
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "", "backpack", "umbrella", "", "", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "", "dining table", "", "", "toilet", "", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"
    ]
    
    # Colors for different classes (BGR format)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 165, 0), (0, 128, 0),
        (128, 128, 0)
    ]
    
    # Draw bounding boxes
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        confidence = row['confidence']
        class_id = int(row['class'])
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_id < len(class_names) and class_names[class_id]:
            label = class_names[class_id]
        else:
            label = f"class_{class_id}"
        label += f" {int(confidence * 100)}%"
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save image
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to: {output_path}")

def run_yolo_and_visualize():
    """Run YOLO detection and create visualization"""
    # Load model
    model = YOLO('../YoloModel/yolov8n.pt')
    
    # Run detection using the defined constants with matching NMS IoU threshold
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'input', 'TestImage.jpeg')
    results = model(image_path, conf=CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD, save=False, verbose=False)
    
    # Save results to CSV (rerun to ensure fresh results)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy()
                
                # Apply score threshold filter
                if confidence >= SCORE_THRESHOLD:
                    detections.append([x1, y1, x2, y2, confidence, class_id])
    
    # Save to CSV
    df = pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
    csv_path = 'output/output_py.csv'
    df.to_csv(csv_path, index=False)
    print(f"Python CSV saved to: {csv_path}")
    print(f"Found {len(detections)} detections")
    
    # Create visualization
    draw_detections_from_csv(image_path, csv_path, 'output/detections_py.jpg')

if __name__ == "__main__":
    run_yolo_and_visualize() 
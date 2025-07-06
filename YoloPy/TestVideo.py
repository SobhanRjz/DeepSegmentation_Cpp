import cv2
from ultralytics import YOLO
import time

model = YOLO('YoloModel/yolov8m_SegmentationDetection_dynamic.pt')  # or your custom model
video_path = 'YoloPy/20241216_LiverpolHighRes.mp4'
cap = cv2.VideoCapture(video_path)

Imagepath = 'Dataset/input_SegmentDetection/TestImage.png'

while True:
    # Read the image from the specified path
    frame = cv2.imread(Imagepath)
    

    if frame is None:
        print("❌ Error: Could not read the image.")
        break

    # Inference (get results for this frame)
    t = time.time()
    results = model.predict(frame, conf=0.2, verbose=False)
    end_predict = time.time()
    RealTime = end_predict - t
    print(f"✅ Model predicted in {RealTime:.4f} seconds")


# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Inference (get results for this frame)
#     t = time.time()
#     results = model.predict(frame, conf=0.2, verbose=False)
#     end_predict = time.time()
#     RealTime = end_predict - t
#     print(f"✅ Model predicted in {RealTime:.4f} seconds")






cap.release()
out.release()
cv2.destroyAllWindows()
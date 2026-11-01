import cv2
import torch
from ultralytics import YOLO

# 1. Hardware acceleration detection (Apple Silicon GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Executing system on device: {device}")

# 2. Deploying stable YOLOv8 Small architecture
model = YOLO("yolov8s.pt").to(device)

video_source = "traffic_long.mp4"
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: Could not open video source: {video_source}")
    exit()

# Setup tracking counters and dictionaries for trajectory logic
object_positions = dict()
tracked_ids = set()

car_count = 0
truck_count = 0
bus_count = 0
motorcycle_count = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        height, width, _ = frame.shape
        line_y = int(height * 0.8) # Position line near the bottom (80%)
        
        # 3. Running tracking inference safely
        results_list = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device)
        
        if results_list and len(results_list) > 0:
            res = results_list[0]
            
            # Draw standard bounding boxes and labels automatically
            if hasattr(res, 'plot'):
                annotated_frame = res.plot()
            else:
                annotated_frame = frame
                
            # Internal mathematical logic to capture IDs for counting without attribute errors
            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.int().cpu().tolist()
                track_ids = res.boxes.id.int().cpu().tolist()
                class_ids = res.boxes.cls.int().cpu().tolist()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    y1, y2 = box[1], box[3]
                    center_y = int((y1 + y2) / 2)
                    
                    # Vector crossing verification
                    if track_id in object_positions:
                        prev_center_y = object_positions[track_id]
                        
                        if prev_center_y <= line_y and center_y > line_y:
                            if track_id not in tracked_ids:
                                tracked_ids.add(track_id)
                                
                                # Class IDs mapping based on COCO Dataset
                                if class_id == 2:    # Car
                                    car_count += 1
                                elif class_id == 7:  # Truck
                                    truck_count += 1
                                elif class_id == 5:  # Bus
                                    bus_count += 1
                                elif class_id == 3:  # Motorcycle
                                    motorcycle_count += 1
                                    
                    object_positions[track_id] = center_y
        else:
            annotated_frame = frame
            
        # Draw the bottom counting boundary line (White color)
        cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 255, 255), 3)
        cv2.putText(annotated_frame, "Telemetry Counting Line", (20, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the dynamic telemetry counters on screen
        cv2.putText(annotated_frame, f"Cars: {car_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Trucks: {truck_count}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Buses: {bus_count}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(annotated_frame, f"Motorcycles: {motorcycle_count}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("YOLOv8 Edge Traffic Monitor", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

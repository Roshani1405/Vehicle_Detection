import cv2
from ultralytics import YOLO

# Load YOLO model (replace with your weights if needed)
model = YOLO("wronglane.pt")

# Open webcam (try 0, 1, or 2 depending on your system)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Get frame properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

# Define line positions
line_up = int(height * 0.7)   # upward line at 70% height
line_down = int(height * 0.5) # downward line at 50% height

# Road split boundaries
left_road_max_x = int(width * 0.45)
right_road_min_x = int(width * 0.55)

object_states = {}
up_count, down_count = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue   # retry instead of quitting

    # Run YOLOv8 + ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          conf=0.5, iou=0.4, imgsz=640)  # stricter conf

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id
        classes = r.boxes.cls.cpu().numpy()

        if ids is None:
            continue

        ids = ids.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):
            class_name = model.names[int(cls)]
            if class_name not in ["car", "truck", "bus"]:
                continue

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw bounding box and track ID on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if track_id not in object_states:
                object_states[track_id] = {"up": None, "down": None}

            # Upward line logic
            prev_up = object_states[track_id]["up"]
            current_up = "above" if center_y < line_up else "below"
            if prev_up is None:
                object_states[track_id]["up"] = current_up
            else:
                if prev_up == "above" and current_up == "below" and center_x < left_road_max_x:
                    up_count += 1
                    print(f"UP: Vehicle {track_id} ({class_name}) crossed")
                object_states[track_id]["up"] = current_up

            # Downward line logic
            prev_down = object_states[track_id]["down"]
            current_down = "above" if center_y < line_down else "below"
            if prev_down is None:
                object_states[track_id]["down"] = current_down
            else:
                if prev_down == "below" and current_down == "above" and center_x > right_road_min_x:
                    down_count += 1
                    print(f"DOWN: Vehicle {track_id} ({class_name}) crossed")
                object_states[track_id]["down"] = current_down

    # Show counts in console
    print(f"Up count: {up_count}, Down count: {down_count}")
    
    # Draw reference lines on frame
    cv2.line(frame, (0, line_up), (width, line_up), (255, 0, 0), 2)  # Blue line (upward)
    cv2.line(frame, (0, line_down), (width, line_down), (0, 0, 255), 2)  # Red line (downward)
    
    # Draw counts on frame
    cv2.putText(frame, f"UP: {up_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"DOWN: {down_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    try:
        cv2.imshow("Vehicle Detection", frame)
    except cv2.error:
        pass

    # Press 'q' to quit
    try:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        # If GUI is not available, press Ctrl+C to exit
        pass

cap.release()
cv2.destroyAllWindows()


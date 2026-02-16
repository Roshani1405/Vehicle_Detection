import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("wronglane.pt")  # use your trained weights

# Open webcam (0 = default camera, try 1 or 2 if you have multiple)
cap = cv2.VideoCapture(0)

# Get frame properties (fallbacks if not available)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Video writer (optional: saves processed output)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# Define line positions
line_up = int(height * 0.7)   # upward line at 70% height
line_down = int(height * 0.5) # downward line at 50% height

# Road split boundaries
left_road_max_x = int(width * 0.45)
right_road_min_x = int(width * 0.55)

object_states = {}
up_count, down_count = 0, 0
up_ids, down_ids = set(), set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received from webcam")
        break

    # Run YOLOv8 + ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          conf=0.05, iou=0.4, imgsz=1280)

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

            # Draw bounding box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0,0,255), -1)
            cv2.putText(frame, f"ID {track_id} {class_name}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

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
                    up_ids.add(track_id)
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
                    down_ids.add(track_id)
                    print(f"DOWN: Vehicle {track_id} ({class_name}) crossed")
                object_states[track_id]["down"] = current_down

    # Draw partial lines
    cv2.line(frame, (0, line_up), (left_road_max_x, line_up), (0,255,0), 3)
    cv2.line(frame, (right_road_min_x, line_down), (width, line_down), (0,0,255), 3)

    # Show counts
    cv2.putText(frame, f"Up: {up_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Down: {down_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    out.write(frame)
    cv2.imshow("Vehicle Detection (Webcam)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

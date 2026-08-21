import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("test_traffic_3.mp4")
counted_ids = set()
class_counts = {"car": 0, "bus": 0, "bike": 0, "truck": 0}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))
    height, width, _ = frame.shape

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False)
    
    right_line_y = int(height * 0.6)
    right_x1, right_x2 = int(width * 0.52), int(width * 0.95)

    left_line_y = int(height * 0.6)
    left_x1, left_x2 = int(width * 0.05), int(width * 0.48)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = y2  

            class_name = model.names.get(cls_id, "vehicle")

            is_right = (right_x1 <= center_x <= right_x2) and (abs(center_y - right_line_y) <= 8)
            is_left = (left_x1 <= center_x <= left_x2) and (abs(center_y - left_line_y) <= 8)

            if is_right or is_left:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts["car"] = class_counts.get("car", 0) + 1

    cv2.line(frame, (right_x1, right_line_y), (right_x2, right_line_y), (0, 0, 255), 3)
    cv2.line(frame, (left_x1, left_line_y), (left_x2, left_line_y), (255, 0, 0), 3)

    total_vehicles = sum(class_counts.values())
    cv2.putText(frame, f"Total: {total_vehicles}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cars: {class_counts.get('car', 0)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Buses: {class_counts.get('bus', 0)}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Bikes: {class_counts.get('bike', 0)}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Trucks: {class_counts.get('truck', 0)}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Traffic Police - Part 1", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

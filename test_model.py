from ultralytics import YOLO
import cv2
import math

# Load trained model
model = YOLO(r"D:\Pycharm\dog_driveaway.v3\runs\detect\train6\weights\best.pt")

# COCO classes
PERSON = 0
DOG = 16

# Distance threshold
DIST_THRESHOLD = 150

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    persons = []
    dogs = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls == PERSON:
                persons.append((cx, cy))
                label = f"person {conf:.2f}"
                color = (255, 0, 0)

            elif cls == DOG:
                dogs.append((cx, cy))
                label = f"dog {conf:.2f}"
                color = (0, 255, 0)

            else:
                continue  # ❌ ignore all other classes

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 🔥 Behavior logic (fake but useful)
    threat = "CALM"

    for dx, dy in dogs:
        for px, py in persons:
            dist = math.sqrt((dx - px)**2 + (dy - py)**2)

            if dist < DIST_THRESHOLD:
                threat = "POTENTIAL THREAT"

    # Display behavior
    cv2.putText(frame, f"Behavior: {threat}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("YOLO DOG + HUMAN ONLY", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
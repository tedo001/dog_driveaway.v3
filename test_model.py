from ultralytics import YOLO
import cv2

# Load your trained model (IMPORTANT: use full path)
model = YOLO(r"D:\Pycharm\dog_driveaway.v3\runs\detect\train6\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, conf=0.5)

    # Draw results
    annotated_frame = results[0].plot()

    # Show
    cv2.imshow("YOLO Detection (DOG + PERSON)", annotated_frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
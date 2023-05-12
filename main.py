import cv2
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture('./pen.mp4')

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        if cls == 0:    
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, "Player", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        if cls == 32:
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, "Ball", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    tl = (250, 1070)
    bl = (1450, 1400)
    tr = (1350, 760)
    br = (2650, 920)

    cv2.circle(frame, tl, 5, (255, 0, 0), -1)
    cv2.circle(frame, bl, 5, (255, 0, 0), -1)
    cv2.circle(frame, tr, 5, (255, 0, 0), -1)
    cv2.circle(frame, br, 5, (255, 0, 0), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 1570], [2912, 0], [2912, 1570]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    cv2.imshow("Img", frame)
    cv2.imshow("transformed_frame Bird's Eye View", transformed_frame)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Img", cv2.WND_PROP_VISIBLE) < 1:
        break

        
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
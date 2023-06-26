import cv2
import numpy as np
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture('./data/pen.mp4')

model = YOLO("yolov8m.pt")
yolo_seg = YOLOSegmentation("yolov8m-seg.pt")

def get_average_color(a):
    return tuple(np.array(a).mean(axis=0).mean(axis=0).round().astype(int))

# identifies most common color
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gs, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    penalty_box_lines = []
    semicircle_points = []

    for line in lines:
        if len(line) == 4:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, dx, dy = line[0]
            x2, y2 = x1 + dx, y1 + dy

        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

        if 80 <= angle <= 100 and y1 > frame.shape[0] / 2:
            penalty_box_lines.append(line)
        elif angle == 0 and x1 > frame.shape[1] / 2:
            semicircle_points.append((x1, y1))
            semicircle_points.append((x2, y2))

    for line in penalty_box_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(semicircle_points) == 2:
        center_x = int((semicircle_points[0][0] + semicircle_points[1][0]) / 2)
        center_y = int((semicircle_points[0][1] + semicircle_points[1][1]) / 2)
        radius = int(np.sqrt((semicircle_points[0][0] - semicircle_points[1][0]) ** 2 +
                             (semicircle_points[0][1] - semicircle_points[1][1]) ** 2) / 2)

        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

    cv2.imshow("Img", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

        
cap.release()
cv2.destroyAllWindows()
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

    B = frame[:,:,2]
    Y = 255-B

    thresh = cv2.adaptiveThreshold(Y,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV,35,5)

    contours, hierarchy = cv2.findContours(thresh,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    x=[]
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) > 100:
            x.append(contours[i])
    cv2.drawContours(frame, x, -1, (255,0,0), 2)

    cv2.imshow("Img", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

        
cap.release()
cv2.destroyAllWindows()
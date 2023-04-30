#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import torch
from ultralytics import YOLO


# In[4]:


print(torch.backends.mps.is_available())


# In[5]:


cap = cv2.VideoCapture('./pen.mov')


# In[6]:


model = YOLO("yolov8m.pt")


# In[7]:


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)



    cv2.imshow("Img", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

        
cap.release()
cv2.destroyAllWindows()


# Penalty Kick Encroachment Tracker ⚽️

### 🏁 Automatically Detect Encroachment in Penalty Kicks!
Penalty Kicks are one of football's most exciting and important moments, which is why it is so important that the referee gets the call right. This project aims to leverage Computer Vision and Machine Learning to automatically detect encroachment during penalty kicks.
![pen1](https://github.com/AggieSportsAnalytics/PenaltyEncroachment/assets/53201392/9e4bd4b5-d1fd-4981-abfd-9a23f5ee57aa)
_Our starting point_


# What is encroachment?
Encroachment is when players (attackers or defenders) enter the box before the ball is kicked by the penalty taker.

# Why is this a problem?
This is a problem because if undetected, players who enter the box prematurely will gain an unfair advantage reaching the loose ball.

# Why is this project needed?
The inspiration for this project came when undetected encroachment cost Borussia Dortmund the game against Chelsea in the 2022-2023 Champions League. Referees often overlook encroachment while watching the goalkeepers and seeing if they stay on their line. Such an autonomous tool could create a more authentic refereeing experience, also eliminating human error.

# 🔑 Key Features
## Player and Ball Tracking
The project employs object detection and tracking algorithms to identify and track the positions of players on the field throughout the Penalty sequence.
![pen2](https://github.com/AggieSportsAnalytics/PenaltyEncroachment/assets/53201392/0dd5556d-cc3f-45b7-bbbd-9edb59560d64)
### 💻 Code
For the intial player detection, we utilized a pre trained YOLOv8 model

```py
from ultralytics import YOLO
# import model from pretrained weights file
model = YOLO("yolov8m.pt")

# read a frame from video
ret, frame = cap.read()

# run model on the frame
results = model(frame, device="mps")
```

Then we need to extract the information we want from the model's results, which will be the bounding boxes and the classes

```py
bounding_boxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
classes = np.array(result.boxes.cls.cpu(), dtype="int")
```

Now, we need to label the frame with the information we gathered

```py
for cls, bbox in zip(classes, bounding_boxes):
        # label players
        if cls == 0:
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, "Player", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        # label ball
        if cls == 32:
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, "Ball", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
```

## Segmentation and Team Classification
The program also detects players more finely and additionally, and classifies them based on which team they belong to.
![pen3](https://github.com/AggieSportsAnalytics/PenaltyEncroachment/assets/53201392/91d2b6e8-7f2f-4336-a6a3-46b6c7ec145b)
_Notice the finer bounding boxes. The player's jerseys RGB value is displayed above them, and colored in that same value_
### 💻 Code
We can now use YOLO Segmentation to acquire finer bounding boxes for the players and then, using the more precise segmentation, an algorithm to identify which team each player belongs to.
First, let's implement segmentation:

```py
# import yolo segmentation model and use pre trained weights file
from yolo_segmentation import YOLOSegmentation
yolo_seg = YOLOSegmentation("yolov8m-seg.pt")

# bounding_boxes - the bounding boxes
# classes - the object classes
# segementation - the values to segment within the boxes
# scores - the confidence score of the detection
bounding_boxes, classes, segmentations, scores = yolo_seg.detect(frame)
```

Now let's create a function to pull the most dominant color from within a box

```py
def get_average_color(a):
    return tuple(np.array(a).mean(axis=0).mean(axis=0).round().astype(int))
```

Finally, lets put our loop together to annotate the segmentations and common color!
```py
for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        if class_id == 0:
            (x, y, x2, y2) = bbox
            
            minY = np.max(seg[:, 1])
            bottomVal = int(2*(minY - seg[0][1])/3 + seg[0][1])
            
            a = frame2[seg[0][1]:bottomVal, seg[0][0]:seg[len(seg)-1][0]]

            cv2.polylines(frame, [seg], True, (0, 0, 225), 2)
            cv2.rectangle(frame, (seg[0][0], seg[0][1]), (seg[len(seg)-1][0], bottomVal), (225, 0, 0), 2)
            cv2.putText(frame, str(get_average_color(a)), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (int(get_average_color(a)[0]), int(get_average_color(a)[1]), int(get_average_color(a)[2])), 4)
```



## 🚀 Further Uses
- Goalkeeper checking: Eventually, the project can also be extended to track goalkeepers and making sure that they stay on their line.
- Player Jersey Number Recognition: The system could later utilizes Optical Character Recognition (OCR) techniques to read the jersey numbers of players on the field. This allows the identification of the offending player.

## 💻  Technology
- Ultralytics
- OpenCV
- NumPy
- YoloV8 / YoloSegmentation

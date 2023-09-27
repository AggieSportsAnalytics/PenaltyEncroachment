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
bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
classes = np.array(result.boxes.cls.cpu(), dtype="int")
```

Now, we need to label the frame with the information we gathered

```py
for cls, bbox in zip(classes, bboxes):
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
We can now use YOLO Segmentation to acquire finer bounding boxes for the players and then, using the more precise segmentation, an algorithm to identify which team each player belongs to



## 🚀 Further Uses
- Goalkeeper checking: Eventually, the project can also be extended to track goalkeepers and making sure that they stay on their line.
- Player Jersey Number Recognition: The system could later utilizes Optical Character Recognition (OCR) techniques to read the jersey numbers of players on the field. This allows the identification of the offending player.

## 💻  Technology
- Ultralytics
- OpenCV
- NumPy
- YoloV8 / YoloSegmentation

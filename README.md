# Penalty Kick Encroachment Tracker ⚽️

### 🏁 Automatically Detect Encroachment in Penalty Kicks!

# What is encroachment?
Encroachment is when players (attackers or defenders) enter the box before the ball is kicked by the penalty taker.

# Why is this a problem?
This is a problem because if undetected, players who enter the box prematurely will gain an unfair advantage reaching the loose ball.

# Why is this project needed?
The inspiration for this project came when undetected encroachment cost Borussia Dortmund the game against Chelsea in the 2022-2023 Champions League. Referees often overlook encroachment while watching the goalkeepers and seeing if they stay on their line. Such an autonomous tool could create a more authentic refereeing experience, also eliminating human error.

## 🔑 Key Features
- Player Tracking: The project employs object detection and tracking algorithms to identify and track the positions of players on the field throughout the Penalty sequence.
- Box Detection: The project utilizes OpenCV contours, canny, and hough line methods to accurately detect the box, even through persepctive interference (like players standing over the box)
- Reliability: The project emphasizes achieving high accuracy and reliability in offside decisions. Once completed, it should deliver high accuracy detections of encroachment.
- Real-Time Video Analysis: On a system with enough compute, the program will be able to process and run it's algorithms on live footage, providing real-time assistant refereeing to the game.

## 🚀 Further Uses
- Goalkeeper checking: Eventually, the project can also be extended to track goalkeepers and making sure that they stay on their line.
- Player Jersey Number Recognition: The system could later utilizes Optical Character Recognition (OCR) techniques to read the jersey numbers of players on the field. This allows the identification of the offending player.

## 💻  Technology
- Ultralytics
- OpenCV
- NumPy
- YoloV8 / YoloSegmentation

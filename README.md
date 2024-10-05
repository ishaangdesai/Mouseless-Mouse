# Mouseless-Mouse

## Overview
This project implements Mediapipe's Hand Landmarks Detection model to track hand movements using a livestream from a webcam. It then extracts the x, y coordinates of 21 hand landmarks, then uses this data with a custom trained convolutional neural network to:

- Control the mouse pointer movements based on the movement of the hand.
- Perform left and right mouse clicks based on the gesture of the hand.

## Features
- Real-time hand tracking: Detects hand landmarks using Mediapipe and captures x, y coordinates.
- Data recording: Save hand landmark data to CSV with a key press, for use in training or analysis.
- Mouse control: Use convolutional neural network with hand gestures to control mouse movements and clicks.
- Gesture mapping: You can customize which hand movements trigger specific which mouse actions by collecting new data and retraining the model.

## Project Structure
### CSV Files
- generalTrainingData.csv
- leftClickTrainingData.csv
- rightClickTrainingData.csv
- scrollingTrainingData.csv
All CSV files are created by trainingDataCollection.py, which stores hand landmark data in them based on which key is pressed

### Models
- clickIdentification.h5
- hand_landmarker.task
clickIdentification.h5 is the convolutional neural network made to input the CSV files and detect which gesture is happening
<br/>hand_landmarker.task is mediapipe's model for finding hand landmarks

### Model Training
- clickModelTraining.py
- trainingDataCollection.py
clickModelTraining.py trains a Convolutional Neural Network to convert landmark inputs to gesture outputs.
<br/> trainingDataCollection.py uses keyboard inputs to save hand landmark data in their respective CSV files

### Testing
- handTrackingTesting.py
handTrackingTesting.py is to test the camera to see if it can detect the hand from it's current position

### Implimentation
 - mouselessMouse.py
mouselessMouse.py is the code that actually controls the mouse

## Installation
1. Clone this repository
2. Install all dependencies
Main ones:
 - tensorflow
 - pyautogui
 - mediapipe
 - cv2
 - pandas
3. Set up the camera so it looks down at a flat, plain colored surface. It should be around 2 feet off the surface
<br/>  ![image](https://github.com/user-attachments/assets/7a3efed7-a2e7-4d0d-ad2a-49ec508a7eff)
4. Run handTrackingTesting.py, and make sure the code properly tracks your hand
5. Run mouselessMousepy to use the mouse

## Usage
As of now, this code is used to attempt to replace a traditional mouse. Using the following gestures, different actions will be performed.
<br/>Left Click:
<br/> ![image](https://github.com/user-attachments/assets/5511ffee-96b0-416f-9bce-295de2ad6e04)
<br/> push thumb under palm

Right Click:
<br/> ![image](https://github.com/user-attachments/assets/56782e64-c683-4f09-b4a5-086ce580a483)
<br/> bend pointer finger down

Moving around normally:
<br/> ![image](https://github.com/user-attachments/assets/cf3481e5-da3a-4a6c-9374-de8aedf182ff)
<br/> keep hand relaxed

## Dataset
To identify landmarks, a pretrained model from mediapipe is used, no datasets are used.
<br/>To identify gestures, a custom data set is created for each gestures using trainingDataCollection.py

## Model Training and Evaluation
<img width="374" alt="image" src="https://github.com/user-attachments/assets/5db523e6-c0aa-4c6e-9c2f-fbc938391ed8">
<br/> I chose to use a CNN model since they're good at identifying patterns in images, and although I'm not using the full image, it would still be good for identifying relationships between landmarks.
<br/><img width="553" alt="image" src="https://github.com/user-attachments/assets/cd0ebbc4-fb0f-4965-87f8-8bc8abbc2a7c">
<br/> After 100 training epochs, it was able to identify accurately 100% of the time on a 30-70 test train split. 100% accuracy is essential for a mouse, because it would be extremely bad if it clicked somewhere at the wrong time.

## Results
The following video shows the mouse in action.
[Video Link](https://www.youtube.com/watch?v=o9eTms-SIuE)

## Future Improvements:
### Increased Mouse Accuracy
 - Current model shakes between frames
 - Filter to remove this, but also prevents precise hand movements
 - Need to somehow remove jitter or differentiate between jittering and small hand movements
### Scrolling:
 - Already have training data for it, just need to impliment in the current model or train a new one
 - Will make mouse useable in normal situations
### Usability:
 - Current setup required is very precise, doesn't work without it
 - Make it less setup dependent so it can be practically used in different situations

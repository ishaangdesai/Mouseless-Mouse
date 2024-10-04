#Imports
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import pandas as pd
import pyautogui
import time

pyautogui.PAUSE = 0.001

clickModel=tf.keras.models.load_model('clickIdentification.h5')

#Function to draw the landmarks given an input image and hand landmarker data 
MARGIN = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
num=0


right_click_data = pd.read_csv('rightClickTrainingData.csv')
left_click_data = pd.read_csv('leftClickTrainingData.csv')
general_data = pd.read_csv('generalTrainingData.csv')

# Extracting x and y values
X_right = right_click_data.values[:, :42]  # First 42 columns as features

X_left = left_click_data.values[:, :42]

X_general = general_data.values[:, :42]

# Combine datasets
Xvalues = np.vstack((X_right, X_left, X_general))


def landmarksToModelData(landmarks):
    coordinates=[]
    for i in landmarks:
        coordinates.append(i.x)
        coordinates.append(i.y)
    #print(coordinates)
    coordinates=np.array(coordinates)
    coordinates = (coordinates - np.mean(Xvalues, axis=0)) / np.std(Xvalues, axis=0)

    coordinates = coordinates.reshape(1, 21, 2, 1)  # Shape: (1, 21, 2, 1)

    # Check the shape
    #print(coordinates.shape)
    return coordinates


def smoothMouseGlide(dx, dy):
   # Define start and end positions
  steps = 10  # Number of steps in the movement
  x = dx/steps
  y = dy/steps
# Move in small increments for smoother movement
  for i in range(steps):
    pyautogui.move(x, y)



hand_landmarks=[]
#Creating a hand landmarker object from mediapipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.01, min_hand_presence_confidence=0.05 )
detector = vision.HandLandmarker.create_from_options(options)



key=''
x_sensitivity=8000
y_sensitivity=5000
dx=0
dy=0
oldx=0
oldy=0
previous_click = ''

# Initialize the camera 
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
iterations=0
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Display the live ∏
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


    #detecting landmarks given image
    detection_result = detector.detect(mp_image)

    
    #visualizing results using previous function
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    key = cv2.waitKey(1) & 0xFF  # Get key press (1 ms delay)

    # Press 'q' to exit the live stream
    if key == ord('q'):
        break
    # If no hand is detected, skip the frame
    if detection_result.handedness == []:
       continue
    #Add the hand landmarks to a list
    hand_landmarks.append(detection_result.hand_landmarks)
    current_landmarks = detection_result.hand_landmarks[0]
    if iterations==0:
       oldx=current_landmarks[0].x
       oldy=current_landmarks[0].y
       iterations+=1
    dx=(oldx-current_landmarks[0].x)*x_sensitivity
    dy=(oldy-current_landmarks[0].y)*y_sensitivity
    print(dx)
    print(dy)
    oldx=current_landmarks[0].x
    oldy=current_landmarks[0].y

    current_landmarks=landmarksToModelData(current_landmarks)
    click_prediction=clickModel.predict(current_landmarks)

    #print(prediction[0][0])
    #print(prediction[0][1])
    #print(prediction[0][2])
    predicted=''

    
    if abs(dx)<30 and abs(dy)<30 or abs(dx)>500 or abs(dy)>500:
      pyautogui.move(0, 0)
    elif abs(dy)<50:
      smoothMouseGlide(dx, 0)
    elif abs(dx)<50:
      smoothMouseGlide(0, dy)
    else:
      smoothMouseGlide(dx, dy)



    #checking if the model is sure enough about any of the clicks, but also making sure it isnt the same action as the last frame processed
    #if the action is the same as the last frame procecced it simply skips it to increase speed
    if click_prediction[0][0] > 0.5 and previous_click != 'right':
      pyautogui.mouseDown(button="right")
      previous_click = 'right'
      print('right')
    elif click_prediction[0][1] > 0.4 and previous_click != 'left':
      pyautogui.mouseDown(button="left")
      print('left')
      previous_click = 'left'
    elif click_prediction[0][2] > 0.4 and previous_click != 'general':
      pyautogui.mouseUp(button="right")
      pyautogui.mouseUp(button="left")
      print('none')
      previous_click = 'general' 
    print(previous_click)
    cv2.putText(annotated_image, previous_click, (50, 50), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 1)
    cv2.imshow('hand', annotated_image)

    

#end capture when q is pressed
cap.release()
cv2.destroyAllWindows()



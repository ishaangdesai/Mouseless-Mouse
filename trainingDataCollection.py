#Imports
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv


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

#function to clear and initialize each csv file to properly format the data
def initCSV():
  headers=[]
  for i in range(1, 22):  # Start numbering from 1
      headers.append(f'x{i}')
      headers.append(f'y{i}')
  # Open the file for appending
  with open("/Users/ishaandesai/Desktop/MouselessMouse/leftClickTrainingData.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
  with open("/Users/ishaandesai/Desktop/MouselessMouse/rightClickTrainingData.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
  with open("/Users/ishaandesai/Desktop/MouselessMouse/scrollingTrainingData.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
  with open("/Users/ishaandesai/Desktop/MouselessMouse/generalTrainingData.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
        
        








#function to transform and add data to csv 
def dataToCsv(fileName, data):
  landmarks_dict = {}
    
  #looping through all 23 landmarks and adding them to a dictionary by numberp
  for i, landmark in enumerate(data, start=1):  # Start numbering from 1
      landmarks_dict[f'x{i}'] = landmark.x
      landmarks_dict[f'y{i}'] = landmark.y
  # Open the file for appending
  with open(fileName, mode='a', newline='') as file:
      writer = csv.DictWriter(file, fieldnames=landmarks_dict.keys())
      writer.writerow(landmarks_dict)
  print("written")














initCSV()

handLandmarks=[]
#Creating a hand landmarker object from mediapipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.01, min_hand_presence_confidence=0.05 )
detector = vision.HandLandmarker.create_from_options(options)



key=''

# Initialize the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

tempLandmarkDict=[]
dataSaved=False
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Display the live stream
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


    #detecting landmarks given image
    detection_result = detector.detect(mp_image)

    
    #visualizing results using previous function
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow('hand', annotated_image)
    key = cv2.waitKey(1) & 0xFF  # Get key press (1 ms delay)

    # Press 'q' to exit the live stream
    if key == ord('q'):
        break
    # If no hand is detected, skip the frame
    if detection_result.handedness == []:
       continue
    #Add the hand landmarks to a list
    handLandmarks.append(detection_result.hand_landmarks)
    currentLandmarks = detection_result.hand_landmarks[0]


    # Save landmarks based on key presses
    if key == ord('l'):
        dataToCsv("/Users/ishaandesai/Desktop/MouselessMouse/leftClickTrainingData.csv", currentLandmarks)
        num += 1
        print(num)

    if key == ord('r'):
      dataToCsv("/Users/ishaandesai/Desktop/MouselessMouse/rightClickTrainingData.csv", currentLandmarks)
      num += 1
      print(num)

    if key == ord('s'):
      dataToCsv("/Users/ishaandesai/Desktop/MouselessMouse/scrollingTrainingData.csv", currentLandmarks)
      num += 1
      print(num)
    if key == ord('g'):
      dataToCsv("/Users/ishaandesai/Desktop/MouselessMouse/generalTrainingData.csv", currentLandmarks)
      num += 1
      print(num)
         

        
    

#end capture when q is pressed
cap.release()
cv2.destroyAllWindows()



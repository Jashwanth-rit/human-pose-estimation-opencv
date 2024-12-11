# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH


#implimenting webcam to human detection.

# import cv2 as cv
# import numpy as np
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
# parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
# parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
# parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

# args = parser.parse_args()

# BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# inWidth = args.width
# inHeight = args.height

# net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# cap = cv.VideoCapture(args.input if args.input else 0)

# while cv.waitKey(1) < 0:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         cv.waitKey()
#         break

#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
    
#     net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#     out = net.forward()
#     out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

#     assert(len(BODY_PARTS) == out.shape[1])

#     points = []
#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponging body's part.
#         heatMap = out[0, i, :, :]

#         # Originally, we try to find all the local maximums. To simplify a sample
#         # we just find a global one. However only a single pose at the same time
#         # could be detected this way.
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         # Add a point if it's confidence is higher than threshold.
#         points.append((int(x), int(y)) if conf > args.thr else None)

#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert(partFrom in BODY_PARTS)
#         assert(partTo in BODY_PARTS)

#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]

#         if points[idFrom] and points[idTo]:
#             cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

#     t, _ = net.getPerfProfile()
#     freq = cv.getTickFrequency() / 1000
#     cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

#     cv.imshow('OpenPose using OpenCV', frame)


# using vedios and photos for human detection .

# Import the necessary libraries
# import cv2 as cv
# import numpy as np


# # Define a dictionary mapping human body parts to their corresponding indices in the model's output
# BODY_PARTS = {
#     "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#     "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#     "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#     "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
# }

# # Define a list of pairs representing the body parts that should be connected to visualize the pose
# POSE_PAIRS = [
#     ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#     ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#     ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#     ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#     ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
# ]

# # Specify the input dimensions for the neural network
# width = 368
# height = 368
# inWidth = width
# inHeight = height

# # Load the pre-trained OpenPose model from a file
# net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
# thr = 0.2  # Set a confidence threshold for detecting keypoints

# # Define a function to detect poses in an input frame
# def poseDetector(frame):
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]

#     # Prepare the input for the model by resizing and mean normalization
#     net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#     out = net.forward()
#     out = out[:, :19, :, :]  # Extract the first 19 elements, corresponding to the body part keypoints

#     # Ensure the number of detected body parts matches the predefined BODY_PARTS
#     assert(len(BODY_PARTS) == out.shape[1])

#     points = []  # Initialize a list to hold the detected keypoints
#     # Iterate over each body part to find the keypoints
#     for i in range(len(BODY_PARTS)):
#         # Extract the heatmap for the current body part
#         heatMap = out[0, i, :, :]
#         # Find the point with the maximum confidence
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         # Scale the point's coordinates back to the original frame size
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         # Add the point to the list if its confidence is above the threshold
#         points.append((int(x), int(y)) if conf > thr else None)

#     # Draw lines and ellipses to represent the pose in the frame
#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         # Ensure the body parts are in the BODY_PARTS dictionary
#         assert(partFrom in BODY_PARTS)
#         assert(partTo in BODY_PARTS)

#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]

#         # If both keypoints are detected, draw the line and keypoints
#         if points[idFrom] and points[idTo]:
#             cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

#     t, _ = net.getPerfProfile()  # Optional: Retrieve the network's performance profile

#     return frame  # Return the frame with the pose drawn

# # Load an input image
# input = cv.imread("./image2.jpg")
# # Pass the image to the poseDetector function
# output = poseDetector(input)
# # Display the output image with the detected pose
# cv.imshow("Pose Detection", output)
# cv.waitKey(0)  # Wait for a key press to close the window
# cv.destroyAllWindows()
# # cv2_imshow(output)



# Implimneting both Human and fall detection in single one using vedio.

import cv2 as cv
import cvzone
import numpy as np
import math
from ultralytics import YOLO

# OpenPose Parameters
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Load models
pose_net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
fall_model = YOLO("yolov8s.pt")

# Load class names
classnames = []
with open("classes.txt", "r") as f:
    classnames = f.read().splitlines()

# Video input (replace 'fall.mp4' with '0' for webcam)
cap = cv.VideoCapture("fall1.mp4")

# Pose detection parameters
pose_thr = 0.2
inWidth, inHeight = 368, 368

# Frame skipping parameters
skip_frames = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue  # Skip this frame

    frame = cv.resize(frame, (640, 480))  # Reduce resolution

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # ------------------ OpenPose Detection ------------------ #
    pose_net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    pose_out = pose_net.forward()
    pose_out = pose_out[:, :19, :, :]  # Extract relevant outputs

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = pose_out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / pose_out.shape[3]
        y = (frameHeight * point[1]) / pose_out.shape[2]
        points.append((int(x), int(y)) if conf > pose_thr else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        if points[BODY_PARTS[partFrom]] and points[BODY_PARTS[partTo]]:
            cv.line(frame, points[BODY_PARTS[partFrom]], points[BODY_PARTS[partTo]], (0, 255, 0), 3)
            cv.ellipse(frame, points[BODY_PARTS[partFrom]], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[BODY_PARTS[partTo]], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # ------------------ Fall Detection ------------------ #
    results = fall_model(frame)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_detect_name = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect_name == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect_name}', [x1 + 8, y1 - 12], thickness=2, scale=2)
                if threshold < 0:
                    cvzone.putTextRect(frame, "Fall Detected", [x1, y1 - 30], thickness=2, scale=2)

    # Display frame
    cv.imshow("Combined Pose and Fall Detection", frame)

    # Exit on pressing 't'
    if cv.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv.destroyAllWindows()

# Implimenting the abow for webcam 

# import cv2 as cv
# import cvzone
# import numpy as np
# import math
# from ultralytics import YOLO

# # OpenPose Parameters
# BODY_PARTS = {
#     "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#     "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#     "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#     "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
# }

# POSE_PAIRS = [
#     ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#     ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#     ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#     ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#     ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
# ]

# # Load models
# pose_net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
# fall_model = YOLO("yolov8s.pt")

# # Load class names
# classnames = []
# with open("classes.txt", "r") as f:
#     classnames = f.read().splitlines()

# # Video input (use '0' for webcam)
# cap = cv.VideoCapture(0)  # Webcam input

# # Pose detection parameters
# pose_thr = 0.2
# inWidth, inHeight = 368, 368

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break  # Exit if no frame is captured
    
#     frame = cv.resize(frame, (980, 740))
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]

#     # ------------------ OpenPose Detection ------------------ #
#     pose_net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#     pose_out = pose_net.forward()
#     pose_out = pose_out[:, :19, :, :]  # Extract relevant outputs

#     points = []
#     for i in range(len(BODY_PARTS)):
#         heatMap = pose_out[0, i, :, :]
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / pose_out.shape[3]
#         y = (frameHeight * point[1]) / pose_out.shape[2]
#         points.append((int(x), int(y)) if conf > pose_thr else None)

#     for pair in POSE_PAIRS:
#         partFrom, partTo = pair
#         if points[BODY_PARTS[partFrom]] and points[BODY_PARTS[partTo]]:
#             cv.line(frame, points[BODY_PARTS[partFrom]], points[BODY_PARTS[partTo]], (0, 255, 0), 3)
#             cv.ellipse(frame, points[BODY_PARTS[partFrom]], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[BODY_PARTS[partTo]], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

#     # ------------------ Fall Detection ------------------ #
#     results = fall_model(frame)
#     for info in results:
#         parameters = info.boxes
#         for box in parameters:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             confidence = box.conf[0]
#             class_detect = int(box.cls[0])
#             class_detect_name = classnames[class_detect]
#             conf = math.ceil(confidence * 100)

#             height = y2 - y1
#             width = x2 - x1
#             threshold = height - width

#             if conf > 80 and class_detect_name == 'person':
#                 cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
#                 cvzone.putTextRect(frame, f'{class_detect_name}', [x1 + 8, y1 - 12], thickness=2, scale=2)
#                 if threshold < 0:
#                     cvzone.putTextRect(frame, "Fall Detected", [x1, y1 - 30], thickness=2, scale=2)

#     # Display frame
#     cv.imshow("Combined Pose and Fall Detection", frame)

#     # Exit on pressing 't'
#     if cv.waitKey(1) & 0xFF == ord('t'):
#         break

# cap.release()
# cv.destroyAllWindows()


# implimentation to find all the itwm visible to cemara 

# import cv2 as cv
# import cvzone
# import numpy as np
# import os
# import face_recognition
# from ultralytics import YOLO
# import sqlite3
# import time

# # Load YOLO model
# human_model = YOLO("yolov8s.pt")

# # Load class names
# classnames = []
# with open("classes.txt", "r") as f:
#     classnames = f.read().splitlines()

# # Initialize database
# db_conn = sqlite3.connect('human_tracking.db')
# cursor = db_conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS HumanData (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     name TEXT,
#     photo_path TEXT,
#     appearances INTEGER DEFAULT 1,
#     last_seen TIMESTAMP
# )''')
# db_conn.commit()

# # Video input (use '0' for webcam)
# cap = cv.VideoCapture(0)

# def add_human_to_db(name, face_img):
#     """Add new human data to the database."""
#     photo_path = f'photos/{name}_{int(time.time())}.jpg'
#     os.makedirs("photos", exist_ok=True)
#     cv.imwrite(photo_path, face_img)
#     cursor.execute("INSERT INTO HumanData (name, photo_path, last_seen) VALUES (?, ?, ?)",
#                    (name, photo_path, time.strftime('%Y-%m-%d %H:%M:%S')))
#     db_conn.commit()

# def update_human_appearance(human_id):
#     """Update appearance count and last seen time for an existing human."""
#     cursor.execute("UPDATE HumanData SET appearances = appearances + 1, last_seen = ? WHERE id = ?",
#                    (time.strftime('%Y-%m-%d %H:%M:%S'), human_id))
#     db_conn.commit()

# def get_all_faces():
#     """Retrieve all faces from the database for recognition."""
#     cursor.execute("SELECT id, photo_path FROM HumanData")
#     records = cursor.fetchall()
#     face_encodings = []
#     ids = []
#     for record in records:
#         face_img = face_recognition.load_image_file(record[1])
#         encoding = face_recognition.face_encodings(face_img)
#         if encoding:
#             face_encodings.append(encoding[0])
#             ids.append(record[0])
#     return ids, face_encodings

# known_ids, known_encodings = get_all_faces()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv.resize(frame, (980, 740))

#     # Detect humans
#     results = human_model(frame)
#     for info in results:
#         parameters = info.boxes
#         for box in parameters:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             class_detect = int(box.cls[0])
#             class_detect_name = classnames[class_detect]

#             if class_detect_name == 'person':
#                 # Crop face for recognition
#                 face_img = frame[y1:y2, x1:x2]
#                 face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
#                 face_encoding = face_recognition.face_encodings(face_img_rgb)

#                 if face_encoding:
#                     face_encoding = face_encoding[0]
#                     matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
#                     distances = face_recognition.face_distance(known_encodings, face_encoding)

#                     if True in matches:
#                         match_index = matches.index(True)
#                         human_id = known_ids[match_index]
#                         update_human_appearance(human_id)
#                         cvzone.putTextRect(frame, f"Known Person #{human_id}", (x1, y1 - 10), thickness=2, scale=1)
#                     else:
#                         # New face detected
#                         new_name = f"Person_{len(known_ids) + 1}"
#                         add_human_to_db(new_name, face_img)
#                         known_ids, known_encodings = get_all_faces()
#                         cvzone.putTextRect(frame, f"New Person Added", (x1, y1 - 10), thickness=2, scale=1)

#                 cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)

#     # Display frame
#     cv.imshow("Human Detection and Tracking", frame)

#     # Exit on pressing 't'
#     if cv.waitKey(1) & 0xFF == ord('t'):
#         break

# cap.release()
# cv.destroyAllWindows()
# db_conn.close()


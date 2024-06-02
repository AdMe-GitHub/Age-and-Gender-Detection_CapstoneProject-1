# Age-and-Gender-Detection_CapstoneProject-1
The provided code uses OpenCV's Deep Neural Network (DNN) module to detect faces in an image or a video stream. Once a face is detected, it predicts the age and gender of the person using pre-trained models. Here's a step-by-step explanation:

Import Libraries: The code imports necessary libraries such as OpenCV, math, and argparse for handling image processing and command-line arguments.

Define highlightFace Function: This function uses a pre-trained face detection model to detect faces in the given frame. It marks detected faces with rectangles and returns the modified frame and the coordinates of the face boxes.

Parse Command-Line Arguments: The code uses argparse to allow users to input an image path via command-line arguments.

Load Pre-trained Models: The code loads pre-trained models for face detection (faceNet), age prediction (ageNet), and gender prediction (genderNet).

Capture Video or Image: The code captures video from the webcam or reads an image if a path is provided.

Process Each Frame: In a loop, the code processes each frame:

Detects faces using the highlightFace function.
For each detected face, it crops the face region and prepares it for age and gender prediction.
Predicts gender and age using the respective pre-trained models.
Annotates the image with the predicted gender and age.
Display Results: The code displays the annotated image with detected faces and predicted information.

Sample Image Processing
Let's visualize the processing of a sample image.

Original Image

Processed Image with Annotations

The processed image would have rectangles around detected faces, and each rectangle would be annotated with the predicted gender and age.

Example Usage
To run the code, you would use the command line as follows:

bash
Copy code
python detect_age_gender.py --image path_to_image.jpg
If you don't provide an image path, it will default to using the webcam for live video processing.

Explanation of Key Parts
highlightFace Function:

Converts the frame to a blob format suitable for the DNN.
Detects faces with a confidence threshold.
Draws rectangles around detected faces.
Model Loading:

Loads face detection, age prediction, and gender prediction models from specified files.
Face Detection and Annotation:

For each detected face, the code extracts the face region, predicts the gender and age, and annotates the image.
Complete Code
Here's the complete code:

python
Copy code
import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                     :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

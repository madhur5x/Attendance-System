import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition

# Define the path to the image folder
path = r'C:\Users\madhu\OneDrive\Desktop\CV\image_folder'

# Define the folder for attendance
attendance_folder = os.path.join(os.getcwd(), 'attendance')

# Ensure the attendance folder exists
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Get the current date and format it for the filename
today_date = datetime.now().strftime('%Y-%m-%d')
attendance_file = os.path.join(attendance_folder, f'Attendance_{today_date}.csv')

# Check if today's attendance file exists, and if not, create it
if f'Attendance_{today_date}.csv' not in os.listdir(attendance_folder):
    df = pd.DataFrame(columns=["Name", "Course", "Year", "Batch", "Time"])
    df.to_csv(attendance_file, index=False)

# Initialize lists for images and student data
images = []
studentData = []
myList = os.listdir(path)
print(myList)

# Load images and extract student data from filenames
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    
    # Assuming the filename format is Name_Course_Year_Batch.jpg
    name, course, year, batch = os.path.splitext(cl)[0].split('_')
    studentData.append((name.upper(), course, year, batch))
print(studentData)

# Function to find encodings of faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if any face encodings are found
            encodeList.append(encodings[0])
        else:
            print("No faces found in the image.")
    return encodeList

# Function to mark attendance
def markAttendance(name, course, year, batch):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()  # Get the current time when a face is detected
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{course},{year},{batch},{dtString}\n')

# Find encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Choose between webcam and URL image source
use_webcam = True  # Set to False to use an image URL

if use_webcam:
    cap = cv2.VideoCapture(0)  # Open webcam
else:
    url = 'http://your-ip-camera-url'  # Replace with your IP camera or image URL

while True:
    if use_webcam:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam")
            break
        
        # Flip the image horizontally for mirror effect
        img = cv2.flip(img, 1)
    else:
        # Load image from URL
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        
        # Flip the image horizontally for mirror effect
        img = cv2.flip(img, 1)

    # Resize and convert the image
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and find encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name, course, year, batch = studentData[matchIndex]

            # Draw a rectangle around the face and display details
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'Name: {name}', (x1 + 6, y1 - 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Course: {course}', (x1 + 6, y1 - 35), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Year: {year}', (x1 + 6, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Batch: {batch}', (x1 + 6, y1 + 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name, course, year, batch)  # Mark attendance with additional info

    # Display the image with annotations
    cv2.imshow('Webcam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

# Clean up
if use_webcam:
    cap.release()  # Release the webcam

cv2.destroyAllWindows()

# Inform the user where the Attendance.csv file has been created
print(f"\nProgram terminated. The attendance file has been saved at: {attendance_file}")

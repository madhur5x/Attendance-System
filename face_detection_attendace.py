import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
import pickle
import face_recognition
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
        

# Define the path to the image folder
path = r'C:\Users\madhu\OneDrive\Desktop\Attendance\image_folder'

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

# Filenames for storing known encodings, student data, and processed images
encodings_file = 'known_encodings.pkl'
students_file = 'student_data.pkl'
processed_images_file = 'processed_images.pkl'

# Load existing encodings, student data, and processed image filenames if available
if os.path.exists(encodings_file) and os.path.exists(students_file):
    with open(encodings_file, 'rb') as f:
        encodeListKnown = pickle.load(f)
    with open(students_file, 'rb') as f:
        studentData = pickle.load(f)
else:
    encodeListKnown = []
    studentData = []

# Load the list of processed images
if os.path.exists(processed_images_file):
    with open(processed_images_file, 'rb') as f:
        processed_images = pickle.load(f)
else:
    processed_images = set()

# Check for new images
new_images = []
new_student_data = []
for cl in myList:
    if cl not in processed_images:
        curImg = cv2.imread(f'{path}/{cl}')
        new_images.append(curImg)

        # Assuming the filename format is Name_Course_Year_Batch.jpg
        name, course, year, batch = os.path.splitext(cl)[0].split('_')
        new_student_data.append((name.upper(), course, year, batch))

        # Add to the processed images set
        processed_images.add(cl)

if new_images:
    print(f"New images found: {len(new_images)}. Encoding...")

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

    # Find encodings for new faces
    new_encodings = findEncodings(new_images)

    # Append new encodings and student data to the known lists
    encodeListKnown.extend(new_encodings)
    studentData.extend(new_student_data)

    # Save updated encodings, student data, and processed images
    with open(encodings_file, 'wb') as f:
        pickle.dump(encodeListKnown, f)
    with open(students_file, 'wb') as f:
        pickle.dump(studentData, f)
    with open(processed_images_file, 'wb') as f:
        pickle.dump(processed_images, f)

else:
    print("No new images found.")

# Function to mark attendance
def markAttendance(name, course, year, batch):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()  # Get the current time when a face is detected
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{course},{year},{batch},{dtString}\n')

print('Encoding Complete')

# ================== Accuracy Test Mode ==================
test_mode = True  # Set to True to run accuracy test, False for real-time attendance
test_path = r'C:\Users\madhu\OneDrive\Desktop\Attendance\test_images'  # Update this path

if test_mode:
    if not os.path.exists(test_path):
        print(f"Test directory {test_path} not found. Skipping accuracy test.")
    else:
        test_images = os.listdir(test_path)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Store true and predicted labels for comprehensive metrics
        y_true = []
        y_pred = []
        confidence_scores = []
        student_names = [student[0] for student in studentData]
        
        # Create a dictionary to track performance at different thresholds
        threshold_results = {t/100: {"TP": 0, "FP": 0, "FN": 0} for t in range(30, 101, 5)}
        
        for image_file in test_images:
            # Extract expected name from filename
            try:
                expected_name = os.path.splitext(image_file)[0].split('_')[0].upper()
                y_true.append(expected_name)
            except:
                print(f"Skipping invalid filename: {image_file}")
                continue

            img_path = os.path.join(test_path, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image {image_file}")
                y_pred.append("ERROR")
                confidence_scores.append(0)
                continue

            # Preprocess image
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Detect faces
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            if not encodesCurFrame:
                false_negatives += 1
                print(f"No face detected in {image_file}")
                y_pred.append("NO_FACE")
                confidence_scores.append(0)
                continue

            # Process the detected face (assuming one face per test image)
            encodeFace = encodesCurFrame[0]
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if len(faceDis) == 0:
                matchIndex = -1
                confidence = 0
            else:
                matchIndex = np.argmin(faceDis)
                confidence = 1 - faceDis[matchIndex]  # Convert distance to confidence score
                
                # Evaluate at different thresholds
                for threshold in threshold_results.keys():
                    if confidence >= threshold:
                        if studentData[matchIndex][0] == expected_name:
                            threshold_results[threshold]["TP"] += 1
                        else:
                            threshold_results[threshold]["FP"] += 1
                    else:
                        threshold_results[threshold]["FN"] += 1

            if matchIndex != -1 and matches[matchIndex]:
                detected_name = studentData[matchIndex][0]
                y_pred.append(detected_name)
                confidence_scores.append(confidence)
                
                if detected_name == expected_name:
                    true_positives += 1
                else:
                    false_positives += 1
                    print(f"False positive: Detected {detected_name} instead of {expected_name} in {image_file}")
            else:
                false_negatives += 1
                print(f"Face detected but not recognized in {image_file}")
                y_pred.append("UNKNOWN")
                confidence_scores.append(confidence)

        # Calculate basic metrics with rounding
        total_tests = len(test_images)
        precision = round(true_positives / (true_positives + false_positives), 2) if (true_positives + false_positives) > 0 else 0
        recall = round(true_positives / (true_positives + false_negatives), 2) if (true_positives + false_negatives) > 0 else 0
        accuracy = round(true_positives / total_tests, 2) if total_tests > 0 else 0
        
        # Import required libraries for visualization
        # --------------------------
        # 1) Basic Metrics Bar Chart
        # --------------------------
        def plot_bar_chart(ax):
            ax.clear()
            metrics = ['Precision', 'Recall', 'Accuracy']
            values = [precision, recall, accuracy]
            bars = ax.bar(metrics, values, color=['blue', 'green', 'red'])
            ax.set_title('Face Recognition Accuracy Metrics', fontsize=14)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        # --------------------------------------------------
        # 2) Improved Scrollable Confusion Matrix (showing all classes)
        # --------------------------------------------------
        def plot_scrollable_confusion_matrix(fig, y_true, y_pred, student_names):
            fig.clear()
            
            # Filter to only include valid predictions
            valid_indices = [i for i, pred in enumerate(y_pred) if pred in student_names and y_true[i] in student_names]
            if not valid_indices:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Not enough valid matches for confusion matrix", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title('Confusion Matrix')
                return
            
            filtered_y_true = [y_true[i] for i in valid_indices]
            filtered_y_pred = [y_pred[i] for i in valid_indices]
            
            # Get unique labels without limiting
            unique_labels = sorted(list(set(filtered_y_true + filtered_y_pred)))
            num_classes = len(unique_labels)
            
            # Create the confusion matrix
            cm = confusion_matrix(filtered_y_true, filtered_y_pred, labels=unique_labels)
            
            # Normalize confusion matrix by row (true labels) to show accuracy percentages
            row_sums = cm.sum(axis=1)
            cm_normalized = np.zeros_like(cm, dtype=float)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:
                    cm_normalized[i] = cm[i] / row_sums[i]
            
            # Set up the main confusion matrix plot with adjustable size
            # Create a grid spec for the main plot, slider area, and colorbar
            gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], width_ratios=[20, 1],
                                 left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.02)
            
            ax = fig.add_subplot(gs[0, 0])
            
            # Initial display range - adjust visible matrix part size based on number of classes
            display_size = min(20, num_classes)  # Start with at most 15 classes
            current_start = 0
            
            # Function to update the matrix display
            def update_matrix(start_idx=0):
                ax.clear()
                end_idx = min(start_idx + display_size, num_classes)
                
                visible_cm = cm[start_idx:end_idx, start_idx:end_idx]
                visible_labels = unique_labels[start_idx:end_idx]
                
                # Create heatmap
                sns.heatmap(
                    visible_cm,
                    annot=True,
                    fmt='d',
                    cmap="Blues",
                    xticklabels=visible_labels,
                    yticklabels=visible_labels,
                    ax=ax,
                    cbar=False,  # We'll add a custom colorbar
                    linewidths=0.5,
                    linecolor='gray'
                )
                
                # Highlight diagonal elements (correct predictions)
                for i in range(len(visible_cm)):
                    if visible_cm[i, i] > 0:
                        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=2))
                
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('True', fontsize=12)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                plt.setp(ax.get_yticklabels(), rotation=45, ha="right", fontsize=8)
                
                # Add class distribution annotation
                total_samples = len(filtered_y_true)
                fig.text(0.5, 0.01, f"Total valid samples: {total_samples} | Showing classes {start_idx+1}-{end_idx} of {num_classes}",
                        horizontalalignment='center', fontsize=8)
                
                # Enhanced title with overall accuracy
                overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
                ax.set_title(f'Confusion Matrix (Overall Accuracy: {overall_accuracy:.2%})', fontsize=14)
            
            # Add a colorbar manually to avoid reshaping with slider updates
            cax = fig.add_subplot(gs[0, 1])
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0, np.max(cm)))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Count')
            
            # Add sliders for scrolling through matrix
            # ----------------------------------------
            # 1. Horizontal slider (scroll classes)
            ax_slider = fig.add_subplot(gs[1, 0])
            horizontal_slider = Slider(
                ax=ax_slider,
                label='Class Index',
                valmin=0,
                valmax=max(0, num_classes - display_size),
                valinit=0,
                valstep=1,
                color='green'
            )
            
            def update(val):
                start_idx = int(horizontal_slider.val)
                update_matrix(start_idx)
                fig.canvas.draw_idle()
            
            horizontal_slider.on_changed(update)
            
            # Add a legend to explain the highlighted diagonal
            legend_elements = [plt.Line2D([0], [0], color='green', lw=2, label='Correct Predictions')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Initial update
            update_matrix(current_start)
            
            return horizontal_slider
            
        # --------------------------------------
        # 3) Threshold Analysis (Precision/Recall)
        # --------------------------------------
        def plot_threshold_analysis(ax):
            ax.clear()
            thresholds = list(threshold_results.keys())
            precision_values = []
            recall_values = []
            f1_values = []
            
            for t in thresholds:
                t_tp = threshold_results[t]["TP"]
                t_fp = threshold_results[t]["FP"]
                t_fn = threshold_results[t]["FN"]
                t_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
                t_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
                t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0
                
                precision_values.append(t_precision)
                recall_values.append(t_recall)
                
            ax.plot(thresholds, precision_values, 'b-', label='Precision')
            ax.plot(thresholds, recall_values, 'g-', label='Recall')
            
            ax.set_title('Performance Metrics at Different Confidence Thresholds', fontsize=14)
            ax.set_xlabel('Confidence Threshold')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
       
        # ==============
        # SHOW EACH PLOT
        # ==============
        plt.style.use('ggplot')  # Use a more modern style

        # 1) Basic Metrics Bar Chart
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        plot_bar_chart(ax1)
        plt.tight_layout()
        plt.show()

        # 2) Scrollable Confusion Matrix (allowing to see all classes)
        plt.figure()
        fig2 = plt.figure(figsize=(10, 8))
        slider = plot_scrollable_confusion_matrix(fig2, y_true, y_pred, student_names)
        plt.tight_layout()
        plt.show()

        # 3) Threshold Analysis (Precision, Recall & F1)
        plt.figure()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        plot_threshold_analysis(ax3)
        plt.show()

else:
    # ================== Real-time Attendance Mode ==================
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
            
            if len(faceDis) == 0:
                continue
                
            matchIndex = np.argmin(faceDis)
            confidence = 1 - faceDis[matchIndex]  # Convert distance to confidence

            if matches[matchIndex]:
                name, course, year, batch = studentData[matchIndex]

                # Draw rectangle and display details on the face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add a background for text
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                
                # Display student info with confidence
                cv2.putText(img, f'Name: {name} ({confidence:.2f})', (x1 + 6, y1 - 60), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, f'Course: {course}', (x1 + 6, y1 - 35), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, f'Year: {year}', (x1 + 6, y1 - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, f'Batch: {batch}', (x1 + 6, y1 + 15), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

                markAttendance(name, course, year, batch)  # Mark attendance
            else:
                # Display unknown face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x1 + 6, y1 - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        # Show the webcam feed
        cv2.imshow('Webcam', img)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    # Clean up resources
    if use_webcam:
        cap.release()
    cv2.destroyAllWindows()

# Inform the user where the Attendance.csv file has been created
print(f"\nProgram terminated. The attendance file has been saved at: {attendance_file}")

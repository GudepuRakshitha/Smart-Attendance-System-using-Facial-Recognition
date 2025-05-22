import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# ---------------------------- Configuration ---------------------------- #

# Paths
TRAINING_DIR = 'TrainingImages'      # Directory containing training images
NEW_FACES_DIR = 'NewFaces'           # Directory to save new face images
ATTENDANCE_DIR = 'AttendanceLogs'     # Directory to save attendance Excel files
os.makedirs(NEW_FACES_DIR, exist_ok=True)  # Create if doesn't exist
os.makedirs(ATTENDANCE_DIR, exist_ok=True) # Create if doesn't exist

# Webcam settings
CAMERA_INDEX = 0                     # Change if multiple webcams are connected
FRAME_RESIZE_SCALE = 0.25            # Scale down frame for faster processing

# Attendance logging settings
ATTENDANCE_COLUMNS = ['Roll Number', 'Date', 'Time']

# ---------------------------- Utility Functions ---------------------------- #

def get_current_month_filename():
    """Generates the attendance filename based on the current month and year."""
    now = datetime.now()
    filename = f'Attendance_{now.year}_{now.month:02d}.xlsx'
    return os.path.join(ATTENDANCE_DIR, filename)

def load_training_images():
    """
    Loads and encodes training images.
    Assumes each image filename starts with the student's roll number followed by an underscore.
    Example: 01_1.jpg, 02_2.jpg, etc.
    """
    encode_list = []
    roll_numbers = []

    print("Loading and encoding training images...")
    for img_name in os.listdir(TRAINING_DIR):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract roll number from filename
            try:
                roll_number = img_name.split('_')[0]
            except IndexError:
                print(f"Filename {img_name} does not follow the 'rollnumber_uniqueid.jpg' format. Skipping.")
                continue

            img_path = os.path.join(TRAINING_DIR, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}. Skipping.")
                continue

            print(f"Processing image: {img_name}, Shape: {img.shape}")

            # Convert image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img_rgb)

            if len(face_locations) != 1:
                print(f"Image {img_name} has {len(face_locations)} faces; exactly one face is required. Skipping.")
                continue

            face_encoding = face_recognition.face_encodings(img_rgb, face_locations)[0]
            encode_list.append(face_encoding)
            roll_numbers.append(roll_number)
            print(f"Encoded Roll Number: {roll_number} from image {img_name}")

    print("All training images encoded.\n")
    return encode_list, roll_numbers

def initialize_attendance_file():
    """Initializes the attendance Excel file for the current month."""
    filename = get_current_month_filename()
    if os.path.exists(filename):
        attendance_df = pd.read_excel(filename)
        print(f"Loaded existing attendance file: {filename}\n")
    else:
        attendance_df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
        attendance_df.to_excel(filename, index=False)
        print(f"Created new attendance file: {filename}\n")
    return attendance_df

def mark_attendance(attendance_df, roll_number):
    """Marks attendance for a given roll number if not already marked for today."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not ((attendance_df['Roll Number'] == roll_number) & (attendance_df['Date'] == date_str)).any():
        new_entry = pd.DataFrame({
            'Roll Number': [roll_number],
            'Date': [date_str],
            'Time': [time_str]
        })
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        print(f"Marked attendance for Roll Number: {roll_number} at {time_str} on {date_str}")
    else:
        print(f"Attendance already marked for Roll Number: {roll_number} on {date_str}")

    return attendance_df

def save_attendance(attendance_df):
    """Saves the attendance DataFrame to the Excel file for the current month."""
    filename = get_current_month_filename()
    attendance_df.to_excel(filename, index=False)
    print(f"Attendance saved to {filename}\n")

# ---------------------------- Main Function ---------------------------- #

def main():
    # Load and encode training images
    encode_list, roll_numbers = load_training_images()

    if not encode_list:
        print("No training images were encoded. Exiting.")
        sys.exit(1)

    # Initialize attendance DataFrame
    attendance_df = initialize_attendance_file()

    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    print("Webcam initialized.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting.")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces and encode
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare faces
                matches = face_recognition.compare_faces(encode_list, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(encode_list, face_encoding)
                best_match_index = np.argmin(face_distances)

                # If a match is found
                if matches[best_match_index]:
                    roll_number = roll_numbers[best_match_index]
                    attendance_df = mark_attendance(attendance_df, roll_number)
                    save_attendance(attendance_df)

                    # Draw rectangle and roll number on the frame
                    top, right, bottom, left = face_location
                    top *= int(1/FRAME_RESIZE_SCALE)
                    right *= int(1/FRAME_RESIZE_SCALE)
                    bottom *= int(1/FRAME_RESIZE_SCALE)
                    left *= int(1/FRAME_RESIZE_SCALE)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll No: {roll_number}", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    # It's a new face, save the image
                    roll_number = f"new_{datetime.now().strftime('%Y%m%d%H%M%S')}"  # Unique name for new face
                    new_face_path = os.path.join(NEW_FACES_DIR, f"{roll_number}.jpg")
                    cv2.imwrite(new_face_path, frame[top:bottom, left:right])
                    encode_list.append(face_encoding)  # Add new face encoding
                    roll_numbers.append(roll_number)    # Save the new roll number
                    print(f"Saved new face image: {new_face_path}")

                    # Draw rectangle for the new face
                    top, right, bottom, left = face_location
                    top *= int(1/FRAME_RESIZE_SCALE)
                    right *= int(1/FRAME_RESIZE_SCALE)
                    bottom *= int(1/FRAME_RESIZE_SCALE)
                    left *= int(1/FRAME_RESIZE_SCALE)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "New Face", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Display the resulting frame
            cv2.imshow('Attendance System', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

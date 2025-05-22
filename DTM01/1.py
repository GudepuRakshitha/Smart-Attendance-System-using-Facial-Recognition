import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import sys
import mediapipe as mp
from PIL import Image

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Check if image is in RGB mode
            if img.mode != 'RGB':
                print(f"Converting {image_path} to RGB.")
                img = img.convert('RGB')
            # Save it back in the same format
            img.save(image_path)
            return True
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return False

def load_encoding(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: The image file {image_path} does not exist.")
        return None

    if not is_valid_image(image_path):
        return None

    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            print(f"Loaded encoding for {image_path}")
            return encodings[0]
        else:
            print(f"No faces found in the image {image_path}. Skipping this image.")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def initialize_video_capture():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video capture. Exiting.")
        sys.exit(1)
    else:
        print("Video capture initialized successfully.")
    return video_capture

def initialize_csv(filename):
    file_exists = os.path.isfile(filename)
    try:
        f = open(filename, 'a', newline='')
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Time'])
            print(f"CSV file {filename} created with headers.")
        else:
            print(f"CSV file {filename} opened for appending.")
        return f, writer
    except Exception as e:
        print(f"Error opening CSV file {filename}: {e}")
        sys.exit(1)

def main():
    print("Starting the face recognition script...")

    # Initialize video capture
    video_capture = initialize_video_capture()

    # Load known faces
    known_faces = {
        "jobs": "photos/jobs.jpg",
        "ratan tata": "photos/tata.jpg",
        "GudepuRakshitha": "photos/GudepuRakshitha.jpg",
        "tesla": "photos/tesla.jpg"
    }

    known_face_encodings = []
    known_faces_names = []

    for name, image_path in known_faces.items():
        encoding = load_encoding(image_path)
        if encoding is not None:  # Only append if encoding is valid
            known_face_encodings.append(encoding)
            known_faces_names.append(name)

    if not known_face_encodings:
        print("No valid face encodings found. Exiting.")
        sys.exit(1)

    # Create a copy of known names to track attendance
    students = known_faces_names.copy()
    print(f"Students to track: {students}")

    # Get current date for CSV filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"{current_date}.csv"

    # Initialize CSV file
    f, lnwriter = initialize_csv(csv_filename)

    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results = face_detection.process(rgb_frame)

            face_locations = []
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box information
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    top = int(bboxC.ymin * ih)
                    right = int((bboxC.xmin + bboxC.width) * iw)
                    bottom = int((bboxC.ymin + bboxC.height) * ih)
                    left = int(bboxC.xmin * iw)
                    face_locations.append((top, right, bottom, left))

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare face encodings with known encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                face_names.append(name)

                # If the face is recognized and not already marked, log it
                if name in known_faces_names:
                    if name in students:
                        students.remove(name)
                        print(f"Logged {name}. Remaining students: {students}")
                        current_time = datetime.now().strftime("%H:%M:%S")
                        lnwriter.writerow([name, current_time])
                        f.flush()  # Ensure data is written to the file

            # Display the resulting frame
            cv2.imshow("Face Recognition", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit command received. Exiting.")
                break

            # Optional: If all students are logged, you can choose to exit
            if not students:
                print("All students have been logged. Exiting.")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Release video capture and close windows
        video_capture.release()
        cv2.destroyAllWindows()
        f.close()
        print("Resources released. Program terminated.")

if __name__ == "__main__":
    main()

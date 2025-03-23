import cv2
import numpy as np
import time
import sqlite3
import logging
import os
from insightface.app import FaceAnalysis

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")  # Use 'buffalo_l' model
app.prepare(ctx_id=0)  # Use CPU (ctx_id=-1 for GPU)

# Configure logging
logging.basicConfig(filename="attendance.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Connect to SQLite database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS reference_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Load reference embeddings
def get_face_embedding(image_path):
    """Extract face embedding from an image."""
    image = cv2.imread(image_path)
    faces = app.get(image)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None
    return faces[0].normed_embedding  # Return embedding of the first face

reference_embeddings = {
    "Alice": get_face_embedding("alice.jpg"),
    "Bob": get_face_embedding("bob.jpg"),
    "Charlie": get_face_embedding("charlie.jpg")
}

# Remove None values (if any)
reference_embeddings = {k: v for k, v in reference_embeddings.items() if v is not None}

# Convert reference embeddings to a numpy array
ref_ids = list(reference_embeddings.keys())
ref_embeddings = np.array(list(reference_embeddings.values()))

def compare_embeddings_in_batches(face_embedding, ref_embeddings, batch_size=100):
    """Compare face embedding with reference embeddings in batches."""
    best_match_id = None
    best_match_score = -1
    for i in range(0, len(ref_embeddings), batch_size):
        batch_embeddings = ref_embeddings[i:i + batch_size]
        scores = np.dot(batch_embeddings, face_embedding)
        max_score_idx = np.argmax(scores)
        if scores[max_score_idx] > best_match_score:
            best_match_score = scores[max_score_idx]
            best_match_id = ref_ids[i + max_score_idx]
    return best_match_id, best_match_score

def calculate_dynamic_threshold(ref_embeddings, face_embedding):
    """Calculate a dynamic threshold based on reference embeddings and the current face embedding."""
    mean_distance = np.mean(np.dot(ref_embeddings, face_embedding))
    return mean_distance * 0.8  # Example threshold calculation

# Log attendance to the log file
def log_attendance(name):
    logging.info(f"Attendance logged for {name}")

# Log unknown face to the log file
def log_unknown_face():
    logging.info("Unknown face detected")

# Save unknown face to a file
def save_unknown_face(frame, bbox, frame_count, face_index):
    x1, y1, x2, y2 = bbox
    face_img = frame[y1:y2, x1:x2]
    filename = f"unknown_faces/unknown_{frame_count}_{face_index}.jpg"
    cv2.imwrite(filename, face_img)
    logging.info(f"Unknown face saved as {filename}")

# Insert attendance record into the database
def insert_attendance(name):
    cursor.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
    conn.commit()

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Initialize FPS variables
start_time = time.time()
frame_count = 0

# Create a directory to save unknown faces
os.makedirs("unknown_faces", exist_ok=True)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        # Detect faces in the frame
        faces = app.get(frame)
        if len(faces) == 0:
            print("No faces detected in the frame.")
            continue  # Skip to the next frame

        # Process each detected face
        for i, face in enumerate(faces):
            face_embedding = face.normed_embedding

            # Compare with all reference embeddings
            best_match_id, best_match_score = compare_embeddings_in_batches(face_embedding, ref_embeddings)

            # Calculate dynamic threshold
            confidence_threshold = calculate_dynamic_threshold(ref_embeddings, face_embedding)

            # Check if the best match meets the threshold
            if best_match_score > confidence_threshold:
                print(f"Face {i+1} matched with {best_match_id} (Score: {best_match_score:.2f})")
                log_attendance(best_match_id)
                insert_attendance(best_match_id)
                label = f"Welcome, {best_match_id}! Good luck!"
                color = (0, 255, 0)  # Green for matched faces
            else:
                print(f"Face {i+1} did not match any reference.")
                log_unknown_face()
                save_unknown_face(frame, face.bbox.astype(int), frame_count, i)
                label = "Unknown. Please report to your department."
                color = (0, 0, 255)  # Red for unknown faces

            # Draw bounding box and label
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display the frame
        try:
            cv2.imshow("Examination Attendance Monitor", frame)
        except cv2.error as e:
            print(f"Error displaying frame: {e}")
            break

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"Error closing windows: {e}")
    conn.close()
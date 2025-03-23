import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")  # Use 'buffalo_l' model
app.prepare(ctx_id=0)  # Use CPU (ctx_id=-1 for GPU)

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
    "person1": get_face_embedding("person1.jpg"),
    "person2": get_face_embedding("person2.png"),
    "person3": get_face_embedding("person3(1).jpg")
}

# Remove None values (if any)
reference_embeddings = {k: v for k, v in reference_embeddings.items() if v is not None}

# Convert reference embeddings to a numpy array
ref_ids = list(reference_embeddings.keys())
ref_embeddings = np.array(list(reference_embeddings.values()))

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Initialize FPS variables
start_time = time.time()
frame_count = 0

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
            similarities = np.dot(ref_embeddings, face_embedding)
            best_match_idx = np.argmax(similarities)
            best_match_score = similarities[best_match_idx]
            best_match_id = ref_ids[best_match_idx]

            # Check if the best match meets the threshold
            confidence_threshold = 0.5  # Adjust as needed
            if best_match_score > confidence_threshold:
                print(f"Face {i+1} matched with {best_match_id} (Score: {best_match_score:.2f})")
                label = f"{best_match_id} ({best_match_score:.2f})"
                color = (0, 255, 0)  # Green for matched faces
            else:
                print(f"Face {i+1} did not match any reference.")
                label = "Unknown"
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
            cv2.imshow("Real-Time Face Recognition", frame)
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
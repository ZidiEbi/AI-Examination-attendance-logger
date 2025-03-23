import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Load InsightFace Model (Face Detection + Recognition)
app = FaceAnalysis(name="buffalo_l")  # 'buffalo_l' is a lightweight, accurate model
app.prepare(ctx_id=0)  # Use CPU (set ctx_id=-1 for GPU if available)

def get_face_embedding(image_path):
    """Extract face embedding using InsightFace."""
    image = cv2.imread(image_path)

    # Detect faces
    faces = app.get(image)

    if len(faces) == 0:
        print("❌ No face detected!")
        return None

    # Extract embedding from the first detected face
    embedding = faces[0].normed_embedding  
    return np.array(embedding)

# Test with two images
embedding1 = get_face_embedding("person1.jpg")  # Replace with actual image path
embedding2 = get_face_embedding("person2.png")

if embedding1 is not None and embedding2 is not None:
    # Compute cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    print(f"Face Similarity Score: {similarity:.2f}")

    # Set a threshold (e.g., 0.5) for recognizing the same person
    if similarity > 0.5:
        print("✅ Faces match!")
    else:
        print("❌ Faces do not match.")

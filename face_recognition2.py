import cv2
import numpy as np
import time
import os
from insightface.app import FaceAnalysis

# Configuration
EMBEDDINGS_DIR = "embeddings"
IMAGES_DIR = "images"  # New dedicated images folder
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Reference images (now using full paths)
REFERENCE_IMAGES = {
    "Zidideke Ebikake": os.path.join(IMAGES_DIR, "person1.jpg"),
    "person2": os.path.join(IMAGES_DIR, "person2.png"),
    "Enai Tiemo": os.path.join(IMAGES_DIR, "person3.jpg")
}

# Initialize model with explicit CPU provider
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)  # Use CPU

def validate_image(image_path):
    """Check if image is valid and contains a detectable face"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image {image_path}")
        print("Please check:")
        print("- File exists and path is correct")
        print("- File is a valid image (jpg/png)")
        return None
    
    # Resize if too small (min 200x200 recommended)
    if min(img.shape[:2]) < 200:
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
    
    # Debug: Show image temporarily
    cv2.imshow("Validating Image - Press any key", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    return img

def get_or_create_embedding(image_path, person_name):
    """Get embedding from cache or generate new one with validation"""
    npy_path = os.path.join(EMBEDDINGS_DIR, f"{person_name}.npy")
    
    # Try to load existing embedding
    if os.path.exists(npy_path):
        print(f"✅ Using cached embedding for {person_name}")
        return np.load(npy_path)
    
    # Generate new embedding with validation
    print(f"\n🔍 Processing new face: {person_name}")
    img = validate_image(image_path)
    if img is None:
        return None
        
    # Detect faces with enhanced parameters
    faces = app.get(img)
    if len(faces) == 0:
        print(f"❌ No face detected in {image_path}")
        print("Possible solutions:")
        print("- Use a clearer frontal face image")
        print("- Ensure good lighting (no shadows)")
        print("- Crop image to focus on face")
        print("- Try higher resolution (min 640x480 recommended)")
        return None
        
    embedding = faces[0].normed_embedding
    np.save(npy_path, embedding)
    print(f"✅ Saved new embedding for {person_name}")
    return embedding

def check_image_requirements():
    """Display requirements for reference images"""
    print("\n📝 Image Requirements:")
    print("- Clear frontal face (no sunglasses/obstructions)")
    print("- Well-lit (no harsh shadows)")
    print("- Minimum 640x480 resolution recommended")
    print("- File formats: JPG, PNG")
    print(f"- Store images in: {os.path.abspath(IMAGES_DIR)}")

# Main execution
if __name__ == "__main__":
    check_image_requirements()
    
    # Load or generate all reference embeddings
    reference_embeddings = {}
    for name, img_path in REFERENCE_IMAGES.items():
        embedding = get_or_create_embedding(img_path, name)
        if embedding is not None:
            reference_embeddings[name] = embedding

    if not reference_embeddings:
        print("\n❌ Error: No valid embeddings generated")
        print(f"Please check your images in {IMAGES_DIR} and try again")
        exit()

    # Real-time recognition
    print("\n🎥 Starting webcam recognition... (Press 'q' to quit)")
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Webcam error - check camera connection")
                break
                
            faces = app.get(frame)
            if len(faces) == 0:
                cv2.putText(frame, "Scanning...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                for face in faces:
                    embedding = face.normed_embedding
                    similarities = np.dot(
                        np.array(list(reference_embeddings.values())), 
                        embedding
                    )
                    best_idx = np.argmax(similarities)
                    best_score = similarities[best_idx]
                    best_id = list(reference_embeddings.keys())[best_idx]
                    
                    color = (0,255,0) if best_score > 0.5 else (0,0,255)
                    label = f"{best_id} ({best_score:.2f})" if best_score > 0.5 else "Unknown"
                    
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # FPS counter
            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            
            cv2.imshow("Attendance Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n🛑 Webcam released")
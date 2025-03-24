import cv2
import mediapipe as mp
import face_recognition

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load known faces and their encodings
known_face_encodings = [
    face_recognition.face_encodings(face_recognition.load_image_file("person1.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("person2.jpg"))[0]
]
known_face_names = [
    "Person 1",
    "Person 2"
]

def detect_and_recognize_faces():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

                # Extract face location
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                int(bboxC.width * iw), int(bboxC.height * ih))

                # Recognize face
                face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Draw name
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Face Detection and Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Start real-time face detection and recognition
detect_and_recognize_faces()

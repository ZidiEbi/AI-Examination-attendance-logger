import sqlite3
import numpy as np
from face_recognition import get_face_embedding  # Import function

def save_embedding(name, embedding):
    """Store a student's face embedding in the SQLite database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Convert NumPy array to binary data
    embedding_bytes = embedding.tobytes()

    # Insert data
    cursor.execute("INSERT INTO students (name, embedding) VALUES (?, ?)", (name, embedding_bytes))

    conn.commit()
    conn.close()
    print(f"✅ Face embedding saved for {name}")

# Register a new student
student_name = input("Enter student name: ")
image_path = input("Enter path to student image: ")

embedding = get_face_embedding(image_path)  # Extract embedding
if embedding is not None:
    save_embedding(student_name, embedding)

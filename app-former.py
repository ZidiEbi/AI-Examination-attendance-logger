import os
import cv2
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
import csv
import zipfile
import shutil

# Pillow is needed for generating a default passport image if one doesn't exist.
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None
    print("Pillow not installed. Cannot generate default-passport.jpg placeholder.")
    print("Please install Pillow: pip install Pillow")


app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Change for production!
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Face recognition setup
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Database setup
def get_db():
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        # Students table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            level TEXT NOT NULL,
            embedding_path TEXT NOT NULL
        )""")
        
        # Exams table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            department TEXT NOT NULL,
            level TEXT NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL,
            status TEXT DEFAULT 'scheduled'
        )""")

        # Add status column if it doesn't exist (if upgrading from an older schema)
        try:
            conn.execute("ALTER TABLE exams ADD COLUMN status TEXT DEFAULT 'scheduled'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Exam participants table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS exam_participants (
            exam_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            PRIMARY KEY (exam_id, student_id)
        )""")

        # Try to add verified column if it doesn't exist
        try:
            conn.execute("ALTER TABLE exam_participants ADD COLUMN verified INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Attendance logs
        conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY,
            exam_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence REAL NOT NULL,
            FOREIGN KEY (exam_id, student_id) REFERENCES exam_participants(exam_id, student_id) ON DELETE CASCADE
        )""")

        # Create indexes for better query performance
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_exams_time 
        ON exams(start_time, end_time)
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_students_dept 
        ON students(department, level)
        """)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exam_status ON exams(status)")
        except sqlite3.OperationalError:
            pass  # Column doesn't exist
    
    # Enable foreign key support after connection (important for ON DELETE CASCADE)
    with get_db() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")


init_db()

# Auth decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def generate_embedding(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}") # Debugging print
        return False
    
    faces = face_app.get(img)
    if len(faces) == 0:
        print(f"Error: No face detected in image at {image_path}") # Debugging print
        return False
    
    embedding = faces[0].normed_embedding
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)
    print(f"Embedding saved to {output_path}") # Debugging print
    return True

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['logged_in'] = True
            flash('Logged in successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/admin')
@login_required
def admin_dashboard():
    with get_db() as conn:
        raw_exams = conn.execute("""
            SELECT id, title, department, level, start_time, end_time, status 
            FROM exams 
            ORDER BY start_time DESC
            LIMIT 5
        """).fetchall()
        
        exams = []
        for exam in raw_exams:
            exam_dict = dict(exam)
            try:
                if isinstance(exam_dict['start_time'], str):
                    exam_dict['start_time'] = datetime.strptime(exam_dict['start_time'], '%Y-%m-%d %H:%M:%S')
                if isinstance(exam_dict['end_time'], str):
                    exam_dict['end_time'] = datetime.strptime(exam_dict['end_time'], '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"Error parsing datetime for exam ID {exam_dict.get('id', 'N/A')}: {e}")
            exams.append(exam_dict)
        
        recent_students = conn.execute("""
            SELECT * FROM students 
            ORDER BY id DESC 
            LIMIT 5
        """).fetchall()
    
    return render_template('admin_dashboard.html', 
                           exams=exams,
                           students=recent_students)

@app.route('/admin/students')
@login_required
def list_students():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '').strip()
    
    with get_db() as conn:
        if search:
            query = "SELECT * FROM students WHERE name LIKE ? OR student_id LIKE ? ORDER BY name LIMIT ? OFFSET ?"
            students = conn.execute(query, (f"%{search}%", f"%{search}%", per_page, (page-1)*per_page)).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM students WHERE name LIKE ? OR student_id LIKE ?", 
                                 (f"%{search}%", f"%{search}%")).fetchone()[0]
        else:
            students = conn.execute("SELECT * FROM students ORDER BY name LIMIT ? OFFSET ?", 
                                   (per_page, (page-1)*per_page)).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    
    total_pages = (total + per_page - 1) // per_page
    
    return render_template('students_list.html', 
                           students=students,
                           page=page,
                           total_pages=total_pages,
                           search=search)

@app.route('/admin/students/export')
@login_required
def export_students():
    """Export all students as CSV"""
    try:
        with get_db() as conn:
            students = conn.execute("""
                SELECT name, student_id, department, level 
                FROM students ORDER BY name
            """).fetchall()
        
        csv_data = "Name,Student ID,Department,Level\n"
        csv_data += "\n".join(
            f'{s["name"]},{s["student_id"]},{s["department"]},{s["level"]}'
            for s in students
        )
        
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=students_export.csv"}
        )
        
    except Exception as e:
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('list_students'))    

@app.route('/bulk_upload', methods=['GET', 'POST'])
@login_required
def bulk_upload():
    if request.method == 'POST':
        csv_path = None
        images_dir = os.path.join(app.static_folder, 'images', 'students')
        os.makedirs(images_dir, exist_ok=True)
        
        try:
            csv_file = request.files.get('csv_file')
            if not csv_file:
                flash('No CSV file provided.', 'error')
                return redirect(url_for('bulk_upload'))

            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(csv_file.filename))
            csv_file.save(csv_path)
            
            photos_zip = request.files.get('photos_zip')
            if not photos_zip:
                flash('No photos ZIP file provided.', 'error')
                return redirect(url_for('bulk_upload'))

            with zipfile.ZipFile(photos_zip, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        student_id = os.path.splitext(os.path.basename(file))[0]
                        student_id = ''.join(c for c in student_id if c.isalnum() or c in ('-', '_'))
                        
                        temp_path = os.path.join(images_dir, f"temp_{student_id}.jpg")
                        with open(temp_path, 'wb') as f:
                            f.write(zip_ref.read(file))
                        
                        img = cv2.imread(temp_path)
                        if img is not None:
                            img = cv2.resize(img, (413, 531))
                            final_path = os.path.join(images_dir, f"{student_id}.jpg")
                            cv2.imwrite(final_path, img)
                        os.remove(temp_path)
            
            students_processed = 0
            with open(csv_path, 'r', encoding='utf-8-sig') as f, get_db() as conn:
                reader = csv.DictReader(f)
                for row in reader:
                    if not all(k in row for k in ['name', 'student_id', 'department', 'level']):
                        flash(f"CSV row missing required columns: {row}", 'warning')
                        continue

                    student_id = row['student_id'].strip()
                    clean_id = ''.join(c for c in student_id if c.isalnum() or c in ('-', '_'))
                    img_path = os.path.join(images_dir, f"{clean_id}.jpg")
                    
                    embeddings_dir = 'embeddings'
                    os.makedirs(embeddings_dir, exist_ok=True)
                    embedding_output_path = os.path.join(embeddings_dir, f"{clean_id}.npy")
                    
                    if os.path.exists(img_path):
                        if generate_embedding(img_path, embedding_output_path):
                            try:
                                existing_student = conn.execute(
                                    "SELECT id FROM students WHERE student_id = ?", (student_id,)
                                ).fetchone()

                                if existing_student:
                                    conn.execute(
                                        """UPDATE students SET name=?, department=?, level=?, embedding_path=? 
                                           WHERE student_id=?""",
                                        (row['name'], row['department'], row['level'], embedding_output_path, student_id)
                                    )
                                    flash(f'Student {student_id} updated.', 'info')
                                else:
                                    conn.execute(
                                        """INSERT INTO students (name, student_id, department, level, embedding_path)
                                           VALUES (?, ?, ?, ?, ?)""",
                                        (row['name'], student_id, row['department'], row['level'], embedding_output_path)
                                    )
                                    students_processed += 1
                                conn.commit()
                            except sqlite3.IntegrityError as ie:
                                flash(f'Database integrity error for student {student_id}: {ie}', 'error')
                                print(f"Integrity Error: {ie} for student {student_id}")
                                conn.rollback()
                            except Exception as db_e:
                                flash(f'Error saving student {student_id} to DB: {db_e}', 'error')
                                print(f"DB Save Error: {db_e} for student {student_id}")
                                conn.rollback()
                        else:
                            flash(f'Could not generate embedding for {student_id}. Skipping student.', 'warning')
                    else:
                        flash(f'Photo for {student_id} not found after processing. Skipping student.', 'warning')
            
            flash(f'Bulk upload complete. {students_processed} new students registered/updated.', 'success')
        except zipfile.BadZipFile:
            flash('The uploaded photos file is not a valid ZIP archive.', 'error')
        except KeyError as ke:
            flash(f'Missing file in form: {ke}. Please ensure both CSV and ZIP are uploaded.', 'error')
        except Exception as e:
            flash(f'Error during bulk upload: {str(e)}', 'error')
            for file_name in os.listdir(images_dir):
                if file_name.startswith('temp_'):
                    try:
                        os.remove(os.path.join(images_dir, file_name))
                    except OSError as ose:
                        print(f"Error cleaning up temp file {file_name}: {ose}")
        finally:
            if csv_path and os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except OSError as ose:
                    print(f"Error removing CSV temp file {csv_path}: {ose}")
            
        return redirect(url_for('list_students'))
    
    return render_template('bulk_upload.html')

@app.route('/camera/<int:exam_id>')
@login_required
def camera(exam_id):
    # Retrieve exam details to display on the camera page
    with get_db() as conn:
        exam_row = conn.execute("SELECT id, title, department, level, status FROM exams WHERE id = ?", (exam_id,)).fetchone()
        if not exam_row:
            flash(f"Exam with ID {exam_id} not found.", 'error')
            return redirect(url_for('list_exams'))
        exam_details = dict(exam_row)
    return render_template('camera.html', exam_id=exam_id, exam_details=exam_details)

@app.route('/download-template')
def download_template():
    csv_data = "name,student_id,department,level\nJohn Doe,CS1001,Computer Science,300"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=students_template.csv"}
    )

@app.route('/admin/exams/<int:exam_id>/manage')
@login_required
def manage_exam(exam_id):
    with get_db() as conn:
        exam_row = conn.execute("""
            SELECT id, title, department, level, status,
                   start_time, end_time
            FROM exams WHERE id = ?
        """, (exam_id,)).fetchone()
        
        if exam_row:
            exam = dict(exam_row)
            try:
                if isinstance(exam['start_time'], str):
                    exam['start_time'] = datetime.strptime(exam['start_time'], '%Y-%m-%d %H:%M:%S')
                if isinstance(exam['end_time'], str):
                    exam['end_time'] = datetime.strptime(exam['end_time'], '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"Error converting datetime for exam ID {exam_id}: {e}")
        else:
            flash(f"Exam with ID {exam_id} not found.", 'error')
            return redirect(url_for('list_exams'))
        
        participants = conn.execute("""
            SELECT student_id, verified FROM exam_participants WHERE exam_id = ?
        """, (exam_id,)).fetchall()
        participant_ids = [p['student_id'] for p in participants]
        verified_status = {p['student_id']: p['verified'] for p in participants}
        
        students = conn.execute("""
            SELECT id, name, student_id, department, level 
            FROM students ORDER BY name
        """).fetchall()
    
    return render_template('manage_exam.html', 
                           exam=exam,
                           students=students,
                           participant_ids=participant_ids,
                           verified_status=verified_status)

@app.route('/admin/exams/<int:exam_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_exam(exam_id):
    with get_db() as conn:
        exam_row = conn.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
        if not exam_row:
            flash('Exam not found!', 'error')
            return redirect(url_for('list_exams'))
        
        exam = dict(exam_row)

        if isinstance(exam['start_time'], str):
            try:
                exam['start_time'] = datetime.strptime(exam['start_time'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass

        if isinstance(exam['end_time'], str):
            try:
                exam['end_time'] = datetime.strptime(exam['end_time'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass

    if request.method == 'POST':
        try:
            title = request.form['title']
            department = request.form['department']
            level = request.form['level']
            start_time_str = request.form['start_time']
            end_time_str = request.form['end_time']

            try:
                new_start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
                new_end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')
            except ValueError as e:
                flash(f'Invalid date format: {str(e)}. Please use YYYY-MM-DDTHH:MM.', 'error')
                return redirect(url_for('edit_exam', exam_id=exam_id))

            if new_end_time <= new_start_time:
                flash('End time must be after start time', 'error')
                return redirect(url_for('edit_exam', exam_id=exam_id))

            sqlite_new_start = new_start_time.strftime('%Y-%m-%d %H:%M:%S')
            sqlite_new_end = new_end_time.strftime('%Y-%m-%d %H:%M:%S')

            with get_db() as conn:
                conn.execute(
                    """UPDATE exams SET title = ?, department = ?, level = ?, 
                       start_time = ?, end_time = ? WHERE id = ?""",
                    (title, department, level, sqlite_new_start, sqlite_new_end, exam_id)
                )
                conn.commit()
            flash('Exam updated successfully!', 'success')
            return redirect(url_for('manage_exam', exam_id=exam_id))
        except Exception as e:
            flash(f'Error updating exam: {str(e)}', 'error')
            print(f"Error updating exam ID {exam_id}: {e}")
            return redirect(url_for('edit_exam', exam_id=exam_id))

    return render_template('edit_exam.html', exam=exam)


@app.route('/admin/exams/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_exam(exam_id):
    """Delete an exam and its participants and attendance logs."""
    with get_db() as conn:
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("DELETE FROM exams WHERE id = ?", (exam_id,)) # Cascades due to foreign keys
            conn.commit()
            flash('Exam and associated data deleted successfully!', 'success')
        except sqlite3.Error as e:
            conn.rollback()
            flash(f'Error deleting exam: {str(e)}', 'error')
            print(f"Database error during exam deletion: {e}")
    return redirect(url_for('list_exams'))     

@app.route('/admin/exams/<int:exam_id>/add_students', methods=['POST'])
@login_required
def add_students_to_exam(exam_id):
    student_ids = request.form.getlist('student_ids')
    
    with get_db() as conn:
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("""
                DELETE FROM exam_participants WHERE exam_id = ?
            """, (exam_id,))
            
            for student_id in student_ids:
                conn.execute("""
                    INSERT INTO exam_participants (exam_id, student_id, verified)
                    VALUES (?, ?, ?)
                """, (exam_id, student_id, 0))
            conn.commit()
            flash('Students updated for exam successfully!', 'success')
        except sqlite3.IntegrityError as e:
            conn.rollback()
            flash(f'Error adding students: {str(e)}. A student might already be linked.', 'error')
            print(f"Integrity Error adding students to exam: {e}")
        except Exception as e:
            conn.rollback()
            flash(f'Error updating participants: {str(e)}', 'error')
            print(f"General error updating participants: {e}")

    return redirect(url_for('manage_exam', exam_id=exam_id))

@app.route('/admin/exams/add', methods=['GET', 'POST'])
@login_required
def add_exam():
    if request.method == 'POST':
        try:
            title = request.form['title']
            department = request.form['department']
            level = request.form['level']
            
            start_time_str = request.form['start_time']
            end_time_str = request.form['end_time']
            
            try:
                start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
                end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')
            except ValueError as e:
                flash(f'Invalid date format: {str(e)}. Please use YYYY-MM-DDTHH:MM.', 'error')
                return redirect(url_for('add_exam'))

            if end_time <= start_time:
                flash('End time must be after start time', 'error')
                return redirect(url_for('add_exam'))
            
            sqlite_start = start_time.strftime('%Y-%m-%d %H:%M:%S')
            sqlite_end = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO exams (title, department, level, start_time, end_time, status)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (title, department, level, sqlite_start, sqlite_end, 'scheduled')
                )
                conn.commit()
            
            flash('Exam added successfully!', 'success')
            return redirect(url_for('list_exams'))
            
        except Exception as e:
            flash(f'Error creating exam: {str(e)}', 'error')
            print(f"Error creating exam: {e}")
            return redirect(url_for('add_exam'))
    
    return render_template('add_exam.html')

@app.route('/admin/exams')
@login_required
def list_exams():
    """List all exams and ensure datetime objects for template rendering."""
    with get_db() as conn:
        raw_exams = conn.execute("SELECT id, title, department, level, start_time, end_time, status FROM exams ORDER BY start_time DESC").fetchall()
    
    exams = []
    for exam_row in raw_exams:
        exam_dict = dict(exam_row)
        try:
            if isinstance(exam_dict['start_time'], str):
                exam_dict['start_time'] = datetime.strptime(exam_dict['start_time'], '%Y-%m-%d %H:%M:%S')
            if isinstance(exam_dict['end_time'], str):
                exam_dict['end_time'] = datetime.strptime(exam_dict['end_time'], '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            print(f"Error converting datetime for exam ID {exam_dict.get('id', 'N/A')}: {e}")
        exams.append(exam_dict)

    return render_template('exam_list.html', exams=exams)

@app.route('/api/exam/<int:exam_id>/verify', methods=['POST'])
@login_required
def verify_student_for_exam(exam_id): 
    try:
        # Check if current exam is active
        with get_db() as conn:
            exam_status_row = conn.execute("SELECT status FROM exams WHERE id = ?", (exam_id,)).fetchone()
            if not exam_status_row or exam_status_row['status'] != 'active':
                return jsonify({'status': 'error', 'message': 'Exam is not active for verification.', 'bbox': None}), 400

        img_file = request.files.get('image')
        if not img_file:
            return jsonify({'status': 'error', 'message': 'No image provided for verification.', 'bbox': None}), 400

        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        faces = face_app.get(img)
        if not faces:
            return jsonify({'status': 'no_face_detected', 'message': 'No face detected in the image.', 'bbox': None})

        # Process the first detected face (assuming one face per frame for attendance)
        # If multiple faces are expected, loop through them and find best match for each
        detected_face = faces[0]
        detected_embedding = detected_face.normed_embedding
        detected_bbox = detected_face.bbox.tolist() # Convert numpy array to list for JSON serialization

        best_match = None
        best_score = 0.0
        VERIFICATION_THRESHOLD = 0.5 # Adjustable threshold

        with get_db() as conn:
            students_for_exam = conn.execute("""
                SELECT s.student_id, s.name, s.embedding_path 
                FROM students s
                JOIN exam_participants ep ON s.student_id = ep.student_id
                WHERE ep.exam_id = ?
            """, (exam_id,)).fetchall()

            if not students_for_exam:
                return jsonify({'status': 'error', 'message': 'No students registered for this exam.', 'bbox': detected_bbox})

            for student in students_for_exam:
                embedding_path = student['embedding_path']
                if not os.path.exists(embedding_path):
                    print(f"Warning: Embedding file not found for student {student['student_id']}: {embedding_path}")
                    continue

                try:
                    stored_emb = np.load(embedding_path)
                except Exception as load_e:
                    print(f"Error loading embedding for student {student['student_id']}: {load_e}")
                    continue

                score = np.dot(detected_embedding, stored_emb)
                
                if score > best_score:
                    best_score = score
                    best_match = student

        if best_match and best_score > VERIFICATION_THRESHOLD:
            with get_db() as conn:
                already_verified_row = conn.execute("""
                    SELECT verified FROM exam_participants 
                    WHERE exam_id = ? AND student_id = ?
                """, (exam_id, best_match['student_id'])).fetchone()

                if already_verified_row and already_verified_row['verified'] == 1:
                    return jsonify({
                        'status': 'already_verified',
                        'student_id': best_match['student_id'],
                        'name': best_match['name'],
                        'message': f"{best_match['name']} ({best_match['student_id']}) already verified for this exam.",
                        'confidence': float(best_score),
                        'bbox': detected_bbox
                    })

                # Log verification
                conn.execute("""
                    INSERT INTO attendance_logs (exam_id, student_id, confidence)
                    VALUES (?, ?, ?)
                """, (exam_id, best_match['student_id'], float(best_score)))
                
                # Mark as verified in exam_participants
                conn.execute("""
                    UPDATE exam_participants 
                    SET verified = 1 
                    WHERE exam_id = ? AND student_id = ?
                """, (exam_id, best_match['student_id']))
                conn.commit()
                
                return jsonify({
                    'status': 'verified',
                    'student_id': best_match['student_id'],
                    'name': best_match['name'],
                    'confidence': float(best_score),
                    'message': f"{best_match['name']} ({best_match['student_id']}) verified successfully!",
                    'bbox': detected_bbox
                })
        else:
            return jsonify({
                'status': 'unknown', 
                'message': f'No matching student found or confidence too low ({best_score:.2f} < {VERIFICATION_THRESHOLD}).',
                'bbox': detected_bbox
            })

    except Exception as e:
        print(f"Error during verification: {e}")
        return jsonify({'status': 'error', 'message': f'Server error during verification: {str(e)}', 'bbox': None}), 500

@app.route('/admin/exams/<int:exam_id>/start', methods=['POST'])
@login_required
def start_exam(exam_id):
    with get_db() as conn:
        try:
            conn.execute("UPDATE exams SET status = 'active' WHERE id = ?", (exam_id,))
            conn.commit()
            flash('Exam started successfully!', 'success')
        except sqlite3.Error as e:
            conn.rollback()
            flash(f'Error starting exam: {str(e)}', 'error')
    return redirect(url_for('manage_exam', exam_id=exam_id))

@app.route('/admin/exams/<int:exam_id>/end', methods=['POST'])
@login_required
def end_exam(exam_id):
    with get_db() as conn:
        try:
            conn.execute("UPDATE exams SET status = 'completed' WHERE id = ?", (exam_id,))
            conn.commit()
            flash('Exam ended successfully!', 'success')
        except sqlite3.Error as e:
            conn.rollback()
            flash(f'Error ending exam: {str(e)}', 'error')
    return redirect(url_for('manage_exam', exam_id=exam_id))

@app.route('/api/exam/<int:exam_id>/attendance_report')
@login_required
def attendance_report(exam_id):
    with get_db() as conn:
        participants = conn.execute("""
            SELECT s.student_id, s.name, s.department, s.level,
                   ep.verified as verified_status,
                   MAX(al.confidence) as confidence,
                   MAX(al.timestamp) as verification_time
            FROM exam_participants ep
            JOIN students s ON ep.student_id = s.student_id
            LEFT JOIN attendance_logs al ON ep.student_id = al.student_id AND ep.exam_id = al.exam_id
            WHERE ep.exam_id = ?
            GROUP BY s.student_id
            ORDER BY s.name
        """, (exam_id,)).fetchall()

    csv_data = "Student ID,Name,Department,Level,Attended,Confidence,Verification Time\n"
    for p in participants:
        attended_status = "✓" if p["verified_status"] == 1 else "✗"
        
        verification_time_str = "N/A"
        if p["verification_time"] and isinstance(p["verification_time"], str):
            try:
                dt_obj = datetime.strptime(p["verification_time"], '%Y-%m-%d %H:%M:%S')
                verification_time_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                print(f"Warning: Could not parse timestamp {p['verification_time']} for student {p['student_id']}")
                verification_time_str = "Invalid Time Format"

        csv_data += f'{p["student_id"]},{p["name"]},{p["department"]},{p["level"]},' \
                    f'{attended_status},{p["confidence"] or "N/A"},{verification_time_str}\n'
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=exam_{exam_id}_attendance.csv"}
    )

@app.route('/api/exam/<int:exam_id>/attendance_stats')
@login_required
def attendance_stats(exam_id):
    """API endpoint to get attendance statistics for a given exam."""
    with get_db() as conn:
        total_participants_row = conn.execute("SELECT COUNT(*) FROM exam_participants WHERE exam_id = ?", (exam_id,)).fetchone()
        total_participants = total_participants_row[0] if total_participants_row else 0

        attended_count_row = conn.execute("SELECT COUNT(*) FROM exam_participants WHERE exam_id = ? AND verified = 1", (exam_id,)).fetchone()
        attended_count = attended_count_row[0] if attended_count_row else 0
    
    percentage = (attended_count / total_participants * 100) if total_participants > 0 else 0

    return jsonify({
        'total': total_participants,
        'attended': attended_count,
        'percentage': round(percentage, 2)
    })

# NEW ENDPOINT: Fetch participants for display in a separate list
@app.route('/api/exam/<int:exam_id>/participants')
@login_required
def get_exam_participants(exam_id):
    """API endpoint to get detailed list of participants for an exam, including their verified status."""
    with get_db() as conn:
        try:
            participants = conn.execute("""
                SELECT 
                    s.student_id, 
                    s.name, 
                    ep.verified AS verified_status -- Renamed to avoid clash with python's verified_status in Flask template
                FROM exam_participants ep
                JOIN students s ON ep.student_id = s.student_id
                WHERE ep.exam_id = ?
                ORDER BY s.name
            """, (exam_id,)).fetchall()

            participants_list = []
            for p in participants:
                p_dict = dict(p)
                # Construct photo URL based on student_id and known image path
                # Assuming images are stored in static/images/students/[student_id].jpg
                student_photo_filename = f"{p_dict['student_id']}.jpg"
                p_dict['photo_url'] = url_for('static', filename=f'images/students/{student_photo_filename}')
                
                # Convert verified_status from INTEGER (0 or 1) to string 'pending' or 'verified'
                p_dict['verified_status'] = 'verified' if p_dict['verified_status'] == 1 else 'pending'
                
                participants_list.append(p_dict)
            
            return jsonify({'participants': participants_list})
        except Exception as e:
            print(f"Error fetching participants for exam {exam_id}: {e}")
            return jsonify({'message': 'Failed to fetch participants', 'error': str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify_student_alias():
    flash("Please go to an exam's management page to initiate student verification.", "warning")
    return redirect(url_for('list_exams'))

if __name__ == '__main__':
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/images/students', exist_ok=True)
    
    default_passport_path = os.path.join(app.static_folder, 'images', 'default-passport.jpg')
    if not os.path.exists(default_passport_path) and Image:
        print(f"Default passport image not found. Attempting to generate a placeholder at: {default_passport_path}")
        try:
            img = Image.new('RGB', (413, 531), color = (200, 200, 200))
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
                print("Could not load arial.ttf, using default Pillow font.")
            text = "No Photo"
            text_bbox = d.textbbox((0,0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (img.width - text_width) / 2
            y = (img.height - text_height) / 2
            d.text((x, y), text, fill=(100,100,100), font=font)
            img.save(default_passport_path)
            print(f"Generated placeholder: {default_passport_path}")
        except Exception as e:
            print(f"Error generating default-passport.jpg: {e}")
            print("Please ensure Pillow is installed (`pip install Pillow`) and check permissions.")


    app.run(host='0.0.0.0', port=5000, debug=True)


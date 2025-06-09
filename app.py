import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
import csv
import zipfile
import shutil
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import text # Import text for raw SQL queries

# Pillow is needed for generating a default passport image if one doesn't exist.
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None
    print("Pillow not installed. Cannot generate default-passport.jpg placeholder.")
    print("Please install Pillow: pip install Pillow")

app = Flask(__name__)

# --- Configuration for Production and Development ---
# Get secret key from environment variable, fallback for development
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-super-secret-key-development-only') 
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Configuration (MySQL)
# Get database URL from environment variable, fallback for local testing (adjust as needed)
# Example: mysql+pymysql://user:password@host:port/database_name
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:@localhost/attendance_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Disable tracking modifications for performance

db = SQLAlchemy(app)

# Face recognition setup
from insightface.app import FaceAnalysis
# It's good practice to ensure model download paths are handled if needed,
# though insightface usually handles this internally.
# For production, consider pre-downloading models to avoid runtime delays.
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# --- Database Models ---
class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(10), nullable=False)
    embedding_path = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<Student {self.student_id}>'

class Exam(db.Model):
    __tablename__ = 'exams'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(10), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='scheduled') # 'scheduled', 'active', 'completed'

    def __repr__(self):
        return f'<Exam {self.title}>'

class ExamParticipant(db.Model):
    __tablename__ = 'exam_participants'
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id', ondelete='CASCADE'), primary_key=True)
    student_id = db.Column(db.String(50), db.ForeignKey('students.student_id', ondelete='CASCADE'), primary_key=True)
    verified = db.Column(db.Integer, default=0) # 0 for pending, 1 for verified

    # Relationships
    exam = db.relationship('Exam', backref=db.backref('participants_link', lazy=True))
    student = db.relationship('Student', backref=db.backref('exams_participated', lazy=True))

    def __repr__(self):
        return f'<ExamParticipant ExamID:{self.exam_id} StudentID:{self.student_id}>'

class AttendanceLog(db.Model):
    __tablename__ = 'attendance_logs'
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id', ondelete='CASCADE'), nullable=False)
    student_id = db.Column(db.String(50), db.ForeignKey('students.student_id', ondelete='CASCADE'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    confidence = db.Column(db.Float, nullable=False)

    # Relationships
    exam = db.relationship('Exam', foreign_keys=[exam_id], backref=db.backref('attendance_records', lazy=True))
    student = db.relationship('Student', foreign_keys=[student_id], backref=db.backref('attendance_records', lazy=True))

    def __repr__(self):
        return f'<AttendanceLog ExamID:{self.exam_id} StudentID:{self.student_id} Timestamp:{self.timestamp}>'

# --- Database Initialization (using Flask-SQLAlchemy) ---
# This function will create tables based on models if they don't exist
# In production, you'd typically use Alembic for migrations, but for simplicity here,
# db.create_all() is used.
with app.app_context():
    db.create_all()

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
        # In a real app, use secure password hashing (e.g., bcrypt)
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
    # Use SQLAlchemy queries
    exams = Exam.query.order_by(Exam.start_time.desc()).limit(5).all()
    students = Student.query.order_by(Student.id.desc()).limit(5).all()
    
    return render_template('admin_dashboard.html', 
                           exams=exams,
                           students=students)

@app.route('/admin/students')
@login_required
def list_students():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '').strip()
    
    query = Student.query
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (Student.name.like(search_pattern)) | 
            (Student.student_id.like(search_pattern))
        )
    
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    students = pagination.items
    total_pages = pagination.pages
    
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
        students = Student.query.order_by(Student.name).all()
        
        csv_data = "Name,Student ID,Department,Level\n"
        csv_data += "\n".join(
            f'"{s.name}","{s.student_id}","{s.department}","{s.level}"' # Added quotes for CSV safety
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
                        student_id_from_filename = os.path.splitext(os.path.basename(file))[0]
                        student_id_from_filename = ''.join(c for c in student_id_from_filename if c.isalnum() or c in ('-', '_'))
                        
                        temp_path = os.path.join(images_dir, f"temp_{student_id_from_filename}.jpg")
                        with open(temp_path, 'wb') as f:
                            f.write(zip_ref.read(file))
                        
                        img = cv2.imread(temp_path)
                        if img is not None:
                            img = cv2.resize(img, (413, 531))
                            final_path = os.path.join(images_dir, f"{student_id_from_filename}.jpg")
                            cv2.imwrite(final_path, img)
                        os.remove(temp_path)
            
            students_processed = 0
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
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
                                existing_student = Student.query.filter_by(student_id=student_id).first()

                                if existing_student:
                                    existing_student.name = row['name']
                                    existing_student.department = row['department']
                                    existing_student.level = row['level']
                                    existing_student.embedding_path = embedding_output_path
                                    db.session.commit()
                                    flash(f'Student {student_id} updated.', 'info')
                                else:
                                    new_student = Student(
                                        name=row['name'], 
                                        student_id=student_id, 
                                        department=row['department'], 
                                        level=row['level'], 
                                        embedding_path=embedding_output_path
                                    )
                                    db.session.add(new_student)
                                    db.session.commit()
                                    students_processed += 1
                            except IntegrityError as ie:
                                db.session.rollback()
                                flash(f'Database integrity error for student {student_id}: {ie}', 'error')
                                print(f"Integrity Error: {ie} for student {student_id}")
                            except SQLAlchemyError as db_e:
                                db.session.rollback()
                                flash(f'Error saving student {student_id} to DB: {db_e}', 'error')
                                print(f"DB Save Error: {db_e} for student {student_id}")
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
    exam_details = Exam.query.get(exam_id)
    if not exam_details:
        flash(f"Exam with ID {exam_id} not found.", 'error')
        return redirect(url_for('list_exams'))
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
    exam = Exam.query.get(exam_id)
    if not exam:
        flash(f"Exam with ID {exam_id} not found.", 'error')
        return redirect(url_for('list_exams'))
    
    # Fetch participant_ids and their verification status
    participants = ExamParticipant.query.filter_by(exam_id=exam_id).all()
    participant_ids = [p.student_id for p in participants]
    verified_status = {p.student_id: p.verified for p in participants}
    
    students = Student.query.order_by(Student.name).all()
    
    return render_template('manage_exam.html', 
                           exam=exam,
                           students=students,
                           participant_ids=participant_ids,
                           verified_status=verified_status)

@app.route('/admin/exams/<int:exam_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_exam(exam_id):
    exam = Exam.query.get(exam_id)
    if not exam:
        flash('Exam not found!', 'error')
        return redirect(url_for('list_exams'))
        
    if request.method == 'POST':
        try:
            exam.title = request.form['title']
            exam.department = request.form['department']
            exam.level = request.form['level']
            start_time_str = request.form['start_time']
            end_time_str = request.form['end_time']

            new_start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
            new_end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')

            if new_end_time <= new_start_time:
                flash('End time must be after start time', 'error')
                return redirect(url_for('edit_exam', exam_id=exam_id))

            exam.start_time = new_start_time
            exam.end_time = new_end_time

            db.session.commit()
            flash('Exam updated successfully!', 'success')
            return redirect(url_for('manage_exam', exam_id=exam_id))
        except ValueError as e:
            flash(f'Invalid date format: {str(e)}. Please use YYYY-MM-DDTHH:MM.', 'error')
        except SQLAlchemyError as e:
            db.session.rollback()
            flash(f'Error updating exam: {str(e)}', 'error')
            print(f"Error updating exam ID {exam_id}: {e}")
        return redirect(url_for('edit_exam', exam_id=exam_id))

    return render_template('edit_exam.html', exam=exam)


@app.route('/admin/exams/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_exam(exam_id):
    """Delete an exam and its participants and attendance logs."""
    try:
        exam_to_delete = Exam.query.get(exam_id)
        if exam_to_delete:
            db.session.delete(exam_to_delete)
            db.session.commit()
            flash('Exam and associated data deleted successfully!', 'success')
        else:
            flash('Exam not found!', 'error')
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f'Error deleting exam: {str(e)}', 'error')
        print(f"Database error during exam deletion: {e}")
    return redirect(url_for('list_exams'))     

@app.route('/admin/exams/<int:exam_id>/add_students', methods=['POST'])
@login_required
def add_students_to_exam(exam_id):
    student_ids = request.form.getlist('student_ids')
    
    try:
        # Delete existing participants for this exam first
        ExamParticipant.query.filter_by(exam_id=exam_id).delete()
        
        # Add new participants
        for student_id in student_ids:
            # Check if student_id actually exists in the Student table before adding
            if Student.query.filter_by(student_id=student_id).first():
                new_participant = ExamParticipant(exam_id=exam_id, student_id=student_id, verified=0)
                db.session.add(new_participant)
            else:
                flash(f"Student with ID {student_id} not found in database. Skipping.", 'warning')
        
        db.session.commit()
        flash('Students updated for exam successfully!', 'success')
    except SQLAlchemyError as e:
        db.session.rollback()
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
            
            start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')

            if end_time <= start_time:
                flash('End time must be after start time', 'error')
                return redirect(url_for('add_exam'))
            
            new_exam = Exam(
                title=title, 
                department=department, 
                level=level, 
                start_time=start_time, 
                end_time=end_time, 
                status='scheduled'
            )
            db.session.add(new_exam)
            db.session.commit()
            
            flash('Exam added successfully!', 'success')
            return redirect(url_for('list_exams'))
            
        except ValueError as e:
            flash(f'Invalid date format: {str(e)}. Please use YYYY-MM-DDTHH:MM.', 'error')
        except SQLAlchemyError as e:
            db.session.rollback()
            flash(f'Error creating exam: {str(e)}', 'error')
            print(f"Error creating exam: {e}")
        return redirect(url_for('add_exam'))
    
    return render_template('add_exam.html')

@app.route('/admin/exams')
@login_required
def list_exams():
    exams = Exam.query.order_by(Exam.start_time.desc()).all()
    return render_template('exam_list.html', exams=exams)

@app.route('/api/exam/<int:exam_id>/verify', methods=['POST'])
@login_required
def verify_student_for_exam(exam_id): 
    try:
        exam = Exam.query.get(exam_id)
        if not exam or exam.status != 'active':
            return jsonify({'status': 'error', 'message': 'Exam is not active for verification.', 'bbox': None}), 400

        img_file = request.files.get('image')
        if not img_file:
            return jsonify({'status': 'error', 'message': 'No image provided for verification.', 'bbox': None}), 400

        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        faces = face_app.get(img)
        if not faces:
            return jsonify({'status': 'no_face_detected', 'message': 'No face detected in the image.', 'bbox': None})

        detected_face = faces[0]
        detected_embedding = detected_face.normed_embedding
        detected_bbox = [int(coord) for coord in detected_face.bbox.tolist()] 

        best_match = None
        best_score = 0.0
        VERIFICATION_THRESHOLD = 0.5 # Adjustable threshold

        students_for_exam_participants = ExamParticipant.query.filter_by(exam_id=exam_id).all()
        student_ids_in_exam = [p.student_id for p in students_for_exam_participants]
        
        # Eagerly load students for better performance in loop if many participants
        students_data = Student.query.filter(Student.student_id.in_(student_ids_in_exam)).all()
        student_map = {s.student_id: s for s in students_data}

        if not student_ids_in_exam:
            return jsonify({'status': 'error', 'message': 'No students registered for this exam.', 'bbox': detected_bbox})

        for participant in students_for_exam_participants:
            student = student_map.get(participant.student_id)
            if not student:
                print(f"Warning: Student object not found for participant {participant.student_id}")
                continue

            embedding_path = student.embedding_path
            if not os.path.exists(embedding_path):
                print(f"Warning: Embedding file not found for student {student.student_id}: {embedding_path}")
                continue

            try:
                stored_emb = np.load(embedding_path)
            except Exception as load_e:
                print(f"Error loading embedding for student {student.student_id}: {load_e}")
                continue

            score = np.dot(detected_embedding, stored_emb)
            
            if score > best_score:
                best_score = score
                best_match = student

        if best_match and best_score > VERIFICATION_THRESHOLD:
            participant_record = ExamParticipant.query.filter_by(
                exam_id=exam_id, student_id=best_match.student_id
            ).first()

            if participant_record and participant_record.verified == 1:
                return jsonify({
                    'status': 'already_verified',
                    'student_id': best_match.student_id,
                    'name': best_match.name,
                    'message': f"{best_match.name} ({best_match.student_id}) already verified for this exam.",
                    'confidence': float(best_score),
                    'bbox': detected_bbox
                })

            # Log verification
            new_log = AttendanceLog(
                exam_id=exam_id, 
                student_id=best_match.student_id, 
                confidence=float(best_score)
            )
            db.session.add(new_log)
            
            # Mark as verified in exam_participants
            if participant_record:
                participant_record.verified = 1
            else:
                # This case should ideally not happen if students_for_exam_participants was accurate
                # but adds robustness if a student was added to logs without being a participant first.
                # In a real app, you'd ensure this participant exists from exam_participants query.
                print(f"Warning: Participant record not found for {best_match.student_id} in exam {exam_id}. Creating new.")
                new_participant = ExamParticipant(exam_id=exam_id, student_id=best_match.student_id, verified=1)
                db.session.add(new_participant)

            db.session.commit()
            
            return jsonify({
                'status': 'verified',
                'student_id': best_match.student_id,
                'name': best_match.name,
                'confidence': float(best_score),
                'message': f"{best_match.name} ({best_match.student_id}) verified successfully!",
                'bbox': detected_bbox
            })
        else:
            return jsonify({
                'status': 'unknown', 
                'message': f'No matching student found or confidence too low ({best_score:.2f} < {VERIFICATION_THRESHOLD}).',
                'bbox': detected_bbox
            })

    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Database error during verification: {e}")
        return jsonify({'status': 'error', 'message': f'Server database error during verification: {str(e)}', 'bbox': None}), 500
    except Exception as e:
        print(f"Error during verification: {e}")
        return jsonify({'status': 'error', 'message': f'Server error during verification: {str(e)}', 'bbox': None}), 500

@app.route('/admin/exams/<int:exam_id>/start', methods=['POST'])
@login_required
def start_exam(exam_id):
    try:
        exam = Exam.query.get(exam_id)
        if exam:
            exam.status = 'active'
            db.session.commit()
            flash('Exam started successfully!', 'success')
        else:
            flash('Exam not found!', 'error')
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f'Error starting exam: {str(e)}', 'error')
    return redirect(url_for('manage_exam', exam_id=exam_id))

@app.route('/admin/exams/<int:exam_id>/end', methods=['POST'])
@login_required
def end_exam(exam_id):
    try:
        exam = Exam.query.get(exam_id)
        if exam:
            exam.status = 'completed'
            db.session.commit()
            flash('Exam ended successfully!', 'success')
        else:
            flash('Exam not found!', 'error')
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f'Error ending exam: {str(e)}', 'error')
    return redirect(url_for('manage_exam', exam_id=exam_id))

@app.route('/api/exam/<int:exam_id>/attendance_report')
@login_required
def attendance_report(exam_id):
    # Eagerly load relationships to avoid N+1 queries if accessing them in loop
    participants = ExamParticipant.query.filter_by(exam_id=exam_id).options(
        db.joinedload(ExamParticipant.student)
    ).all()

    csv_data = "Student ID,Name,Department,Level,Attended,Confidence,Verification Time\n"
    for p in participants:
        # Get latest attendance log for this participant in this exam
        latest_log = AttendanceLog.query.filter_by(exam_id=exam_id, student_id=p.student_id).order_by(AttendanceLog.timestamp.desc()).first()

        attended_status = "✓" if p.verified == 1 else "✗"
        confidence = f"{latest_log.confidence:.2f}" if latest_log else "N/A"
        verification_time_str = latest_log.timestamp.strftime('%Y-%m-%d %H:%M:%S') if latest_log else "N/A"

        csv_data += f'"{p.student.student_id}","{p.student.name}","{p.student.department}","{p.student.level}",' \
                    f'"{attended_status}","{confidence}","{verification_time_str}"\n' # Added quotes for CSV safety
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=exam_{exam_id}_attendance.csv"}
    )

@app.route('/api/exam/<int:exam_id>/attendance_stats')
@login_required
def attendance_stats(exam_id):
    """API endpoint to get attendance statistics for a given exam."""
    total_participants = ExamParticipant.query.filter_by(exam_id=exam_id).count()
    attended_count = ExamParticipant.query.filter_by(exam_id=exam_id, verified=1).count()
    
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
    try:
        # Eagerly load the related Student object to get name, student_id
        participants = ExamParticipant.query.filter_by(exam_id=exam_id).options(
            db.joinedload(ExamParticipant.student)
        ).all()

        participants_list = []
        for p in participants:
            # Construct photo URL based on student_id and known image path
            student_photo_filename = f"{p.student.student_id}.jpg"
            photo_url = url_for('static', filename=f'images/students/{student_photo_filename}')
            
            participants_list.append({
                'student_id': p.student.student_id,
                'name': p.student.name,
                'verified_status': 'verified' if p.verified == 1 else 'pending',
                'photo_url': photo_url
            })
        
        return jsonify({'participants': participants_list})
    except SQLAlchemyError as e:
        print(f"Database error fetching participants for exam {exam_id}: {e}")
        return jsonify({'message': 'Failed to fetch participants from database.', 'error': str(e)}), 500
    except Exception as e:
        print(f"Error fetching participants for exam {exam_id}: {e}")
        return jsonify({'message': 'Failed to fetch participants.', 'error': str(e)}), 500


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

    # In a production environment, you typically wouldn't run app.run() directly.
    # A WSGI server like Gunicorn or uWSGI would handle serving the application.
    app.run(host='0.0.0.0', port=5000, debug=True)


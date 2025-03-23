from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Database connection
def get_db_connection():
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

# Route to display attendance records
@app.route("/")
def index():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template("index.html", records=records)

# Route to fetch attendance records as JSON (for APIs)
@app.route("/api/attendance")
def get_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return jsonify([dict(record) for record in records])

if __name__ == "__main__":
    app.run(debug=True)
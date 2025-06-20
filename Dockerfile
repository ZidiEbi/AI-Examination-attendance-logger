# Use a slim Python base image for smaller size
FROM python:3.9-slim-buster

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for OpenCV, Pillow (fonts), and other libraries
# python3-opencv includes OpenCV dependencies
# libgl1-mesa-glx and libxext6 are often needed for GUI-less OpenCV operations (e.g., video processing)
# libsm6 and libxrender1 might be needed for InsightFace or other image libraries
# build-essential is for compiling packages with pip, will be removed later
# libfreetype6-dev is crucial for Pillow's font rendering capabilities
# fontconfig and fonts-dejavu-core provide necessary font configurations and actual font files for Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libxext6 \
        libsm6 \
        libxrender1 \
        ffmpeg \
        libfreetype6-dev \
        fontconfig \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y # Clean up build dependencies

# Copy the entire application code
COPY . .

# Create necessary directories at build time if they don't exist
# These will be used for uploads and embeddings
RUN mkdir -p uploads \
           embeddings \
           static/images/students \
           static/images \
    && chmod -R 777 uploads embeddings static/images/students # Adjust permissions as needed for your setup

# --- CRITICAL CHANGE HERE ---
# Instead of embedding long Python code directly, we copy and run a separate script.
# This avoids syntax errors caused by shell escaping.
COPY generate_default_image.py .
RUN python generate_default_image.py
# --- END CRITICAL CHANGE ---

# Expose the port your Flask app will run on
EXPOSE 5000

# Command to run the application using Gunicorn
# 'app:app' means the 'app' variable from the 'app.py' file
# --bind 0.0.0.0:5000 binds to all network interfaces on port 5000
# --workers 2 (or more based on CPU cores) for handling concurrent requests
# --timeout 120 for longer processing times for ML tasks
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]

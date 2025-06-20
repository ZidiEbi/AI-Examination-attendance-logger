FROM python:3.9-slim-buster

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=app.py

# Install system dependencies needed for Python packages and for download/unzip
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    wget \
    unzip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \ # ADDED THIS LINE TO FIX libGL.so.1 ERROR
    # Clean up apt lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y # Clean up build dependencies

# Create the root directory for InsightFace models within the uploads persistent volume
# This directory MUST match INSIGHTFACE_MODEL_ROOT (excluding the 'name' part) in app.py.
# The 'buffalo_l' folder will be created inside this by the unzip command.
RUN mkdir -p uploads/insightface_models

# Download and extract the insightface model directly into the designated path
# This prevents runtime download and startup timeouts.
# The zip extracts into a 'buffalo_l' subfolder, so 'unzip -d uploads/insightface_models/'
# will result in 'uploads/insightface_models/buffalo_l/...'
RUN wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O uploads/insightface_models/buffalo_l.zip \
    && unzip uploads/insightface_models/buffalo_l.zip -d uploads/insightface_models/ \
    && rm uploads/insightface_models/buffalo_l.zip # Clean up the zip file

# Copy your application code and other necessary files
COPY . .

# Create persistent storage directories if they don't exist
# chmod -R 777 is generally not recommended for production due to security risks,
# but might be necessary for certain environments if specific user/group permissions are not set.
RUN mkdir -p uploads \
           embeddings \
           static/images/students \
           static/images \
    && chmod -R 777 uploads embeddings static/images/students # Adjust permissions as needed for your setup

# Generate default image (this Python script is already in the image from COPY . .)
# Make sure generate_default_image.py is correctly copied before this RUN command.
COPY generate_default_image.py .
RUN python generate_default_image.py

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "120"]

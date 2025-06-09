# Use a slim Python base image for smaller size
FROM python:3.9-slim-buster

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for OpenCV and other libraries
# python3-opencv includes OpenCV dependencies
# libgl1-mesa-glx and libxext6 are often needed for GUI-less OpenCV operations (e.g., video processing)
# libsm6 and libxrender1 might be needed for InsightFace or other image libraries
# build-essential is for compiling packages with pip, will be removed later
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libxext6 \
        libsm6 \
        libxrender1 \
        ffmpeg \
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


# Create the default passport image if Pillow is available
# This ensures it's baked into the image, reducing runtime cold start/error
RUN python -c "import os; from PIL import Image, ImageDraw, ImageFont; \
    default_passport_path = 'static/images/default-passport.jpg'; \
    if not os.path.exists(default_passport_path): \
        print('Generating default passport image...'); \
        img = Image.new('RGB', (413, 531), color=(200, 200, 200)); \
        d = ImageDraw.Draw(img); \
        try: font = ImageFont.truetype('arial.ttf', 40); \
        except IOError: font = ImageFont.load_default(); \
        text_bbox = d.textbbox((0,0), 'No Photo', font=font); \
        text_width = text_bbox[2] - text_bbox[0]; \
        text_height = text_bbox[3] - text_bbox[1]; \
        x = (img.width - text_width) / 2; \
        y = (img.height - text_height) / 2; \
        d.text((x, y), 'No Photo', fill=(100,100,100), font=font); \
        img.save(default_passport_path); \
        print('Default passport image generated.'); \
    else: print('Default passport image already exists.');"

# Expose the port your Flask app will run on
EXPOSE 5000

# Command to run the application using Gunicorn
# 'app:app' means the 'app' variable from the 'app.py' file
# --bind 0.0.0.0:5000 binds to all network interfaces on port 5000
# --workers 2 (or more based on CPU cores) for handling concurrent requests
# --timeout 120 for longer processing times for ML tasks
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]

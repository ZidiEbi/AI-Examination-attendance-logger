import os
from PIL import Image, ImageDraw, ImageFont

def generate_default_passport_image(output_path):
    """
    Generates a placeholder default-passport.jpg image if it does not exist.
    This image is used as a fallback for student photos.
    """
    if not os.path.exists(output_path):
        print(f"Generating default passport image at: {output_path}")
        try:
            # Standard passport photo aspect ratio (width x height)
            img = Image.new('RGB', (413, 531), color=(200, 200, 200))
            d = ImageDraw.Draw(img)

            # Try to use a common font, fallback to default Pillow font
            try:
                # Arial is usually available on Debian-based systems after fonts-dejavu-core
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
                print("Could not load arial.ttf, using default Pillow font.")

            text = "No Photo"
            # Get text size and position it in the center
            text_bbox = d.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (img.width - text_width) / 2
            y = (img.height - text_height) / 2

            d.text((x, y), text, fill=(100, 100, 100), font=font)
            img.save(output_path)
            print("Default passport image generated successfully.")
        except Exception as e:
            print(f"Error generating default-passport.jpg: {e}")
            print("Please ensure Pillow is installed and necessary font libraries are present.")
    else:
        print(f"Default passport image already exists at: {output_path}")

if __name__ == "__main__":
    # Define the output path relative to the Docker WORKDIR (/app)
    # The static/images directory is created earlier in the Dockerfile
    default_image_path = 'static/images/default-passport.jpg'
    generate_default_passport_image(default_image_path)

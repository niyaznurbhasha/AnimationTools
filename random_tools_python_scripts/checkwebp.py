from PIL import Image
import os

# Set the directory containing the .webp files
directory = '16_webp/'  # Replace with your actual directory path

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith('.webp'):  # Process only .webp files
        try:
            # Full file path
            file_path = os.path.join(directory, filename)
            
            # Open the image file
            img = Image.open(file_path)
            
            # Convert the filename to .png
            png_filename = filename.rsplit('.', 1)[0] + '.png'
            output_path = os.path.join(directory, png_filename)
            
            # Save as PNG
            img.save(output_path, 'PNG')
            print(f'Converted {filename} to {png_filename}')
        except Exception as e:
            print(f'Error converting {filename}: {e}')

print('Conversion complete.')

from PIL import Image
import os

def remove_alpha_channel(input_path, output_path):
    # Open the input image
    img = Image.open(input_path)

    # Check if the image has an alpha channel
    if img.mode in ('RGBA', 'LA'):
        # Convert the image to RGB
        rgb_img = img.convert('RGB')
    else:
        # If the image does not have an alpha channel, keep it as is
        rgb_img = img

    # Save the image to the output path
    rgb_img.save(output_path)
    print(f"Saved RGB image to {output_path}")
# Example usage:
input_image_path = '/Users/niyaz/Documents/TEXTurePaper/textures/alexander_face.png'
output_image_path =  '/Users/niyaz/Documents/TEXTurePaper/textures/alexander_face_alpha_removed.png'

remove_alpha_channel(input_image_path, output_image_path)

# If you want to process multiple images in a directory, you can use the following code:

""" def remove_alpha_channel_from_directory(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Remove alpha channel from the image
        remove_alpha_channel(input_path, output_path)

# Example usage for a directory:
input_directory = 'path_to_your_input_directory'
output_directory = 'path_to_your_output_directory'

remove_alpha_channel_from_directory(input_directory, output_directory)
 """
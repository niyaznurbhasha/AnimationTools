import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import shutil

# Function to get union of white pixels across multiple images
def get_union_white_pixels(image_paths):
    union_mask = None
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img_np = np.array(img)
        
        binary_mask = (img_np >= 240).astype(np.uint8)  # Threshold for white
        
        if union_mask is None:
            union_mask = binary_mask
        else:
            union_mask = np.logical_or(union_mask, binary_mask)
    
    return union_mask

# Function to count white pixels in a mask
def count_white_pixels(mask):
    return np.sum(mask)

# Function to inpaint the image using Stable Diffusion
def inpaint_image(base_image_path, mask, output_path, pipe):
    base_image = Image.open(base_image_path).convert('RGB')
    base_image = base_image.resize((512, 512))  # Resize to match model input size
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize((512, 512))
    
    prompt = "a 360 degree view of an alternative dimension, a dmt universe, trippy, detailed, fine lines, sci fi style of chris foss, cartoon style of herge"  # Customize your inpainting prompt
    
    result = pipe(prompt=prompt, image=base_image, mask_image=mask_image).images[0]
    result.save(output_path)

# Function to process folders and apply inpainting only when necessary
def process_folders(main_folder, base_image_path, output_folder, pipe):
    previous_white_pixel_count = -1
    previous_output_image = None
    
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_names = set()
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.png'):
                    image_names.add(filename)
            
            for image_name in sorted(image_names):  # Ensure sequential processing
                image_paths = []
                
                for subfolder in os.listdir(main_folder):
                    subfolder_path = os.path.join(main_folder, subfolder)
                    image_path = os.path.join(subfolder_path, image_name)
                    if os.path.isfile(image_path):
                        image_paths.append(image_path)
                
                if image_paths:
                    mask = get_union_white_pixels(image_paths)
                    white_pixel_count = count_white_pixels(mask)
                    
                    output_image_path = os.path.join(output_folder, image_name)
                    
                    if white_pixel_count == previous_white_pixel_count:
                        # No new white pixels, copy previous output
                        if previous_output_image is not None:
                            shutil.copy(previous_output_image, output_image_path)
                    else:
                        # New white pixels, apply inpainting
                        inpaint_image(base_image_path, mask, output_image_path, pipe)
                        previous_white_pixel_count = white_pixel_count
                        previous_output_image = output_image_path

# Paths with your data
main_folder = '/Users/niyaz/Downloads/test_stems/test_scribble/output_midi/scribble_output/'
base_image_path = "/Users/niyaz/Downloads/trippy_env2.png"
output_folder = '/Users/niyaz/Downloads/test_stems/diffusion_inpainting/'
audio_file = '/Users/niyaz/Downloads/test_scribble.mp3'
final_output_video = '/Users/niyaz/Downloads/test_stems/diffusion_inpainting/final_output_video.mp4'

# Load Stable Diffusion inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")  # Use GPU if available

# Process the folders and generate the inpainted images
process_folders(main_folder, base_image_path, output_folder, pipe)

# FFmpeg command to combine images into a video at 25 fps
os.system(f'ffmpeg -r 25 -i {output_folder}/scribble_%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {final_output_video}')

# FFmpeg command to combine video with audio
os.system(f'ffmpeg -i {final_output_video} -i {audio_file} -c:v copy -c:a aac -strict experimental {final_output_video}')

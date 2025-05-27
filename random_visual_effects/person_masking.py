import os
import pixellib
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg
import cv2
import numpy as np


def load_frames(image_dir):
    frames = []
    frames_files = []
    for file in sorted(os.listdir(image_dir)):
        if(file.endswith(".png")):
            image = cv2.imread(image_dir + file)
            image =  cv2.resize(image, (3840, 2160))
            frames.append(image)
            frames_files.append(image_dir + file)
        
    return frames, frames_files
# Replace 'path_to_your_image.jpg' with the path to your wormhole image
frames,frames_files = load_frames('/Users/niyaz/Downloads/dancing_frames/')

# Save the frames to files (optional)
#for i, frame in enumerate(frames):
#    cv2.imwrite(f'frame_{i}.jpg', frame)

image_dir = "/Users/niyaz/Documents/maua-style/test_shimmering/"
goaldir = '/Users/niyaz/Documents/maua-style/test_shimmering_out/'
#os.mkdir(goaldir)
model_path = "/Users/niyaz/Downloads/mask_rcnn_coco.h5"
#change_bg = alter_bg(model_type = "pb")
#change_bg.load_pascalvoc_model(model_path)

# Initialize PixelLib instance segmentation
segment_image = instance_segmentation()
segment_image.load_model(model_path)  # COCO model path

character_image = cv2.imread('/Users/niyaz/Downloads/wormhole.png')
character_image_resized = cv2.resize(character_image, (3840, 2160))# (3840, 2160))

count = 0
for file in sorted(os.listdir(image_dir)):
    if(file.endswith(".png")):
        name = file.split(".")[0]
        image = cv2.resize(cv2.imread(image_dir + file),(3840, 2160))
        
        #image = cv2.imread('/Users/niyaz/Documents/neural-style-pt/examples/inputs/me3.png')

        # Perform instance segmentation
        output,nums=segment_image.segmentImage(frames_files[count], show_bboxes=False)
       # output,nums=segment_image.segmentImage('/Users/niyaz/Documents/neural-style-pt/examples/inputs/me3.png', show_bboxes=False)

        # Now, you can access 'masks' and 'class_ids' from the 'output' dictionary
        masks = output['masks']
        class_ids = output['class_ids']

        # Create a mask for all 'person' segments (class ID for person in COCO is 1)
        person_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        for i, class_id in enumerate(class_ids):
            if class_id == 1:  # COCO class ID for 'person'
                person_mask = np.logical_or(person_mask, masks[:, :, i])

        # Create a blank image with the same dimensions as the original image
        modified_image = np.zeros_like(image)

        # Set 'person' pixels to white (255, 255, 255) and others to black (0, 0, 0)
        modified_image[person_mask] = image[person_mask]#character_image_resized[person_mask]#[255, 255, 255]
     
        modified_image[~person_mask] = [255,255,255] #frames[count][~person_mask]#[0, 0, 0]

        # Save or display the modified image
        cv2.imwrite(goaldir + "out" + str(count) + ".png", modified_image)
        count+=1

       # change_bg.blur_bg(image_dir + file, extreme = True, detect = "person", output_image_name = goaldir + name + "_blurred.png")

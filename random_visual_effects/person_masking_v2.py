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
            image =  cv2.resize(image, (1080, 1920))
            frames.append(image)
            frames_files.append(image_dir + file)
        
    return frames, frames_files

frames, frames_files=load_frames("/Users/niyaz/Documents/test_mau/1024/")
frames2, frames_files2=load_frames("/Users/niyaz/Documents/test_mau/1024/")

image_dir = "/Users/niyaz/Downloads/truth_1/"
goaldir = "/Users/niyaz/Downloads/truth_1_masked4/"
os.mkdir(goaldir)
model_path = "/Users/niyaz/Downloads/mask_rcnn_coco.h5"
#change_bg = alter_bg(model_type = "pb")
#change_bg.load_pascalvoc_model(model_path)

# Initialize PixelLib instance segmentation
segment_image = instance_segmentation()
segment_image.load_model(model_path)  # COCO model path



count2 = 25
count=0
for file in sorted(os.listdir(image_dir)):
    if(file.endswith(".png")):
        name = file.split(".")[0]
        image = cv2.imread(image_dir + file)
        
        #image = cv2.imread('/Users/niyaz/Documents/neural-style-pt/examples/inputs/me3.png')

        # Perform instance segmentation
        output,nums=segment_image.segmentImage(image_dir + file, show_bboxes=False)
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
        modified_image[person_mask] = frames[int(count/2)][person_mask]#frames[255,255,255]#character_image_resized[person_mask]#[255, 255, 255]
     
        modified_image[~person_mask] = frames2[int(count)][~person_mask]#[0,0,0] #frames[count][~person_mask]#[0, 0, 0]

        # Save or display the modified image
        cv2.imwrite(goaldir + "out" + str(count) + ".png", modified_image)
        count+=1
        count2+=1

       # change_bg.blur_bg(image_dir + file, extreme = True, detect = "person", output_image_name = goaldir + name + "_blurred.png")

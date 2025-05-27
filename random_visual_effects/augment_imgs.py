import imgaug.augmenters as iaa
import imageio
import numpy as np
import os
loadir = "/Users/niyaz/Downloads/"
savedir =  "/Users/niyaz/Downloads/aug_char1_4/"
os.mkdir(savedir)
# Load your images (assuming they are in the current working directory and are jpg files)
image_paths = [loadir + "char1_set1_1.png",loadir + "char1_set1_2.png",loadir + "char1_set1_3.png",loadir + "char1_set1_4.png",]
images = [imageio.imread(path) for path in image_paths]

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Fliplr(1.0)), # 50% chance to apply the flip, fully flipped when applied
    iaa.Sometimes(0.5, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ))
])

# Number of augmented versions to generate per original image
num_augmented_versions = 100

# Generate and save augmented images
for idx, image in enumerate(images):
    for aug_idx in range(num_augmented_versions):
        augmented_image = seq(image=image)  # Apply augmentations
        if augmented_image.shape[2] == 4:  # Convert RGBA to RGB if necessary
            augmented_image = augmented_image[:, :, :3]
        # Save the augmented image with a unique filename
        imageio.imwrite(savedir + f'augmented_image_{idx}_{aug_idx}.png', augmented_image)

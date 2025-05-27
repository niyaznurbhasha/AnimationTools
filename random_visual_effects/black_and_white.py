from PIL import Image
import numpy as np
img = Image.open('/Users/niyaz/Downloads/char1_3pose_2.png')
thresh = 200
fn = lambda x : 255 if x > thresh else 0
r = img.convert('L').point(fn, mode='1')
r.save('foo.png')
r = img.convert('1')
r.save('foo2.png')

image_path = 'foo.png'  # Make sure to update this path
img = Image.open(image_path).convert('L')  # Convert to grayscale

# Convert to a NumPy array
img_array = np.array(img)

# Invert the pixel values
# For a strictly black and white image, you can use:
inverted_img_array = np.where(img_array==255, 0, 255)

# Alternatively, for a more general approach:
# inverted_img_array = 255 - img_array

# Convert back to an image
inverted_img = Image.fromarray(inverted_img_array.astype('uint8'))

# Display or save the inverted image
inverted_img.save("foo3.png")
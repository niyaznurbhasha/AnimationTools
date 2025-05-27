import cv2
import numpy as np
from PIL import Image
import torch
import io
from rembg import remove
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load & resize input
img = cv2.imread("Subject3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
H, W = img.shape[:2]

# 2) Load & process manual shirt mask
shirt_mask = cv2.imread("shirt_mask.png", cv2.IMREAD_GRAYSCALE)
shirt_mask = cv2.resize(shirt_mask, (W, H))
_, shirt_bin = cv2.threshold(shirt_mask, 127, 1, cv2.THRESH_BINARY)
shirt_weight = shirt_bin.astype(np.float32)

# 3) Extract person alpha with rembg
rgba = remove(cv2.imencode(".png", img)[1].tobytes())
fg = Image.open(io.BytesIO(rgba)).convert("RGBA")
fg_np = np.array(fg)
alpha = fg_np[:, :, 3].astype(np.float32) / 255.0
alpha = cv2.GaussianBlur(alpha, (41, 41), 0)

# 4) Build a smooth weight map (shirt OR person)
weight_map = np.maximum(alpha, shirt_weight)
weight_map = cv2.GaussianBlur(weight_map, (101, 101), 0)
weight_map = np.clip(weight_map, 0.0, 1.0)
weight_map_3 = np.stack([weight_map] * 3, axis=-1)

# 5) Inpaint your shirt into a professional suit
inp_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
inp_pipe.set_progress_bar_config(disable=True)

suit = inp_pipe(
    prompt=(
        "a tailored charcoal gray wool suit jacket with satin lapels, "
        "crisp white dress shirt, professional headshot lighting"
    ),
    image=Image.fromarray(img),
    mask_image=Image.fromarray((shirt_bin * 255).astype(np.uint8)).convert("L"),
    guidance_scale=7.5,
    num_inference_steps=10
).images[0]
suit_np = np.array(suit)

# 6) Merge suit and original image
combined_rgb = np.where(shirt_weight[:, :, None] > 0.5, suit_np, img)

# 7) Generate a neutral office background
bg_pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
bg_pipe.set_progress_bar_config(disable=True)

bg = bg_pipe(
    prompt=(
        "a modern office interior with clean lines and soft neutral tones, "
        "completely out of focus, true bokeh effect"
    ),
    height=512, width=512,
    guidance_scale=7.5,
    num_inference_steps=10
).images[0]
bg_np = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

# ——— HEAVY BACKGROUND BLUR ———
# apply a very strong blur (sigmaX=100) to wipe away all detail
bg_np = cv2.GaussianBlur(bg_np, (0, 0), sigmaX=100)

# 8) Composite person over the blurred background
combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
composite = (combined_bgr * weight_map_3 + bg_np * (1 - weight_map_3)).astype(np.uint8)

# 9) Remove any halo seam via OpenCV inpainting
person_bin = (weight_map > 0.5).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
edge = cv2.morphologyEx(person_bin, cv2.MORPH_GRADIENT, kernel)
inpaint_mask = cv2.dilate(edge, kernel, iterations=1)
final = cv2.inpaint(composite, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# 10) Save
cv2.imwrite("final_profile.png", final)
print("✅ Saved fully out‑of‑focus background version as final_profile.png")

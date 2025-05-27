import os
import cv2
import glob
import subprocess
import numpy as np
from PIL import Image
import torch
import argparse

import torchvision.transforms as T
import timm

# -- IMPORTS FROM YOUR COMPOSITING UTILS --
from compositing_utils import (
    apply_lighting_adjustment,
    apply_color_contrast_adjustment,
    apply_perspective_transform,
    apply_depth_of_field,
    apply_shadows_and_reflections,
    apply_edge_blending,
    apply_motion_blur,
    apply_atmospheric_effect,
    apply_local_geometry_adjustment
)

########################################
# STABLE DIFFUSION & SAM imports
########################################
from diffusers import StableDiffusionPipeline
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

########################################
# Utility Functions
########################################

def upscale_texture(input_path, output_path, model_path, upscale_factor):
    cmd = [
        "python", "Real-ESRGAN/inference_realesrgan.py",
        "-n", "realesr-general-x4v3",
        "-i", input_path,
        "-o", output_path,
        "--outscale", str(upscale_factor),
        "--model_path", model_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Upscaled texture saved to {output_path}.")

def load_images(directory):
    image_paths = glob.glob(os.path.join(directory, "*.*"))
    images = []
    for ip in image_paths:
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is not None:
            images.append((ip, img))
    return images

def generate_background(prompt, model, device, width=512, height=512):
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False)  # Disable safety checker
    result = pipe(prompt=prompt, width=width, height=height, guidance_scale=7.5, num_inference_steps=50)
    # Convert PIL to OpenCV BGR
    return np.array(result.images[0])[:, :, ::-1]

def resize_with_aspect_ratio(image, width=512):
    h, w = image.shape[:2]
    aspect = h / w
    new_h = int(width * aspect)
    resized = cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)
    return resized

def run_sam_on_image(image, sam_checkpoint, device):
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)
    return masks

def extract_object_image(obj):
    img = obj["original_image"]
    mask = obj["mask"].astype(np.uint8)
    x, y, w, h = obj["bbox"]
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    return cropped_img, cropped_mask

def window_closed(window_name):
    prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    if prop < 1:
        return True
    return False

########################################
# Classification with ImageNet-21k
########################################
def load_imagenet21k_labels(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]
    return labels

def classify_with_in21k_model(img, model, transform, labels, topk=5):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = transform(pil_img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_idxs = probs.topk(topk, dim=1)
    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_idxs = top_idxs.squeeze(0).cpu().numpy()
    top_labels = [labels[i] for i in top_idxs]
    return top_labels, top_probs

def object_matches_segment_list(top_labels, segment_list):
    # Simple substring match, case-insensitive
    segment_list_lower = [s.lower() for s in segment_list]
    for lbl in top_labels:
        lbl_lower = lbl.lower()
        if any(s in lbl_lower for s in segment_list_lower):
            return True
    return False

########################################
# Simple Overlay Placement (No Blending)
########################################

def overlay_object_on_bg(bg_img, obj_img, obj_mask, center_x, center_y):
    """
    Overlays obj_img onto bg_img at the specified center. No advanced blending—just direct overlay.
    
    :param bg_img: Background image (H,W,3).
    :param obj_img: Object image (h,w,3).
    :param obj_mask: Binary mask (h,w).
    :param center_x: X-coordinate in bg_img to place the object's center.
    :param center_y: Y-coordinate in bg_img to place the object's center.
    :return: Modified bg_img with the object overlaid.
    """
    out_bg = bg_img.copy()
    h_bg, w_bg = out_bg.shape[:2]
    h_obj, w_obj = obj_img.shape[:2]

    # Compute top-left corner
    x1 = center_x - w_obj // 2
    y1 = center_y - h_obj // 2
    x2 = x1 + w_obj
    y2 = y1 + h_obj

    # Clip to background bounds
    if x1 < 0: 
        x_offset = -x1
        x1 = 0
    else:
        x_offset = 0
    if y1 < 0:
        y_offset = -y1
        y1 = 0
    else:
        y_offset = 0

    x2 = min(x2, w_bg)
    y2 = min(y2, h_bg)

    # The region of the object that actually fits
    obj_region = obj_img[y_offset:y_offset + (y2 - y1), x_offset:x_offset + (x2 - x1)]
    mask_region = obj_mask[y_offset:y_offset + (y2 - y1), x_offset:x_offset + (x2 - x1)]

    # Overlay
    roi = out_bg[y1:y2, x1:x2]
    roi[mask_region > 0] = obj_region[mask_region > 0]
    out_bg[y1:y2, x1:x2] = roi

    return out_bg

########################################
# Interactive Placement
########################################
def choose_placement_interactive(
    background, 
    objects, 
    args,
    device="cuda"
):
    if len(objects) == 0:
        print("No objects available for placement.")
        return background

    print("Number of objects:", len(objects))

    history = []
    current_bg = background.copy()

    cv2.namedWindow("Place Objects", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Place Objects", min(current_bg.shape[1], 1024), min(current_bg.shape[0], 768))

    obj_index = 0
    while obj_index < len(objects):
        if window_closed("Place Objects"):
            print("Window closed. Exiting.")
            break

        obj = objects[obj_index]
        obj_img, obj_mask = extract_object_image(obj)

        print(f"Placing object {obj_index+1}/{len(objects)} from {obj['image_path']}")
        print("Click on background to place object.")
        print("Press 'n' after placing to finalize placement and move on.")
        print("Press 's' to skip object, 'b' to go back one step, 'q' to finish now, 'r' to restart.")
        click_position = [None, None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_position[0] = x
                click_position[1] = y

        cv2.setMouseCallback("Place Objects", mouse_callback)

        while True:
            if window_closed("Place Objects"):
                print("Window closed. Exiting.")
                obj_index = len(objects)
                break

            display = current_bg.copy()
            if click_position[0] is not None and click_position[1] is not None:
                cx, cy = click_position

                # ------------------------------------------------
                #  APPLY COMPOSITING STEPS BEFORE OVERLAY
                #  (We use the user-selected args flags)
                # ------------------------------------------------
                # We apply each step on a copy of the object
                preview_obj = obj_img.copy()
                preview_mask = obj_mask.copy()

                # 1) LIGHTING
                if args.lighting:
                    # We'll pass in a local patch of background around (cx,cy) if you like
                    # For a simple approach, pass the entire background
                    preview_obj = apply_lighting_adjustment(
                        preview_obj,
                        background,
                        preview_mask,
                        use_intrinsic_decomposition=False
                    )

                # 2) COLOR
                if args.color:
                    preview_obj = apply_color_contrast_adjustment(
                        preview_obj,
                        background,
                        preview_mask,
                        local_patch_size=50
                    )

                # 3) PERSPECTIVE
                # This requires a homography or something. 
                # For demonstration, we skip unless you have a matrix from somewhere.
                if args.perspective:
                    print("Perspective arg is True, but no homography logic is implemented here.")
                    # Example usage if you had a matrix: 
                    # H = np.eye(3) # dummy
                    # preview_obj, preview_mask = apply_perspective_transform(
                    #     preview_obj, preview_mask, H
                    # )

                # 4) DEPTH OF FIELD
                if args.dof:
                    # This requires a background depth map or a guess
                    # We'll do a placeholder
                    preview_obj = apply_depth_of_field(
                        preview_obj,
                        preview_mask,
                        source_depth=20.0,
                        background_depth_map=None,
                        focal_plane=10.0,
                        dof_strength=1.0
                    )

                # 5) SHADOWS & REFLECTIONS
                if args.shadows:
                    # We can't apply shadows directly onto 'preview_obj' unless we place it on the composite.
                    # Typically you do shadows at the final stage. 
                    # So we skip or do a placeholder:
                    print("Shadows & reflections are typically done after object is placed in a composite.")

                # 6) EDGE BLENDING
                if args.edge_blending:
                    preview_obj = apply_edge_blending(
                        preview_obj,
                        preview_mask,
                        blend_radius=5
                    )

                # 7) MOTION BLUR
                if args.motion_blur:
                    preview_obj = apply_motion_blur(
                        preview_obj,
                        preview_mask,
                        blur_direction="horizontal",
                        blur_magnitude=10
                    )

                # 8) ATMOSPHERIC
                if args.atmospheric:
                    preview_obj = apply_atmospheric_effect(
                        preview_obj,
                        preview_mask,
                        distance=50.0,
                        haze_color=(200,200,200),
                        haze_strength=0.3
                    )

                # 9) LOCAL GEOMETRY
                if args.local_geometry:
                    # Typically need a normal map or advanced logic
                    preview_obj = apply_local_geometry_adjustment(
                        preview_obj,
                        preview_mask,
                        normal_map=None
                    )

                # Now overlay for preview
                preview_bg = overlay_object_on_bg(display, preview_obj, preview_mask, cx, cy)
                cv2.imshow("Place Objects", preview_bg)
            else:
                # No click yet, just show the background
                cv2.imshow("Place Objects", display)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('n'):
                if click_position[0] is not None and click_position[1] is not None:
                    cx, cy = click_position

                    # -- Final compositing steps, re-run on final object:
                    final_obj = obj_img.copy()
                    final_mask = obj_mask.copy()

                    if args.lighting:
                        final_obj = apply_lighting_adjustment(final_obj, background, final_mask, False)
                    if args.color:
                        final_obj = apply_color_contrast_adjustment(final_obj, background, final_mask, 50)
                    if args.perspective:
                        print("No actual perspective transform being applied. Placeholder only.")
                    if args.dof:
                        final_obj = apply_depth_of_field(final_obj, final_mask, 20.0, None, 10.0, 1.0)
                    if args.edge_blending:
                        final_obj = apply_edge_blending(final_obj, final_mask, 5)
                    if args.motion_blur:
                        final_obj = apply_motion_blur(final_obj, final_mask, "horizontal", 10)
                    if args.atmospheric:
                        final_obj = apply_atmospheric_effect(final_obj, final_mask, 50.0, (200,200,200), 0.3)
                    if args.local_geometry:
                        final_obj = apply_local_geometry_adjustment(final_obj, final_mask, None)

                    # Overly simple approach to shadows:
                    if args.shadows:
                        print("Shadows & reflections not fully implemented in direct overlay. Skipping here.")

                    new_bg = overlay_object_on_bg(current_bg, final_obj, final_mask, cx, cy)
                    history.append(current_bg)
                    current_bg = new_bg
                    print("Object placed.")
                    obj_index += 1
                    break
                else:
                    print("Please click to place object before pressing 'n'.")
            elif key == ord('s'):
                print("Skipping object.")
                obj_index += 1
                break
            elif key == ord('b'):
                if history:
                    current_bg = history.pop()
                    obj_index = max(obj_index - 1, 0)
                    print("Went back one step.")
                    break
                else:
                    print("No history to go back to.")
            elif key == ord('q'):
                print("Finalizing early.")
                obj_index = len(objects)
                break
            elif key == ord('r'):
                if history:
                    current_bg = history[0]
                    history.clear()
                    obj_index = 0
                    print("Restarted placement.")
                    break
                else:
                    print("Already at start.")
            elif key == 27:  # ESC
                print("Exiting placement.")
                obj_index = len(objects)
                break

    cv2.destroyWindow("Place Objects")
    return current_bg

########################################
# Main
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compositing script with optional advanced compositing steps.")
    parser.add_argument("--input_dir", type=str, default="fusion_inputs/", help="Directory with input images.")
    parser.add_argument("--output_dir", type=str, default="fusion_outputs", help="Directory to save results.")
    parser.add_argument("--background_mode", type=str, default="file", choices=["stable_diffusion", "file"], help="Background mode.")
    parser.add_argument("--background_image_index", type=int, default=0, help="If background_mode=file, which image index to use.")
    parser.add_argument("--background_prompt", type=str, default="A tranquil lakeside scene at sunset, photorealistic", help="Prompt if generating background.")
    parser.add_argument("--upscale_factor", type=int, default=4, help="Upscale factor for ESRGAN.")
    parser.add_argument("--segment_list", type=str, default="dog", help="Comma-separated list of objects to segment/filter.")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint.")
    parser.add_argument("--stable_diffusion_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Stable Diffusion model.")
    parser.add_argument("--esrgan_model_path", type=str, default="Real-ESRGAN/weights/realesr-general-x4v3.pth", help="Path to ESRGAN weights.")
    parser.add_argument("--imagenet21k_map", type=str, default="imagenet21k_map.txt", help="Path to ImageNet-21k lemmas file.")

    # -------- NEW ARGS FOR COMPOSITING STEPS --------
    parser.add_argument("--lighting", action="store_true", help="Apply lighting adjustment.")
    parser.add_argument("--color", action="store_true", help="Apply color/contrast adjustment.")
    parser.add_argument("--perspective", action="store_true", help="Apply perspective transform (requires homography).")
    parser.add_argument("--dof", action="store_true", help="Apply depth-of-field blur on the object.")
    parser.add_argument("--shadows", action="store_true", help="Apply shadows/reflections (placeholder).")
    parser.add_argument("--edge_blending", action="store_true", help="Apply edge blending/matting.")
    parser.add_argument("--motion_blur", action="store_true", help="Apply motion blur.")
    parser.add_argument("--atmospheric", action="store_true", help="Apply fog/haze on the object.")
    parser.add_argument("--local_geometry", action="store_true", help="Apply local geometry/normal map adjustments.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    segment_list = [s.strip() for s in args.segment_list.split(",") if s.strip()]

    images = load_images(args.input_dir)
    if len(images) == 0:
        raise ValueError("No images found in input_dir.")

    # Setup classification model
    model_name = "vit_base_patch16_224_in21k"
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(DEVICE)

    labels = load_imagenet21k_labels(args.imagenet21k_map)
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    if args.background_mode == "file":
        if args.background_image_index < 0 or args.background_image_index >= len(images):
            raise ValueError("background_image_index out of range.")
        bg_path, bg_img = images[args.background_image_index]
        background = bg_img.copy()
    else:
        # Generate background with Stable Diffusion
        background = generate_background(args.background_prompt, args.stable_diffusion_model, DEVICE)

    # Resize background
    background = resize_with_aspect_ratio(background, width=512)

    # Upscale background
    temp_bg_path = os.path.join(args.output_dir, "background_temp.png")
    cv2.imwrite(temp_bg_path, background)
    upscaled_bg_path = os.path.join(args.output_dir, "background_upscaled.png")
    upscale_texture(temp_bg_path, upscaled_bg_path, args.esrgan_model_path, args.upscale_factor)

    # The Real-ESRGAN script might produce "background_temp_out.png" or similar, check your actual output name
    # For the example, let's assume it is "background_temp_out.png"
    upscaled_bg_final = os.path.join(upscaled_bg_path, "background_temp_out.png")
    if os.path.isfile(upscaled_bg_final):
        background = cv2.imread(upscaled_bg_final, cv2.IMREAD_COLOR)
    else:
        # fallback if no sub-output was created
        background = cv2.imread(upscaled_bg_path, cv2.IMREAD_COLOR)

    # Gather objects from the images (excluding the chosen background if file-based)
    if args.background_mode == "file":
        images_for_objs = [(p, i) for (p, i) in images if p != bg_path]
    else:
        images_for_objs = images

    # Segment all images with SAM, classify, and keep only those matching segment_list
    all_objects = []
    for (ip, img) in images_for_objs:
        masks = run_sam_on_image(img, args.sam_checkpoint, DEVICE)
        for m in masks:
            segmentation = m["segmentation"]
            bbox = m["bbox"]
            area = m["area"]
            obj_crop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            top_labels, _ = classify_with_in21k_model(obj_crop, model, transform, labels)
            if segment_list and not object_matches_segment_list(top_labels, segment_list):
                continue
            obj = {
                "image_path": ip,
                "original_image": img,
                "mask": segmentation,
                "bbox": bbox,
                "area": area
            }
            all_objects.append(obj)

    # Sort objects by area (largest first)
    all_objects.sort(key=lambda x: x["area"], reverse=True)
    objects = all_objects

    if not objects:
        print("No objects matched the segment list. Exiting.")
    else:
        print(f"Total objects to place: {len(objects)}")

    final_image = choose_placement_interactive(background, objects, args, device=DEVICE)
    if final_image is not None:
        temp_result_path = os.path.join(args.output_dir, "final_result.png")
        cv2.imwrite(temp_result_path, final_image)
        print("Done. Final composite image saved at:", temp_result_path)
    else:
        print("No final image produced (window closed early or no objects placed).")

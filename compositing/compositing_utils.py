# compositing_utils.py

import cv2
import numpy as np
from typing import Optional, Dict, Any

########################################
# 1. LIGHTING
########################################
def apply_lighting_adjustment(
    source_img: np.ndarray,
    background_img: np.ndarray,
    mask: np.ndarray,
    use_intrinsic_decomposition: bool = False
) -> np.ndarray:
    """
    Adjusts the lighting of source_img to match local lighting in background_img.
    
    :param source_img: The BGR source object image.
    :param background_img: The BGR background image (or local patch) where the object is placed.
    :param mask: Binary mask (same size as source_img) indicating the object region to be adjusted.
    :param use_intrinsic_decomposition: If True, tries an advanced pipeline (e.g., with decomp_diffusion).
    :return: A copy of source_img with updated lighting.
    """
    # Example: Basic brightness matching approach
    # 1) Compute average brightness in background vs. source
    if not use_intrinsic_decomposition:
        # Convert to grayscale for brightness approximation
        bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        
        # Only measure brightness in masked region if you want *source* region stats
        src_values = src_gray[mask > 0]
        
        # If you have a local patch of background, measure brightness there too
        # For example, you can define a region of interest around where the source will be placed
        # But here we just do a simplistic full image approach
        bg_values = bg_gray.flatten()

        # Avoid zero-division
        if len(src_values) == 0 or len(bg_values) == 0:
            return source_img

        src_mean = np.mean(src_values)
        bg_mean = np.mean(bg_values)

        # scale factor
        if src_mean > 0:
            scale = bg_mean / src_mean
        else:
            scale = 1.0

        # apply scale
        adjusted = source_img.astype(np.float32) * scale
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        # Use the mask to blend: only update the object region
        out = source_img.copy()
        out[mask > 0] = adjusted[mask > 0]
        return out
    else:
        # Placeholder for advanced pipeline:
        # 1) Run decomp_diffusion or similar intrinsic decomposition on background + source
        # 2) Transfer shading
        # 3) Reconstruct source with new shading
        # Stub code:
        print("Using advanced intrinsic decomposition approach... (Not implemented)")
        return source_img


########################################
# 2. COLOR & CONTRAST
########################################
def apply_color_contrast_adjustment(
    source_img: np.ndarray,
    background_img: np.ndarray,
    mask: np.ndarray,
    local_patch_size: int = 50
) -> np.ndarray:
    """
    Matches color and contrast of source_img to the background using local histogram or color transfer.
    
    :param source_img: The BGR source object image.
    :param background_img: The BGR background image (or local patch) near where object is placed.
    :param mask: Binary mask indicating the object region.
    :param local_patch_size: Size of local background patch to analyze for local color stats.
    :return: Source image updated with color/contrast adjustments.
    """

    # Example approach: histogram matching on the masked source region 
    # to the entire background_img or a local patch of it.

    # Convert to LAB for color matching (or do direct BGR hist matching)
    # Stub for a direct BGR histogram matching on the entire background:
    def match_histograms(source, reference):
        matched = source.copy()
        for ch in range(3):
            s = source[:,:,ch].ravel()
            r = reference[:,:,ch].ravel()

            s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
            r_values, r_counts = np.unique(r, return_counts=True)

            s_cdf = np.cumsum(s_counts).astype(np.float64)
            s_cdf /= s_cdf[-1]

            r_cdf = np.cumsum(r_counts).astype(np.float64)
            r_cdf /= r_cdf[-1]

            interp_values = np.interp(s_cdf, r_cdf, r_values)
            matched[:,:,ch] = interp_values[bin_idx].reshape(source[:,:,ch].shape).astype(np.uint8)
        return matched

    # Just do a naive approach for demonstration:
    matched = match_histograms(source_img, background_img)

    out = source_img.copy()
    out[mask > 0] = matched[mask > 0]
    return out


########################################
# 3. PERSPECTIVE & SCALE
########################################
def apply_perspective_transform(
    source_img: np.ndarray,
    source_mask: np.ndarray,
    homography_matrix: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Warps source_img and source_mask using a given homography matrix.
    
    :param source_img: The BGR source object.
    :param source_mask: Binary mask for the source.
    :param homography_matrix: 3x3 homography matrix from cv2.findHomography().
    :return: (warped_img, warped_mask).
    """
    h, w = source_img.shape[:2]
    warped_img = cv2.warpPerspective(source_img, homography_matrix, (w, h))
    warped_mask = cv2.warpPerspective(source_mask, homography_matrix, (w, h))
    return warped_img, warped_mask


def compute_homography_from_points(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> Optional[np.ndarray]:
    """
    Computes a homography matrix from matched src_points/dst_points.
    Expects shape (N,2) for each.
    
    :param src_points: Nx2 array of points in the source image.
    :param dst_points: Nx2 array of points in the destination (background).
    :return: 3x3 homography matrix or None if the computation fails.
    """
    if len(src_points) < 4:
        print("Need at least 4 points to compute a homography.")
        return None
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H


########################################
# 4. DEPTH OF FIELD
########################################
def apply_depth_of_field(
    source_img: np.ndarray,
    source_mask: np.ndarray,
    source_depth: float,
    background_depth_map: np.ndarray,
    focal_plane: float,
    dof_strength: float = 1.0
) -> np.ndarray:
    """
    Applies synthetic blur to source_img based on difference between source_depth and focal_plane,
    referencing background_depth_map as an example of how focus changes with distance.
    
    :param source_img: BGR image of the object.
    :param source_mask: Binary mask for the object.
    :param source_depth: Approximate depth of the object.
    :param background_depth_map: Depth map for the background (not always trivial to get).
    :param focal_plane: Depth at which things are in perfect focus.
    :param dof_strength: Multiplier for how strong the blur out-of-focus areas get.
    :return: Source image with simulated DOF blur.
    """
    # Basic approach: The further from focal_plane, the stronger the blur radius
    distance = abs(source_depth - focal_plane)
    max_blur = int(distance * dof_strength)

    if max_blur < 1:
        # In focus, no blur
        return source_img

    # apply blur just inside the mask
    blurred = cv2.GaussianBlur(source_img, (max_blur*2+1, max_blur*2+1), 0)
    out = source_img.copy()
    out[source_mask > 0] = blurred[source_mask > 0]
    return out


########################################
# 5. SHADOWS & REFLECTIONS
########################################
def apply_shadows_and_reflections(
    composite_img: np.ndarray,
    source_img: np.ndarray,
    source_mask: np.ndarray,
    light_direction: tuple = (1, 0),
    shadow_opacity: float = 0.5,
    reflection_opacity: float = 0.3
) -> np.ndarray:
    """
    Very rough placeholder for adding a shadow and reflection of the source to the composite.
    
    :param composite_img: Current background + source composite.
    :param source_img: The object image (BGR).
    :param source_mask: Binary mask for the object.
    :param light_direction: (dx, dy) direction from which light is coming.
    :param shadow_opacity: 0 to 1, how dark the shadow is.
    :param reflection_opacity: 0 to 1, how visible the reflection is.
    :return: composite_img with shadow (and reflection if relevant).
    """
    # 1) Shadows
    # A basic approach: shift the source_mask by some offset determined by light_direction, darken it, overlay it.
    h, w = composite_img.shape[:2]

    dx = int(light_direction[0] * 30)  # arbitrary scale
    dy = int(light_direction[1] * 30)

    shadow_mask = np.roll(source_mask, shift=(dy, dx), axis=(0, 1))
    # zero out edges rolled in
    if dy > 0:
        shadow_mask[:dy, :] = 0
    elif dy < 0:
        shadow_mask[h+dy:, :] = 0
    if dx > 0:
        shadow_mask[:, :dx] = 0
    elif dx < 0:
        shadow_mask[:, w+dx:] = 0

    shadow_layer = np.zeros_like(composite_img)
    shadow_color = (0, 0, 0)  # black
    shadow_layer[shadow_mask > 0] = shadow_color
    composite_with_shadow = cv2.addWeighted(composite_img, 1.0, shadow_layer, shadow_opacity, 0)

    # 2) Reflections
    # Example: flip source vertically below it, fade out
    # This only makes sense if there's a "floor" or "water" below the object
    # For simplicity, assume reflection is a vertical flip under the source
    # and the bottom is at the object's bounding box
    y_indices, x_indices = np.where(source_mask > 0)
    if len(y_indices) > 0:
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # reflection region below y_max
        reflection_height = (y_max - y_min) // 2
        if (y_max + reflection_height) < h:
            # create reflection
            reflection_img = cv2.flip(source_img, 0)  # vertical flip
            reflection_mask = cv2.flip(source_mask, 0)
            # place reflection in composite
            y_start = y_max
            y_end = y_max + (y_max - y_min)
            region = composite_with_shadow[y_start:y_end, :]
            # alpha blend reflection in
            # For actual accuracy, you'd offset it to match a plane, but here's a naive approach:
            for yy in range(reflection_mask.shape[0]):
                for xx in range(reflection_mask.shape[1]):
                    if reflection_mask[yy, xx] > 0 and (y_start+yy) < h:
                        alpha = reflection_opacity * (1.0 - (yy / reflection_mask.shape[0]))
                        composite_with_shadow[y_start+yy, xx] = cv2.addWeighted(
                            composite_with_shadow[y_start+yy, xx].astype(np.float32),
                            1.0,
                            reflection_img[yy, xx].astype(np.float32),
                            alpha,
                            0
                        ).astype(np.uint8)

    return composite_with_shadow


########################################
# 6. EDGE BLENDING / MATTING
########################################
def apply_edge_blending(
    composite_img: np.ndarray,
    object_mask: np.ndarray,
    blend_radius: int = 5
) -> np.ndarray:
    """
    Simple edge feathering or matting to soften the boundary between object and background.
    
    :param composite_img: The already composited image.
    :param object_mask: Binary mask for the object (same size as composite).
    :param blend_radius: Radius for blurring mask edges.
    :return: Image with softened edges.
    """
    # This is a simplistic approach: blur the mask, then alpha-blend
    # In a real scenario, you'd do advanced matting or Poisson blending.
    blurred_mask = cv2.GaussianBlur(object_mask.astype(np.float32), (blend_radius*2+1, blend_radius*2+1), 0)
    blurred_mask = np.clip(blurred_mask, 0, 1)

    # Create an alpha channel from blurred_mask
    alpha = np.zeros((composite_img.shape[0], composite_img.shape[1], 1), dtype=np.float32)
    alpha[:,:,0] = blurred_mask

    # The composite_img already has the object. One approach:
    # - Re-blend the object region from a separate "object-only" image. This gets complicated.
    # For demonstration, let's just lighten the edges slightly to mimic minimal feathering:
    feathered_edges = composite_img.copy().astype(np.float32)
    # Example: darken edges or lighten them
    # We'll simply reduce contrast near edges
    # (In practice, you'd keep a separate 'foreground' + 'background' for a real alpha blend.)
    # This is purely a stub:
    factor = 0.8
    mask_3ch = np.repeat((1.0 - alpha), 3, axis=2)
    feathered_edges = feathered_edges * (1.0 - 0.2 * mask_3ch)
    output = feathered_edges.astype(np.uint8)

    return output


########################################
# 7. MOTION BLUR
########################################
def apply_motion_blur(
    source_img: np.ndarray,
    source_mask: np.ndarray,
    blur_direction: str = "horizontal",
    blur_magnitude: int = 10
) -> np.ndarray:
    """
    Applies a synthetic motion blur to the source object.
    
    :param source_img: BGR object image.
    :param source_mask: Binary mask for the object.
    :param blur_direction: "horizontal", "vertical", or "custom" to define kernel.
    :param blur_magnitude: Strength of the blur (kernel size).
    :return: Blurred object image.
    """
    kernel_size = blur_magnitude
    if kernel_size < 3:
        return source_img  # negligible blur

    # build a simple motion blur kernel
    if blur_direction == "horizontal":
        kernel = np.zeros((1, kernel_size), dtype=np.float32)
        kernel[0, :] = 1.0 / kernel_size
    elif blur_direction == "vertical":
        kernel = np.zeros((kernel_size, 1), dtype=np.float32)
        kernel[:, 0] = 1.0 / kernel_size
    else:
        # custom approach, e.g. diagonal kernel
        kernel = np.eye(kernel_size, dtype=np.float32) / kernel_size

    blurred = cv2.filter2D(source_img, -1, kernel)
    out = source_img.copy()
    out[source_mask > 0] = blurred[source_mask > 0]
    return out


########################################
# 8. ATMOSPHERIC EFFECTS (Fog/Haze)
########################################
def apply_atmospheric_effect(
    source_img: np.ndarray,
    source_mask: np.ndarray,
    distance: float = 50.0,
    haze_color: tuple = (200, 200, 200),
    haze_strength: float = 0.3
) -> np.ndarray:
    """
    Adds a haze/fog effect to the source object based on approximate distance.
    
    :param source_img: BGR object image.
    :param source_mask: Binary mask for object.
    :param distance: Approximate distance from camera (bigger distance => more haze).
    :param haze_color: BGR color of the haze.
    :param haze_strength: 0 to 1, how strongly the haze covers the source.
    :return: Source with haze effect over masked region.
    """
    # scale haze effect by distance
    alpha = min(distance/100.0, 1.0) * haze_strength

    haze_layer = np.full_like(source_img, haze_color, dtype=np.uint8)
    # alpha blend
    source_float = source_img.astype(np.float32)
    haze_float = haze_layer.astype(np.float32)
    out = source_img.copy().astype(np.float32)

    mask_bin = source_mask > 0
    out[mask_bin] = cv2.addWeighted(
        source_float[mask_bin], (1.0 - alpha),
        haze_float[mask_bin], alpha,
        0
    )
    return out.astype(np.uint8)


########################################
# 9. LOCAL GEOMETRY / NORMALS
########################################
def apply_local_geometry_adjustment(
    source_img: np.ndarray,
    source_mask: np.ndarray,
    normal_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Placeholder function for advanced re-lighting or geometry-based transformations 
    using normal maps or partial 3D info.
    
    :param source_img: BGR object image.
    :param source_mask: Binary mask for object.
    :param normal_map: Optional normal map (H,W,3) for the object or background region.
    :return: Potentially re-lit or geometry-adjusted source image.
    """
    # In practice, you'd do advanced computations 
    # (like shading adjustments based on normals + light direction)
    if normal_map is None:
        print("No normal map provided; skipping geometry-based adjustments.")
        return source_img
    # Stub
    print("Applying geometry-based lighting or distortion... (Not implemented)")
    return source_img


########################################
# 10. MASTER FUNCTION: apply_all_compositing
########################################
def apply_all_compositing(
    source_img: np.ndarray,
    source_mask: np.ndarray,
    background_img: np.ndarray,
    config: Dict[str, Any] = {}
) -> np.ndarray:
    """
    Master function that sequentially applies any subset of the compositing 
    considerations, depending on config flags.
    
    :param source_img: The object image (BGR).
    :param source_mask: Binary mask for the object.
    :param background_img: The background or local patch for reference.
    :param config: A dictionary of flags/parameters for each step, e.g.:
       {
         "lighting": True,
         "color_contrast": True,
         "perspective": { "homography": some_matrix, ... },
         "depth_of_field": { "depth": 20.0, "focal_plane": 10.0, ... },
         ...
       }
    :return: Updated source_img after all chosen compositing adjustments.
    """
    out = source_img.copy()
    mask = source_mask.copy()

    # 1) Lighting
    if config.get("lighting", False):
        out = apply_lighting_adjustment(
            out, 
            background_img,
            mask,
            use_intrinsic_decomposition=config.get("use_intrinsic_decomposition", False)
        )

    # 2) Color & Contrast
    if config.get("color_contrast", False):
        out = apply_color_contrast_adjustment(
            out, 
            background_img,
            mask,
            local_patch_size=config.get("local_patch_size", 50)
        )

    # 3) Perspective & Scale
    if "perspective" in config:
        persp_cfg = config["perspective"]
        H = persp_cfg.get("homography")
        if H is not None:
            out, mask = apply_perspective_transform(out, mask, H)
            # scale could be done here as well if needed

    # 4) Depth of Field
    if "depth_of_field" in config:
        dof_cfg = config["depth_of_field"]
        out = apply_depth_of_field(
            out,
            mask,
            dof_cfg.get("source_depth", 20.0),
            dof_cfg.get("background_depth_map", None),
            dof_cfg.get("focal_plane", 10.0),
            dof_cfg.get("dof_strength", 1.0)
        )

    # 5) Shadows & Reflections
    if config.get("shadows_reflections", False):
        # This typically modifies the *composite*, not just the source.
        # So you'd pass in the current composite. For now, let's just do it on the source (stub).
        out_temp = np.zeros_like(background_img)
        # "Paste" the source onto a blank canvas at some location:
        # For a real pipeline, you'd place the object in the final composite first.
        # We'll assume 0,0 top-left for demonstration:
        h_src, w_src = out.shape[:2]
        out_temp[0:h_src, 0:w_src] = out
        # We also need a mask for that region:
        shadow_mask = np.zeros_like(mask)
        shadow_mask[0:h_src, 0:w_src] = mask
        # Then apply shadows on out_temp:
        out_temp = apply_shadows_and_reflections(
            out_temp, out, shadow_mask,
            light_direction=(1, 1),
            shadow_opacity=0.5,
            reflection_opacity=0.3
        )
        # The result is in out_temp, which might need to be extracted back out. 
        # This is just a placeholder. 
        out = out

    # 6) Edge Blending / Matting
    if config.get("edge_blending", False):
        # If we already have a final composite, we'd do the matting there.
        # As a placeholder, just soften the edges in the source itself:
        out = apply_edge_blending(out, mask, blend_radius=config.get("blend_radius", 5))

    # 7) Motion Blur
    if "motion_blur" in config:
        blur_cfg = config["motion_blur"]
        out = apply_motion_blur(
            out,
            mask,
            blur_direction=blur_cfg.get("direction", "horizontal"),
            blur_magnitude=blur_cfg.get("magnitude", 10)
        )

    # 8) Atmospheric (Fog/Haze)
    if "atmospheric" in config:
        atm_cfg = config["atmospheric"]
        out = apply_atmospheric_effect(
            out,
            mask,
            distance=atm_cfg.get("distance", 50.0),
            haze_color=atm_cfg.get("haze_color", (200,200,200)),
            haze_strength=atm_cfg.get("haze_strength", 0.3)
        )

    # 9) Local Geometry / Normals
    if "local_geometry" in config:
        geom_cfg = config["local_geometry"]
        normal_map = geom_cfg.get("normal_map", None)
        out = apply_local_geometry_adjustment(
            out,
            mask,
            normal_map=normal_map
        )

    # Return the final adjusted source + updated mask
    return out, mask


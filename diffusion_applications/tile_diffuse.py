#!/usr/bin/env python
"""
Standalone script to run tiled diffusion on an input image of any dimensions.
If no input image is provided, a blank image is generated using the provided
width and height values.
"""

import argparse
import math
import numpy as np
import os
import subprocess
import tempfile
from PIL import Image


def tile_diffuse(input_img_path, prompt, steps, image_model, diffusion_device,
                 tile_size=512, overlap=64, base_seed=42):
    """
    Splits the input image into overlapping tiles, runs the external diffusion
    script on each tile, and blends the results together.
    
    Args:
        input_img_path (str): Path to the input image.
        prompt (str): Text prompt for the diffusion process.
        steps (int): Number of diffusion steps.
        image_model (str): The diffusion image model to use.
        diffusion_device (str): Device to run the diffusion on (e.g., 'cuda').
        tile_size (int, optional): Size (in pixels) of each tile. Defaults to 512.
        overlap (int, optional): Number of overlapping pixels between adjacent tiles. Defaults to 64.
        base_seed (int, optional): Base random seed to use. Defaults to 42.
        
    Returns:
        PIL.Image: The final diffused image.
    """
    # Open the input image (of any dimensions)
    img = Image.open(input_img_path).convert("RGB")
    width, height = img.size

    # Prepare canvases for weighted blending
    output_canvas = np.zeros((height, width, 3), dtype=np.float32)
    weight_accumulator = np.zeros((height, width), dtype=np.float32)

    # Calculate number of tiles in x and y directions
    x_steps = math.ceil((width - overlap) / (tile_size - overlap))
    y_steps = math.ceil((height - overlap) / (tile_size - overlap))
    print(f"Tiling image into {x_steps} columns and {y_steps} rows...")

    # Process each tile
    for i in range(y_steps):
        for j in range(x_steps):
            x0 = j * (tile_size - overlap)
            y0 = i * (tile_size - overlap)
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tile_width = x1 - x0
            tile_height = y1 - y0

            # Crop the current tile from the image
            tile = img.crop((x0, y0, x1, y1))

            # Optionally vary the seed per tile (here we use the same seed)
            tile_seed = base_seed  # You could modify this if desired.

            # Save the tile to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_tile_in:
                tile_in_path = temp_tile_in.name
                tile.save(tile_in_path)
            tile_out_path = tile_in_path.replace(".png", "_diffused.png")

            # Build the command to run the external diffusion script
            cmd = [
                "python",
                "depth_txt2img.py",
                prompt,
                tile_in_path,
                tile_out_path,
                "--steps", str(steps),
                "--image-model", image_model,
                "--device", diffusion_device
                # You can add "--seed", str(tile_seed) here if your diffusion script accepts it
            ]
            print(f"Diffusing tile ({i}, {j}) at ({x0},{y0}) -> ({x1},{y1}) with seed {tile_seed}.")
            subprocess.run(cmd, check=True)

            # Open the diffused tile
            diffused_tile = Image.open(tile_out_path).convert("RGB")
            diffused_tile_np = np.array(diffused_tile).astype(np.float32)

            # Create a smooth blending window using a Hanning function
            window = np.outer(np.hanning(tile_height), np.hanning(tile_width))
            output_canvas[y0:y1, x0:x1, :] += diffused_tile_np * window[..., None]
            weight_accumulator[y0:y1, x0:x1] += window

            # Clean up temporary files
            os.remove(tile_in_path)
            os.remove(tile_out_path)

    # Avoid division by zero in areas with no overlap
    weight_accumulator = np.clip(weight_accumulator, 1e-6, None)
    # Normalize to produce the final image
    final_diffused = (output_canvas / weight_accumulator[..., None]).astype(np.uint8)
    return Image.fromarray(final_diffused)


def main():
    parser = argparse.ArgumentParser(
        description="Run tiled diffusion on an input image of arbitrary dimensions. "
                    "If no input image is provided, a blank image is generated.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to the input image file. If not provided, a blank image is generated."
    )
    parser.add_argument(
        "--width", type=int, default=1024,
        help="Width of the generated blank image (if --input is not provided)."
    )
    parser.add_argument(
        "--height", type=int, default=1024,
        help="Height of the generated blank image (if --input is not provided)."
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Tile size (in pixels) for diffusion."
    )
    parser.add_argument(
        "--overlap", type=int, default=96,
        help="Overlap (in pixels) between adjacent tiles."
    )
    parser.add_argument(
        "--steps", type=int, default=12,
        help="Number of diffusion steps."
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for the diffusion process."
    )
    parser.add_argument(
        "--image-model", type=str, default="Lykon/dreamshaper-8",
        help="Diffusion image model to use."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run diffusion (e.g., 'cuda')."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for the diffusion process."
    )
    parser.add_argument(
        "--output", type=str, default="diffused_output.png",
        help="File path to save the final diffused image."
    )
    args = parser.parse_args()

    # Determine the image to process.
    if args.input:
        input_path = args.input
    else:
        # Generate a blank white image with the specified dimensions if no input is provided.
        blank_image = Image.new("RGB", (args.width, args.height), (255, 255, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
            input_path = temp_img_file.name
            blank_image.save(input_path)

    # Run the tiled diffusion on the input (or generated) image.
    diffused_image = tile_diffuse(
        input_path,
        prompt=args.prompt,
        steps=args.steps,
        image_model=args.image_model,
        diffusion_device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        base_seed=args.seed,
    )

    # Save the final diffused image.
    diffused_image.save(args.output)
    print(f"Diffused image saved to {args.output}")

    # Clean up the temporary blank image file if one was created.
    if not args.input:
        os.remove(input_path)


if __name__ == "__main__":
    main()

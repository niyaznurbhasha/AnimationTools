import argparse
import os
import torch
import numpy as np
import imageio
from PIL import Image
import cv2

import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    FoVPerspectiveCameras,
    PointLights,
    BlendParams,
)

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_mesh_renderer(image_size=512, device=None):
    if device is None:
        device = get_device()
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # Make sure the background is RGBA = (0,0,0,0) so alpha=0 in the background
    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, blend_params=blend_params),
    )
    return renderer

def render_cow(
    cow_path="data/cow.obj",
    image_size=256,
    num_frames=120,
    duration=4,
    background_path=None,
    static_background=True,
    output_file="output.mp4",
    output_format="mp4",
    device=None,
):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    mesh = load_objs_as_meshes([cow_path], device=device)

    background_image_pil = None
    if background_path:
        print("YES")
        background_image_pil = Image.open(background_path).convert("RGB")
        background_image_pil = background_image_pil.resize((image_size, image_size))

    frames = []
    for i in range(num_frames):
        theta = 360 * i / num_frames

        R = torch.tensor(
            [
                [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
                [0.0, 1.0, 0.0],
                [np.sin(np.radians(theta)), 0.0,  np.cos(np.radians(theta))],
            ],
            dtype=torch.float32,
            device=device,
        )
        T = torch.tensor([[0, 0, 3]], dtype=torch.float32, device=device)

        cameras = FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
        lights = PointLights(location=[[0, 0, -3]], device=device)

        # Render returns shape (1, H, W, 4)
        rendered = renderer(mesh, cameras=cameras, lights=lights)
        rendered = rendered[0].cpu().numpy()  # [H, W, 4]

        rgb = rendered[..., :3]  # [H, W, 3]
        alpha = rendered[..., 3:4]  # [H, W, 1]

        if background_image_pil:
            # static or dynamic background
            if static_background:
                bg = background_image_pil
            else:
                # rotate the background if dynamic
                bg = background_image_pil.rotate(theta)
            bg_np = np.array(bg).astype(np.float32) / 255.0  # [H, W, 3]

            # alpha composite
            out = alpha * rgb + (1 - alpha) * bg_np
        else:
            out = rgb

        out = (out * 255).astype(np.uint8)
        frames.append(out)

    # Make sure the directory for output_file exists
    output_file = os.path.abspath(output_file)  # get full path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save
    fps = num_frames / duration
    if output_format.lower() == "mp4":
        save_as_mp4(frames, output_file, fps)
    else:
        frame_duration = duration / num_frames
        save_as_gif(frames, output_file, frame_duration)

def save_as_mp4(renders, output_file, fps):
    import cv2
    height, width, _ = renders[0].shape

    possible_fourccs = ["avc1", "mp4v", "X264", "H264", "XVID", "MJPG"]
    success = False
    for fourcc_str in possible_fourccs:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        if video.isOpened():
            print(f"Using fourcc='{fourcc_str}' to write the video.")
            for frame in renders:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(bgr)
            video.release()
            success = True
            break
        else:
            print(f"Failed to open VideoWriter with fourcc='{fourcc_str}'.")

    if not success:
        raise RuntimeError(
            "All attempted FourCC codes failed. You may need to install a suitable codec or use an alternative method."
        )

    print(f"MP4 saved to {output_file}")


def save_as_gif(renders, output_file, frame_duration):
    imageio.mimsave(output_file, renders, duration=frame_duration)
    print(f"GIF saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="/mnt/c/Users/niyaz/Documents/ready_characters/tripo/soldier2/mesh-tex.obj")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--output_file", type=str, default="360_vids/360_robot_render.mp4")
    parser.add_argument("--output_format", type=str, choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--image_size", type=int, default =2048)
    parser.add_argument("--background_path", type=str, default="/mnt/c/Users/niyaz/Downloads/forest_bg.jpg")
    parser.add_argument("--static_background", action="store_true")
    args = parser.parse_args()

    render_cow(
        cow_path=args.cow_path,
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        background_path=args.background_path,
        static_background=args.static_background,
        output_file=args.output_file,
        output_format=args.output_format,
    )
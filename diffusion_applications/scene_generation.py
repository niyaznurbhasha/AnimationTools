import os, cv2, argparse, subprocess, imageio, torch, numpy as np
import trimesh, pyrender
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from pygltflib import GLTF2


# ───────────────────────── 1. FBX → glTF ──────────────────────────
def convert_fbx_to_gltf(fbx_path: str) -> str:
    glb_path = os.path.splitext(fbx_path)[0] + ".glb"
    if not os.path.exists(glb_path):
        exe = os.path.join(os.path.dirname(__file__), "FBX2glTF.exe")
        subprocess.run([exe, "--binary", fbx_path, "-o", glb_path], check=True)
    return glb_path


# ─────────────────── 2. Extract root-node translations ─────────────
def extract_animation_transforms(glb_path: str):
    gltf = GLTF2().load(glb_path)
    anim = gltf.animations[0]
    sampler = anim.samplers[0]
    buf = gltf.binary_blob()

    inp_acc = gltf.accessors[sampler.input]
    out_acc = gltf.accessors[sampler.output]
    off_i = gltf.bufferViews[inp_acc.bufferView].byteOffset or 0
    times = np.frombuffer(buf, np.float32, inp_acc.count, off_i)

    off_o = gltf.bufferViews[out_acc.bufferView].byteOffset or 0
    vals = np.frombuffer(buf, np.float32, out_acc.count * 3, off_o)
    trans = vals.reshape(-1, 3)
    return times, trans


# ────────────────────────── 3. Lighting ───────────────────────────
def estimate_lighting(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g >= np.percentile(g, 90))
    if xs.size == 0:
        vec = np.array([0, -1, 0])
    else:
        cx, cy = xs.mean(), ys.mean()
        h, w = g.shape
        vec = np.array([cx - w / 2, cy - h / 2, 0])
        vec /= (np.linalg.norm(vec) + 1e-8)
    return {"dir": vec, "ambient": 0.5}


def apply_character_lighting(rgba, light):
    img = rgba.astype(np.float32) / 255.0
    shade = light["ambient"] + (1 - light["ambient"]) * max(np.array([0, 0, 1]).dot(light["dir"]), 0)
    img[..., :3] *= shade
    return (img * 255).astype(np.uint8)


def generate_contact_shadow(mask, light_dir, target_shape):
    """Return a shadow mask the same H×W as the background."""
    # ── blur & shift ───────────────────────────
    dx, dy = int(light_dir[0] * 20), int(light_dir[1] * 20)
    sh = cv2.warpAffine(mask, np.float32([[1, 0, dx], [0, 1, dy]]),
                        (mask.shape[1], mask.shape[0]))
    sh = cv2.GaussianBlur(sh.astype(np.float32), (31, 31), 10)
    if sh.max() > 0:
        sh = (sh / sh.max()) * 0.5

    # ── resize to bg size if needed ────────────
    h, w = target_shape
    if sh.shape[:2] != (h, w):
        sh = cv2.resize(sh, (w, h), interpolation=cv2.INTER_LINEAR)

    return sh.astype(np.float32)


def poisson_composite(bg, fg_rgb, mask_float, thresh=1e-3):
    """Robust seamless-clone. Falls back to alpha blend on any OpenCV error."""
    h, w = bg.shape[:2]

    # --- resize fg & mask to full frame ------------------------------
    if fg_rgb.shape[:2] != (h, w):
        fg_rgb = cv2.resize(fg_rgb, (w, h), cv2.INTER_AREA)
    if mask_float.shape[:2] != (h, w):
        mask_float = cv2.resize(mask_float, (w, h), cv2.INTER_NEAREST)

    # --- binarise mask (uint8, 0/255) --------------------------------
    bin_mask = (mask_float > thresh).astype(np.uint8) * 255

    # bail if mask empty
    if bin_mask.sum() == 0:
        return bg.copy()

    # --- shrink ROI 1 px so it never touches edge --------------------
    ys, xs = np.where(bin_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    if y0 == 0: y0 += 1
    if x0 == 0: x0 += 1
    if y1 == h - 1: y1 -= 1
    if x1 == w - 1: x1 -= 1
    bin_mask[:y0, :] = 0
    bin_mask[y1 + 1:, :] = 0
    bin_mask[:, :x0] = 0
    bin_mask[:, x1 + 1:] = 0

    try:
        center = (w // 2, h // 2)
        return cv2.seamlessClone(fg_rgb, bg, bin_mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # ---- fallback: simple alpha blend ---------------------------
        alpha = (bin_mask / 255.0)[..., None]
        return (fg_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)


# ─────────────────── 6. Temporal Smoothing ────────────────────────
def temporal_smooth(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    remap_x = (gx - flow[..., 0]).astype(np.float32)
    remap_y = (gy - flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev, remap_x, remap_y, cv2.INTER_LINEAR)
    return cv2.addWeighted(warped, 0.5, curr, 0.5, 0)


# ──────────────────── 7. Stable Diffusion Outpainter ──────────────
class Outpainter:
    def __init__(self, device):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

    def __call__(self, img_bgr, mask_bin):
        pil_i = Image.fromarray(img_bgr)
        pil_m = Image.fromarray((mask_bin * 255).astype(np.uint8))
        res = self.pipe(
            prompt="extend background in identical style",
            image=pil_i, mask_image=pil_m, strength=0.75, guidance_scale=7.5
        ).images[0]
        return cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)


# ─────────────────── 8. Scene & Renderer Setup ────────────────────
def setup_scene(glb, light, size):
    t_scene = trimesh.load(glb, force="scene")
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[light["ambient"]] * 3 + [1.0])
    for geo in t_scene.geometry.values():
        scene.add(pyrender.Mesh.from_trimesh(geo), pose=np.eye(4))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3)
    cam_pose = np.eye(4); cam_pose[2, 3] = 2.5
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.), pose=cam_pose)
    renderer = pyrender.OffscreenRenderer(*size)
    return scene, renderer


# ───────────────────────── 9. Main Blend ──────────────────────────
def blend(env_path, fbx_path, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = cv2.imread(env_path);  h, w = env.shape[:2]

    light = estimate_lighting(env)
    outpaint = Outpainter(device)

    glb = convert_fbx_to_gltf(fbx_path)
    times, trans = extract_animation_transforms(glb)
    scene, renderer = setup_scene(glb, light, (w, h))

    writer = imageio.get_writer(out_path, fps=30, codec="libx264", pixelformat="yuv420p")
    prev = None

    for i in range(len(times)):
        # 1. Shift + outpaint environment
        dx = int(trans[i, 0] * 20)
        shifted = cv2.warpAffine(env, np.float32([[1, 0, dx], [0, 1, 0]]), (w, h), borderValue=(0, 0, 0))
        mask_env = np.all(shifted == 0, axis=-1).astype(np.uint8)
        bg = outpaint(shifted, mask_env)

        # 2. Render character RGBA
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        lit = apply_character_lighting(rgba, light)

        # 3. Shadow
        shadow = generate_contact_shadow(lit[..., 3], light["dir"], (h, w))
        bg_shadow = poisson_composite(bg, np.zeros_like(bg), shadow, thresh=0.05)


        # 4. Composite character
        comp = poisson_composite(bg_shadow, lit[..., :3], lit[..., 3] / 255.0)

        # 5. Temporal smooth
        if prev is not None:
            comp = temporal_smooth(prev, comp)
        prev = comp

        writer.append_data(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))

    writer.close()
    renderer.delete()
    print("✅ Saved:", out_path)


# ─────────────────────────── CLI ─────────────────────────────────
if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("--env", required=True, help="env image")
    ag.add_argument("--fbx", required=True, help="character FBX")
    ag.add_argument("--out", default="blend.mp4", help="output mp4")
    args = ag.parse_args()
    blend(args.env, args.fbx, args.out)

import os
import torch
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.transforms import Transform3d
from sklearn.neighbors import NearestNeighbors

def load_obj_correct_indices(mesh_path, device):
    verts, faces, aux = load_obj(mesh_path)
    # Correctly access the tensor representing the vertex indices in faces
    if faces.verts_idx.min() > 0:  # Access the indices tensor and check if adjustment is needed
        faces.verts_idx -= 1  # Adjust from 1-based to 0-based indices if necessary
    return verts.to(device), faces.verts_idx.to(device), aux

def load_mesh_with_uv_and_texture(mesh_path, texture_image_path, device):
    verts, faces, aux = load_obj_correct_indices(mesh_path, device)  # Updated loader with index adjustment
    image = Image.open(texture_image_path).convert("RGB")
    texture_map = torch.from_numpy(np.array(image)).to(device)
    texture_map = texture_map.permute(2, 0, 1) / 255.0

    if aux.verts_uvs is not None:
        verts_uvs = aux.verts_uvs.to(device)
        faces_uvs = faces.clone()  # Adjusted to correct indices
        textures = TexturesUV(maps=[texture_map], verts_uvs=[verts_uvs], faces_uvs=[faces_uvs])
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        return mesh
    else:
        print("UV data not present.")
        return None

def align_and_scale(source_verts, target_verts):
    # Compute the centroids
    src_centroid = source_verts.mean(0)
    tgt_centroid = target_verts.mean(0)

    # Ensure centroids are valid numbers
    if torch.isnan(src_centroid).any() or torch.isnan(tgt_centroid).any():
        raise ValueError("Computed NaN centroids.")

    # Compute scaling factors
    src_scale = (source_verts - src_centroid).abs().max()
    tgt_scale = (target_verts - tgt_centroid).abs().max()

    # Ensure scales are valid numbers
    if torch.isnan(src_scale) or torch.isnan(tgt_scale):
        raise ValueError("Computed NaN scales.")

    # Compute transformation for alignment and scaling
    transform = Transform3d().translate(-tgt_centroid).scale(src_scale / tgt_scale).translate(src_centroid)
    return transform.transform_points(target_verts)

def transfer_uvs_nearest_neighbor(source_mesh, target_verts):
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(source_mesh.verts_packed().cpu().numpy())
    distances, indices = nn.kneighbors(target_verts.cpu().numpy())
    source_uvs = source_mesh.textures.verts_uvs_packed().cpu().numpy()
    target_uvs = source_uvs[indices.flatten()]
    return torch.from_numpy(target_uvs).to(device)
def export_mesh_as_obj(mesh, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    uvs = mesh.textures.verts_uvs_packed()
    faces_uvs = mesh.textures.faces_uvs_packed()
    save_obj(file_path, verts, faces, verts_uvs=uvs, faces_uvs=faces_uvs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_mesh_path = "/mnt/c/Users/niyaz/Documents/SMPLitex/sample-data/smpl_uv_20200910/smpl_uv.obj" 
target_mesh_path = "/mnt/c/Users/niyaz/Downloads/male_download_aligned.obj"
texture_image_path = "/mnt/c/Users/niyaz/Downloads/upscale_inpainting.png"
output_path = 'new_target_mesh.obj'

source_mesh = load_mesh_with_uv_and_texture(source_mesh_path, texture_image_path, device)
target_verts, target_faces, _ = load_obj(target_mesh_path)
target_verts = target_verts.to(device)
target_faces = target_faces.verts_idx.to(device)

if source_mesh:
    aligned_target_verts = align_and_scale(source_mesh.verts_packed(), target_verts)
    target_uvs = transfer_uvs_nearest_neighbor(source_mesh, aligned_target_verts)
    target_mesh = Meshes(
        verts=[aligned_target_verts],
        faces=[target_faces],
        textures=TexturesUV(maps=source_mesh.textures.maps, verts_uvs=[target_uvs], faces_uvs=[target_faces])
    )
    export_mesh_as_obj(target_mesh, output_path)
    print("Target mesh exported successfully.")
else:
    print("Failed to load or process source mesh.")

# Paths
 # Update to your source mesh path  # Replace with your actual path
  # Replace with your texture image path


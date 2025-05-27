import os
import torch
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

def load_mesh_with_uv_and_texture(mesh_path, texture_image_path, device):
    # Load the texture image
    image = Image.open(texture_image_path).convert("RGB")  # Ensure it's RGB
    texture_map = torch.from_numpy(np.array(image)).to(device)  # Convert image to tensor
    texture_map = texture_map.permute(2, 0, 1) / 255.0  # Rearrange and normalize

    # Load the mesh
    try:
        verts, faces, aux = load_obj(mesh_path)
        print("Mesh loaded successfully.")
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        return None

    verts = verts.to(device)
    faces = faces.verts_idx.to(device)

    # Check for UV data presence
    if aux.verts_uvs is not None:
        verts_uvs = aux.verts_uvs.to(device)
        faces_uvs = faces.clone()  # This line assumes each face vertex index matches UV index
        textures = TexturesUV(maps=[texture_map], verts_uvs=[verts_uvs], faces_uvs=[faces_uvs])
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        print("Mesh and texture loaded successfully.")
        return mesh
    else:
        print("UV data not present in the auxiliary data.")
        return None

def reproject_uvs(source_mesh, target_mesh, device):
    # Get vertices from both meshes
    source_verts = source_mesh.verts_packed().to(device)
    target_verts = target_mesh.verts_packed().to(device)

    # Calculate the closest source vertex for each target vertex
    distances = torch.cdist(target_verts, source_verts)
    indices = torch.argmin(distances, dim=1)

    # Debugging: Print some indices and their corresponding UVs
    print("Sample indices:", indices[:10])  # Print first 10 indices

    # Use indices to gather UV coordinates from source mesh
    source_uvs = source_mesh.textures.verts_uvs_padded().squeeze(0)
    target_uvs = source_uvs[indices]

    # Debugging: Check some of the reassigned UVs
    print("Sample UVs from target:", target_uvs[:10])  # Print first 10 UVs

    # Assuming target mesh uses same face indices for UVs as for vertices
    target_faces_uvs = target_mesh.faces_packed()

    # Create new textures for the target mesh using the reprojected UVs
    new_textures = TexturesUV(maps=source_mesh.textures.maps_padded(), verts_uvs=[target_uvs], faces_uvs=[target_faces_uvs])

    # Return new mesh with the same vertices and faces, but new textures
    new_mesh = Meshes(verts=[target_verts], faces=[target_faces_uvs], textures=new_textures)
    return new_mesh


from pytorch3d.io import save_obj

def save_mesh_as_obj(mesh, obj_path, texture_image_path):
    # Extract mesh components
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    verts_uvs = mesh.textures.verts_uvs_padded().squeeze(0)
    faces_uvs = mesh.textures.faces_uvs_padded().squeeze(0)

    # Save the OBJ file
    save_obj(obj_path, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs)

    # Define the file path for the MTL file (same name as OBJ but with .mtl extension)
    mtl_path = obj_path.replace(".obj", ".mtl")
    material_name = "Material"

    # Manually write the MTL file to include texture information
    with open(mtl_path, 'w') as mtl_file:
        mtl_file.write(f"newmtl {material_name}\n")
        mtl_file.write("Ns 96.078431\n")
        mtl_file.write("Ka 1.000000 1.000000 1.000000\n")
        mtl_file.write("Kd 0.640000 0.640000 0.640000\n")
        mtl_file.write("Ks 0.500000 0.500000 0.500000\n")
        mtl_file.write("Ke 0.000000 0.000000 0.000000\n")
        mtl_file.write("Ni 1.450000\n")
        mtl_file.write("d 1.000000\n")
        mtl_file.write("illum 2\n")
        mtl_file.write(f"map_Kd {texture_image_path}\n")  # Texture image path relative to MTL file

    # Add MTL file reference to the OBJ file
    with open(obj_path, 'r+') as obj_file:
        content = obj_file.read()
        obj_file.seek(0, 0)
        obj_file.write(f'mtllib {mtl_path}\n' + content)

    print(f'Mesh saved to {obj_path}')
    print(f'MTL file saved to {mtl_path}')

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
source_mesh_path = "/mnt/c/Users/niyaz/Documents/SMPLitex/sample-data/smpl_uv_20200910/smpl_uv.obj" 
target_mesh_path = "/mnt/c/Users/niyaz/Downloads/male_download_aligned.obj"
texture_image_path = "/mnt/c/Users/niyaz/Downloads/upscale_inpainting.png"

# Load source mesh with texture
source_mesh = load_mesh_with_uv_and_texture(source_mesh_path, texture_image_path, device)

# Load target mesh (without texture processing, since we will reproject)
try:
    target_verts, target_faces, _ = load_obj(target_mesh_path, device=device)
    target_mesh = Meshes(verts=[target_verts.to(device)], faces=[target_faces.verts_idx.to(device)])
    print("Target mesh loaded successfully.")
except Exception as e:
    print(f"Failed to load target mesh: {e}")
    target_mesh = None

# Reproject UVs and apply texture from source to target mesh
if source_mesh and target_mesh:
    reprojected_mesh = reproject_uvs(source_mesh, target_mesh, device)
    print("UV reprojection complete. Texture has been transferred.")
    
    # Save the reprojected mesh to an OBJ file
    output_file_path = "output_mesh.obj"
    save_mesh_as_obj(reprojected_mesh, output_file_path, r'C:\Users\niyaz\Downloads\upscale_inpainting.png')
else:
    print("Failed to load meshes properly.")

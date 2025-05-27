import numpy as np
import pandas as pd
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Function to load UV coordinates from a CSV file
def load_uv_coordinates(csv_path):
    uv_coords_df = pd.read_csv(csv_path)
    uv_coords = {}
    for part_id, group in uv_coords_df.groupby('part_id'):
        uv_coords[int(part_id)] = group[['u', 'v']].to_numpy()
    return uv_coords

# Load UV coordinates from multiple DensePose outputs
uv_coords_head_1 = load_uv_coordinates('/mnt/data/uv_coordinates_head_1.csv')
uv_coords_head_2 = load_uv_coordinates('/mnt/data/uv_coordinates_head_2.csv')
uv_coords_body_1 = load_uv_coordinates('/mnt/data/uv_coordinates_body_1.csv')
uv_coords_body_2 = load_uv_coordinates('/mnt/data/uv_coordinates_body_2.csv')

# Load texture images for head and body
head_textures = [Image.open(f'/mnt/data/head_texture_{i}.png') for i in range(1, 3)]
body_textures = [Image.open(f'/mnt/data/body_texture_{i}.png') for i in range(1, 3)]

# Define camera parameters (example values, replace with actual parameters)
focal_length = 35  # in mm
sensor_width = 36  # in mm
sensor_height = 24  # in mm
image_width = 1920  # in pixels
image_height = 1080  # in pixels

fx = (focal_length / sensor_width) * image_width
fy = (focal_length / sensor_height) * image_height
cx = image_width / 2
cy = image_height / 2

camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Example extrinsic parameters (rotation and translation vectors for each view)
view_angles = [
    {'rvec': np.array([0, 0, 0]), 'tvec': np.array([0, 0, 0])},          # Front view
    {'rvec': np.array([0, np.pi/4, 0]), 'tvec': np.array([0, 0, -1000])} # Side view
]


# Function to project vertices
def project_vertices(vertices, camera_intrinsics, rvec, tvec):
    rotation_matrix = R.from_rotvec(rvec).as_matrix()
    transformed_vertices = vertices.dot(rotation_matrix.T) + tvec
    projected_vertices = transformed_vertices.dot(camera_intrinsics.T)
    projected_vertices /= projected_vertices[:, -1:]
    return projected_vertices[:, :2]

# Load the 3D mesh
head_mesh = trimesh.load('/mnt/data/head_mesh.obj')
body_mesh = trimesh.load('/mnt/data/body_mesh.obj')

# Project vertices for each view
projected_vertices_head = [project_vertices(head_mesh.vertices, camera_intrinsics, v['rvec'], v['tvec']) for v in view_angles]
projected_vertices_body = [project_vertices(body_mesh.vertices, camera_intrinsics, v['rvec'], v['tvec']) for v in view_angles]

# Function to map texture to UV coordinates from multiple views
def map_texture_to_uv_multiple_views(mesh, uv_coords_list, texture_images):
    atlas_size = (2048, 2048)
    texture_atlas = Image.new('RGB', atlas_size)
    
    for uv_coords, texture_image in zip(uv_coords_list, texture_images):
        texture_image = np.array(texture_image)
        atlas = np.array(texture_atlas)
        for part_id, uv in uv_coords.items():
            for i, vertex_index in enumerate(mesh.faces.flatten()):
                if i < len(uv):
                    x, y = int(uv[i][0] * atlas_size[0]), int(uv[i][1] * atlas_size[1])
                    atlas[y, x] = texture_image[y % texture_image.shape[0], x % texture_image.shape[1]]
        texture_atlas = Image.fromarray(atlas)
    
    return texture_atlas

# Map textures to head and body meshes using multiple views
head_uv_coords_list = [uv_coords_head_1, uv_coords_head_2]
head_texture_images = head_textures
head_texture_atlas = map_texture_to_uv_multiple_views(head_mesh, head_uv_coords_list, head_texture_images)

body_uv_coords_list = [uv_coords_body_1, uv_coords_body_2]
body_texture_images = body_textures
body_texture_atlas = map_texture_to_uv_multiple_views(body_mesh, body_uv_coords_list, body_texture_images)

# Apply the combined texture atlas to the meshes
head_mesh.visual.material.image = head_texture_atlas
body_mesh.visual.material.image = body_texture_atlas

# Combine head and body meshes
combined_mesh = trimesh.util.concatenate([head_mesh, body_mesh])

# Save the combined mesh
combined_mesh.export('/mnt/data/combined_textured_mesh.obj')


import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image

# Load the 3D mesh
mesh = trimesh.load('/mnt/c/Users/niyaz/Documents/custom_mesh.obj')

# Function to load UV coordinates from CSV files
def load_uv_coordinates(csv_path):
    uv_coords_df = pd.read_csv(csv_path)
    uv_coords = {}
    for part_id, group in uv_coords_df.groupby('part_id'):
        uv_coords[int(part_id)] = group[['u', 'v']].to_numpy()
    return uv_coords

# Load the UV coordinates from multiple DensePose outputs
uv_coords_1 = load_uv_coordinates('/mnt/data/uv_coordinates_1.csv')
uv_coords_2 = load_uv_coordinates('/mnt/data/uv_coordinates_2.csv')
# Add more as needed...

# Combine UV coordinates from multiple angles
def combine_uv_coordinates(*uv_coords_list):
    combined_uv_coords = {}
    for uv_coords in uv_coords_list:
        for part_id, coords in uv_coords.items():
            if part_id not in combined_uv_coords:
                combined_uv_coords[part_id] = coords
            else:
                combined_uv_coords[part_id] = np.vstack((combined_uv_coords[part_id], coords))
    return combined_uv_coords

combined_uv_coords = combine_uv_coordinates(uv_coords_1, uv_coords_2)
# Add more as needed...

# Function to project 3D vertices to 2D
def project_vertices(vertices, camera_intrinsics, rvec, tvec):
    rotation_matrix = R.from_rotvec(rvec).as_matrix()
    transformed_vertices = vertices.dot(rotation_matrix.T) + tvec
    projected_vertices = transformed_vertices.dot(camera_intrinsics.T)
    projected_vertices /= projected_vertices[:, -1:]
    return projected_vertices[:, :2]

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

# Example rotation and translation vectors for different views
rvec_1 = np.array([0, 0, 0])
tvec_1 = np.array([0, 0, 0])

rvec_2 = np.array([0, np.pi/4, 0])
tvec_2 = np.array([0, 0, -1000])

# Project 3D vertices to 2D image plane for different views
projected_vertices_1 = project_vertices(mesh.vertices, camera_intrinsics, rvec_1, tvec_1)
projected_vertices_2 = project_vertices(mesh.vertices, camera_intrinsics, rvec_2, tvec_2)

# Segment the mesh for each view
def segment_mesh(mesh, projected_vertices, uv_coords):
    segments = {part_id: [] for part_id in uv_coords.keys()}
    for i, vertex in enumerate(projected_vertices):
        for part_id, uv_coord in uv_coords.items():
            if (vertex == uv_coord).all(axis=1).any():
                segments[part_id].append(i)
    return segments

segments_1 = segment_mesh(mesh, projected_vertices_1, combined_uv_coords)
segments_2 = segment_mesh(mesh, projected_vertices_2, combined_uv_coords)

# Combine segments from multiple views
def combine_segments(*segments_list):
    combined_segments = {}
    for segments in segments_list:
        for part_id, indices in segments.items():
            if part_id not in combined_segments:
                combined_segments[part_id] = indices
            else:
                combined_segments[part_id] = list(set(combined_segments[part_id] + indices))
    return combined_segments

combined_segments = combine_segments(segments_1, segments_2)
# Add more as needed...

# Assign UV coordinates to the mesh
def assign_uv_coordinates(mesh, segments, uv_coords):
    uv_map = np.zeros((len(mesh.vertices), 2))
    for part_id, vertex_indices in segments.items():
        uv_part_coords = uv_coords[part_id]
        for i, vertex_idx in enumerate(vertex_indices):
            if i < len(uv_part_coords):
                uv_map[vertex_idx, :] = uv_part_coords[i]
    return uv_map

uv_map = assign_uv_coordinates(mesh, combined_segments, combined_uv_coords)
mesh.visual.uv = uv_map

# Generate and apply texture
class UVTextureGenerator:
    @classmethod
    def concat_atlas_tex(cls, given_tex):
        tex = None
        for i in range(0, 4):
            tex_tmp = given_tex[6 * i]
            for i in range(1 + 6 * i, 6 + 6 * i):
                tex_tmp = np.concatenate((tex_tmp, given_tex[i]), axis=1)
            if tex is None:
                tex = tex_tmp
            else:
                tex = np.concatenate((tex, tex_tmp), axis=0)
        return tex

    @classmethod
    def create_smpl_from_images(cls, im, iuv, img_size=200):
        i_id, u_id, v_id = 2, 1, 0
        parts_list = [1, 2, 3, 4, 5, 

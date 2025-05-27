import open3d as o3d
import numpy as np
import trimesh
import json
import os
import matplotlib.pyplot as plt

def load_keypoints_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    keypoints = data["keypoints"]
    # Extract neck point approximated as the midpoint between shoulders
    left_shoulder = keypoints[11]  # Left shoulder keypoint (index 11 in COCO format)
    right_shoulder = keypoints[12]  # Right shoulder keypoint (index 12 in COCO format)
    neck_point = ((left_shoulder[0] + right_shoulder[0]) // 2,
                  (left_shoulder[1] + right_shoulder[1]) // 2)
    # Extract torso point as the midpoint between hips
    left_hip = keypoints[23]  # Left hip keypoint (index 23 in COCO format)
    right_hip = keypoints[24]  # Right hip keypoint (index 24 in COCO format)
    torso_point = ((left_hip[0] + right_hip[0]) // 2,
                   (left_hip[1] + right_hip[1]) // 2)
    return neck_point, torso_point

def load_glb_as_trimesh(file_path):
    # Load the .glb file using open3d
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"Unable to load mesh from file: {file_path}")
    # Convert to trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def scale_mesh(mesh, scale_factor):
    # Scale the mesh
    mesh.apply_scale(scale_factor)
    return mesh

def visualize_mesh(mesh, title="Mesh"):
    mesh.export("temp.obj")
    mesh_o3d = o3d.io.read_triangle_mesh("temp.obj")
    o3d.visualization.draw_geometries([mesh_o3d], window_name=title)

def align_and_combine_meshes(head_mesh, body_mesh, neck_point, torso_point):
    # Scale down the head mesh
    head_mesh = scale_mesh(head_mesh, 1/3.0)
    
    # Use the keypoints to align the head and body meshes
    neck_point = np.array([neck_point[0], neck_point[1], 0])
    torso_point = np.array([torso_point[0], torso_point[1], 0])
    
    # Calculate the translation needed to align the keypoints
    translation = torso_point - neck_point
    
    # Debug: Print translation vector
    print(f"Translation vector: {translation}")
    
    # Create a 4x4 transformation matrix for translation
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = translation
    
    # Debug: Print transformation matrix
    print(f"Transformation matrix:\n{transformation_matrix}")
    
    # Apply translation to head mesh
    head_mesh.apply_transform(transformation_matrix)
    
    # Visualize meshes before combining
    visualize_mesh(head_mesh, title="Head Mesh After Transformation")
    visualize_mesh(body_mesh, title="Body Mesh")
    
    # Combine the meshes
    combined_mesh = trimesh.util.concatenate([head_mesh, body_mesh])
    
    # Visualize combined mesh
    visualize_mesh(combined_mesh, title="Combined Mesh")
    
    return combined_mesh

# Paths
image_path = "/Users/niyaz/Downloads/alexander.png"
output_json_file = "/Users/niyaz/Downloads/alexander_mediapipe/keypoints.json"
head_mesh_path = "/tmp/gradio/generated_1722618674.glb"
body_mesh_path = "/tmp/gradio/generated_1722619194.glb"
output_combined_mesh_path = "/Users/niyaz/Downloads/combined_test.glb"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_combined_mesh_path), exist_ok=True)

# Load keypoints from JSON file
neck_point, torso_point = load_keypoints_from_json(output_json_file)

# Load head and body meshes
head_mesh = load_glb_as_trimesh(head_mesh_path)
body_mesh = load_glb_as_trimesh(body_mesh_path)

# Align and combine meshes
combined_mesh = align_and_combine_meshes(head_mesh, body_mesh, neck_point, torso_point)

# Save the combined mesh
combined_mesh.export(output_combined_mesh_path)

print(f"Combined mesh saved to {output_combined_mesh_path}")

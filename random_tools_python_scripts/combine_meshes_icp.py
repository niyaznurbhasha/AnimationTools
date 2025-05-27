import open3d as o3d
import numpy as np
import mediapipe as mp
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def extract_keypoints(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    logging.info("Reading image for MediaPipe processing")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No keypoints detected")

    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])))

    logging.info("Extracting keypoints")
    # Neck point approximated as the midpoint between shoulders
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    neck_point = ((left_shoulder[0] + right_shoulder[0]) // 2,
                  (left_shoulder[1] + right_shoulder[1]) // 2)

    # Extract torso point as the midpoint between hips
    left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
    torso_point = ((left_hip[0] + right_hip[0]) // 2,
                   (left_hip[1] + right_hip[1]) // 2)
    
    return neck_point, torso_point

def load_glb_as_trimesh(file_path):
    logging.info(f"Loading mesh from {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"Unable to load mesh from file: {file_path}")
    return mesh

def icp_registration(source, target, init_transformation):
    threshold = 0.02  # Distance threshold for ICP
    logging.info("Starting ICP registration")

    # Convert meshes to point clouds
    source_pc = source.sample_points_uniformly(number_of_points=100000)
    target_pc = target.sample_points_uniformly(number_of_points=100000)

    # Estimate normals for the point clouds
    source_pc.estimate_normals()
    target_pc.estimate_normals()

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    logging.info("ICP registration completed")
    return reg_p2p.transformation

def scale_mesh(mesh, scale_factor):
    logging.info(f"Scaling mesh by factor {scale_factor}")
    mesh.scale(scale_factor, center=mesh.get_center())
    return mesh

def combine_meshes(head_mesh, body_mesh, transformation):
    # Transform the head mesh using the calculated transformation
    logging.info("Applying transformation to the head mesh")
    head_mesh.transform(transformation)

    # Combine the head and body meshes
    combined_mesh = head_mesh + body_mesh

    return combined_mesh

# Paths
image_path = "/Users/niyaz/Downloads/warrior_statue.png"
head_mesh_path = "/tmp/gradio/generated_1722618674.glb"
body_mesh_path = "/tmp/gradio/generated_1722619194.glb"
output_combined_mesh_path = "/Users/niyaz/Downloads/combined_test.glb"

try:
    # Extract keypoints using MediaPipe
    logging.info("Extracting keypoints from image")
    neck_point, torso_point = extract_keypoints(image_path)

    # Load head and body meshes
    head_mesh = load_glb_as_trimesh(head_mesh_path)
    body_mesh = load_glb_as_trimesh(body_mesh_path)

    # Scale down the head mesh if needed
    head_mesh = scale_mesh(head_mesh, 1/3.0)

    # Compute initial transformation based on keypoints
    neck_point = np.array([neck_point[0], neck_point[1], 0])
    torso_point = np.array([torso_point[0], torso_point[1], 0])
    translation = torso_point - neck_point
    init_transformation = np.eye(4)
    init_transformation[:3, 3] = translation

    # Convert to Open3D meshes for ICP
    head_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(head_mesh.vertices), o3d.utility.Vector3iVector(head_mesh.triangles))
    body_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(body_mesh.vertices), o3d.utility.Vector3iVector(body_mesh.triangles))

    # Perform ICP
    transformation = icp_registration(head_o3d, body_o3d, init_transformation)

    # Combine the meshes
    combined_mesh = combine_meshes(head_o3d, body_o3d, transformation)

    # Save the combined mesh
    logging.info(f"Saving combined mesh to {output_combined_mesh_path}")
    o3d.io.write_triangle_mesh(output_combined_mesh_path, combined_mesh)

    logging.info(f"Combined mesh saved to {output_combined_mesh_path}")

    # Visualize the combined mesh for verification
    logging.info("Visualizing the combined mesh")
    o3d.visualization.draw_geometries([combined_mesh])

except Exception as e:
    logging.error(f"An error occurred: {e}")

from model.treemeshgpt_inference import TreeMeshGPT
import os
import numpy as np
import open3d as o3d
import torch
from accelerate import Accelerator
from pathlib import Path
from fns import center_vertices, normalize_vertices_scale, str2bool
import trimesh
import pyvista as pv
import argparse
import time

# Define argument parser
def get_args():
    parser = argparse.ArgumentParser(description="")
    
    # Add toggle for 7-bit or 9-bit configuration
    parser.add_argument("--version", type=str, choices=["7bit", "9bit"], default="7bit", help="Choose between 7-bit or 9-bit configuration")
    
    # Default values for 7-bit
    config_defaults = {
        "ckpt_path": "./checkpoints/treemeshgpt_7bit.pt",
        "mesh_path": "demo/luma_cat.glb",
        "decimation": True,
        "decimation_target_nfaces": 6000,
        "decimation_boundary_deletion": True,
        "sampling": "uniform",
    }
    
    # Override defaults for 9-bit
    if parser.parse_known_args()[0].version == "9bit":
        config_defaults.update({
            "ckpt_path": "./checkpoints/treemeshgpt_9bit.pt",
            "decimation_boundary_deletion": False,
            "sampling": "fps",
        })
    
    parser.add_argument("--ckpt_path", type=str, default=config_defaults["ckpt_path"], help="Path to the model checkpoint")
    parser.add_argument("--mesh_path", type=str, default=config_defaults["mesh_path"], help="Path to the input mesh file")
    
    parser.add_argument("--decimation", type=str2bool, nargs='?', const=True, default=config_defaults["decimation"], help="Enable or disable mesh decimation")
    parser.add_argument("--decimation_target_nfaces", type=int, default=config_defaults["decimation_target_nfaces"], help="Target number of faces after decimation")
    parser.add_argument("--decimation_boundary_deletion", type=str2bool, nargs='?', const=True, default=config_defaults["decimation_boundary_deletion"], help="Allow boundary vertex deletion (higher faces but lower success rate if False)")
    parser.add_argument("--sampling", type=str, choices=["uniform", "fps"], default=config_defaults["sampling"], help="Sampling method: 'uniform' or 'fps'")
    
    args = parser.parse_args()
    mesh_filename = os.path.splitext(os.path.basename(args.mesh_path))[0]
    args.out_folder = os.path.join("demo", mesh_filename + "_{}".format(args.version))
    
    return args

# Parse arguments
args = get_args()

# Assign parsed values to variables
VERSION = args.version
CKPT_PATH = args.ckpt_path
MESH_PATH = args.mesh_path

DECIMATION = args.decimation
DECIMATION_TARGET_NFACES = args.decimation_target_nfaces
DECIMATION_BOUNDARY_DELETION = args.decimation_boundary_deletion

SAMPLING = args.sampling
OUT_FOLDER = args.out_folder

print("TreeMeshGPT version:", VERSION)
print("Checkpoint Path:", CKPT_PATH)
print("Mesh Path:", MESH_PATH)
print("Decimation Enabled:", DECIMATION)
print("Decimation Target Faces:", DECIMATION_TARGET_NFACES)
print("Boundary Vertex Deletion:", DECIMATION_BOUNDARY_DELETION)
print("Sampling Method:", SAMPLING)
print("Output Folder:", OUT_FOLDER)

# Set up model
transformer = TreeMeshGPT(quant_bit = 7 if VERSION == "7bit" else 9, max_seq_len=30000)
transformer.load(CKPT_PATH)
accelerator = Accelerator(mixed_precision="fp16")
transformer = accelerator.prepare(transformer)

# Load and normalize mesh
mesh = o3d.io.read_triangle_mesh(MESH_PATH)
vertices = np.asarray(mesh.vertices)
vertices = center_vertices(vertices)
vertices = normalize_vertices_scale(vertices)
vertices = np.clip(vertices, a_min=-0.5, a_max = 0.5)
triangles = np.asarray(mesh.triangles)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)

# Mesh decimation
if DECIMATION:
    n_triangles = min(DECIMATION_TARGET_NFACES, len(triangles))
    faces_pyvista = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(np.int64).flatten()
    mesh = pv.PolyData(vertices, faces_pyvista)
    decimated_mesh = mesh.decimate_pro(1-n_triangles/len(triangles), boundary_vertex_deletion=DECIMATION_BOUNDARY_DELETION)
    decimated_vertices = np.array(decimated_mesh.points)
    decimated_faces = np.array(decimated_mesh.faces).reshape(-1, 4)[:, 1:]  # Remove leading '3' per triangle
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(decimated_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(decimated_faces)
    print("Mesh is decimated to {} faces".format(len(decimated_faces)))
else:
    print("Sampling from original mesh with {} faces".format(len(triangles)))

# Point cloud sampling
if SAMPLING == "uniform":
    pc = mesh.sample_points_uniformly(number_of_points=8192)
elif SAMPLING == "fps":
    pc = mesh.sample_points_uniformly(number_of_points=8192*10)
    pc_array = np.asarray(pc.points)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_array)    
    pc = pc.farthest_point_down_sample(8192//2)        
pc_array = np.asarray(pc.points) 
pc = torch.tensor(pc_array).unsqueeze(0).float().cuda()

# Generation
t = time.time()
with accelerator.autocast(), torch.no_grad():
    triangles = transformer.generate(pc,n = 0.25)
elapsed = time.time() - t
    
vertices = triangles.view(-1, 3).cpu().numpy()
n = vertices.shape[0]
faces = torch.arange(1, n + 1).view(-1, 3).numpy()

if min(min(faces.tolist())) == 1:
    faces = (np.array(faces) - 1)
    
# Remove collapsed triangles and duplicates
p0 = vertices[faces[:, 0]]
p1 = vertices[faces[:, 1]]
p2 = vertices[faces[:, 2]]
collapsed_mask = np.all(p0 == p1, axis=1) | np.all(p0 == p2, axis=1) | np.all(p1 == p2, axis=1)
faces = faces[~collapsed_mask]
faces = faces.tolist()
scene_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, force="mesh",
                        merge_primitives=True)
scene_mesh.merge_vertices()
scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
scene_mesh.update_faces(scene_mesh.unique_faces())
scene_mesh.remove_unreferenced_vertices()
scene_mesh.fix_normals()

# Save outputs
points = pc.squeeze(0).cpu().numpy()
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
target_out = os.path.join("generation", OUT_FOLDER)
target_out = Path(target_out)
target_out.mkdir(parents=True, exist_ok=True)
pc_out = os.path.join(target_out, "point_cloud.ply")
mesh_out = os.path.join(target_out, "mesh_out.obj")
o3d.io.write_point_cloud(pc_out, point_cloud)
scene_mesh.export(mesh_out)

print(f" | Generation is finished. Time elapsed: {elapsed:.2f} seconds.")

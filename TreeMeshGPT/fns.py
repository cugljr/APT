import numpy as np
import argparse
import random
from scipy.spatial.transform import Rotation
import open3d as o3d
import trimesh

# Custom function to handle boolean arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    # vert_center = np.mean(vertices, axis=0)
    return vertices - vert_center

def normalize_vertices_scale(vertices):
    """Scale the vertices so that the long diagonal of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents**2))
    # scale = np.max(np.abs(vertices))
    return vertices / (scale + 1e-6)

def dequantize_verts_tensor(verts, n_bits=7):
    """Convert quantized vertices (torch tensor) to floats."""
    scale = 0.5  # same scaling as in the original function
    min_range = -scale
    max_range = scale
    range_quantize = 2 ** n_bits - 1

    verts = verts.float()
    verts = verts * (max_range - min_range) / range_quantize + min_range

    return verts

def quantize_verts(verts, n_bits=7):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    scale = 0.5 #1.0
    min_range = -scale
    max_range = scale
    range_quantize = 2**n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    verts_quantize = np.round(verts_quantize)
    return verts_quantize.astype("int32")


def dequantize_verts(verts, n_bits=7, add_noise=False):
    """Convert quantized vertices to floats."""
    scale = 0.5 #1.0
    min_range = -scale
    max_range = scale
    range_quantize = 2**n_bits - 1
    verts = verts.astype("float32")
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
    return verts

def augment_mesh(vertices, scale_min=0.75, scale_max=0.95, rotation=180.):    
    # vertices [nv, 3]
    for i in range(3):
        # Generate a random scale factor
        scale = random.uniform(scale_min, scale_max)

        # independently applied scaling across each axis of vertices
        vertices[:, i] *= scale
    
    if rotation != 0.:        
        rotate_upright = random.random() < 0.3
        
        if rotate_upright:
            rotation_options = [0.5 * np.pi, -0.5 * np.pi]
            
            # Randomly choose rotation angles for x and y axes
            rot_x = random.choice(rotation_options)
            rot_y = random.choice(rotation_options)
            case = random.choice([1, 2])
            
            # Apply the rotation based on the chosen case
            if case == 1:
                rotation_obj = Rotation.from_rotvec([rot_x, 0, 0])
                vertices = rotation_obj.apply(vertices)
            elif case == 2:
                rotation_obj = Rotation.from_rotvec([0, rot_y, 0])
                vertices = rotation_obj.apply(vertices)
        
        rot_z = random.uniform(-1, 1) * np.pi * 180
        angles = np.array([0, 0, rot_z])
        rotation_obj = Rotation.from_rotvec(angles)
        vertices = rotation_obj.apply(vertices)
    return vertices

def process_mesh(mesh_fn, augment = False):
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    vertices = center_vertices(vertices)
    
    if augment:
        vertices = augment_mesh(vertices)
        
    vertices = normalize_vertices_scale(vertices)
    if np.max(vertices) > 0.5 or np.abs(np.min(vertices)) > 0.5:
        vertices = vertices * 0.97

    vertices = np.clip(vertices, -0.5, 0.5)
    
    return vertices, triangles

def sample_point_cloud(vertices, triangles, sampling='uniform'):
    mesh_pc = o3d.geometry.TriangleMesh()
    mesh_pc.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_pc.triangles = o3d.utility.Vector3iVector(triangles)
    
    if sampling == 'uniform':
        pc = mesh_pc.sample_points_uniformly(number_of_points=8192)
        
        # add a bit noise
        pc_array = np.asarray(pc.points)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0, 0.002)
            noise = np.random.normal(loc=0.0, scale=scale, size=pc_array.shape)
            pc_array = pc_array + noise
    elif sampling == 'fps':
        pc = mesh_pc.sample_points_uniformly(number_of_points=8192*2)
        
        # add a bit noise
        pc_array = np.asarray(pc.points)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0, 0.002)
            noise = np.random.normal(loc=0.0, scale=scale, size=pc_array.shape)
            pc_array = pc_array + noise
        
        # Create an Open3D PointCloud object
        pc = o3d.geometry.PointCloud()

        # Convert numpy array to Open3D format and assign to point cloud
        pc.points = o3d.utility.Vector3dVector(pc_array)    
        pc = pc.farthest_point_down_sample(8192//2)
        pc_array = np.asarray(pc.points)
    
    return pc_array

def quantize_remove_duplicates(vertices, triangles, quant_bit = 7):
    vertices = quantize_verts(vertices, n_bits = quant_bit)
    vertices = dequantize_verts(vertices, n_bits= quant_bit)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # remove degenerate faces
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    collapsed_mask = np.all(p0 == p1, axis=1) | np.all(p0 == p2, axis=1) | np.all(p1 == p2, axis=1)
    faces = faces[~collapsed_mask]  # Keep only non-collapsed triangles
    faces = faces.tolist()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    return vertices, triangles

def prepare_halfedge_mesh(vertices, triangles):
    # sort vertices and faces
    sort_index = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
    vertices = vertices[sort_index]
    index_mapping = np.zeros(sort_index.shape[0], dtype=int)
    index_mapping[sort_index] = np.arange(sort_index.shape[0])
    triangles = index_mapping[triangles]

    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    mesh2.triangles = o3d.utility.Vector3iVector(triangles)
    hf_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh2)

    vertices = np.asarray(hf_mesh.vertices)
    triangles = np.asarray(hf_mesh.triangles)
    sort_index = np.lexsort((triangles[:, 0], triangles[:, 1], triangles[:, 2]))
    sorted_triangles = triangles[sort_index]
    sorted_triangles = sorted_triangles[:, [2,0,1]]
    
    mesh3 = o3d.geometry.TriangleMesh()
    mesh3.vertices = o3d.utility.Vector3dVector(vertices)
    mesh3.triangles = o3d.utility.Vector3iVector(sorted_triangles)
    hf_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh3)

    return hf_mesh, vertices, len(triangles)

def create_io_sequence(hf_mesh, stop_label = -1):
    hf_list = hf_mesh.half_edges
    half_edge_dict = {}
    
    for i, half_edge in enumerate(hf_list):
        key = tuple(half_edge.vertex_indices)
        half_edge_dict[key] = {
            'half_edge': half_edge,
            'added': False,
        }
    
    idx_dict = {i: i for i in range(len(hf_list))}
    stack = []

    segments_edges = []
    segments_gt = []

    def mark_added(idx):
        vertex_indices = tuple(hf_list[idx].vertex_indices)
        half_edge_dict[vertex_indices]['added'] = True
        
    while len(idx_dict) != 0:
        edges = []
        gt = []
        visited = set()  # Set to track visited half-edges
        
        cur_idx = next(iter(idx_dict.items()))[0]
        node = hf_list[cur_idx]
        start = len(edges)
        edges.append(node.vertex_indices.tolist())
                
        # Initialization, always starts with at least 1 triangular face
        right_idx = node.next
        left_idx = hf_list[node.next].next
        
        # Add right half-edge vertex to gt
        gt.append(hf_list[right_idx].vertex_indices.tolist()[1])

        # Add left and right half-edges to the stack
        stack.append(cur_idx)
        mark_added(cur_idx) 
        stack.append(left_idx)
        mark_added(left_idx)
        stack.append(right_idx)
        mark_added(right_idx)
        
        # Loop for traversal
        while stack:
            stack_idx = stack.pop()

            node_stack = hf_list[stack_idx]
            twin_idx = node_stack.twin

            # Append the current edge to the edges list (even if visited)
            edges.append(node_stack.vertex_indices.tolist()[::-1])

            # If this edge is already part of the loop, append -1 to GT and skip processing
            if edges[-1] == edges[start]:
                gt.append(stop_label)
                visited.add(stack_idx)
                continue
            
            # If the edge has already been visited, add -1 to GT but continue to add it
            if stack_idx in visited:
                gt.append(stop_label)
                continue
            
            # Mark the half-edge as visited
            visited.add(stack_idx)

            # If the twin exists, process it
            if twin_idx != -1:
                cur_idx = twin_idx
                mark_added(cur_idx)
                node = hf_list[cur_idx]

                # Mark the twin as visited and process next edges
                visited.add(cur_idx)
                right_idx = node.next
                left_idx = hf_list[node.next].next

                # Add right half-edge vertex to GT
                gt.append(hf_list[right_idx].vertex_indices.tolist()[1])

                # Check if left and right edges are already added
                right_added = half_edge_dict[tuple(hf_list[right_idx].vertex_indices)]['added']
                left_added = half_edge_dict[tuple(hf_list[left_idx].vertex_indices)]['added']

                # If both are added, mark GT as -1
                if left_added and right_added:
                    gt[-1] = stop_label
                else:
                    # Add unvisited left and right half-edges to the stack
                    if not left_added:
                        stack.append(left_idx)
                        mark_added(left_idx)
                    if not right_added:
                        stack.append(right_idx)
                        mark_added(right_idx)
            else:
                # Append -1 to GT if there's no twin
                gt.append(stop_label)
            
        for idx in visited:
            del idx_dict[idx]
        
        segments_edges.append(edges)
        segments_gt.append(gt)
    
    return segments_edges, segments_gt
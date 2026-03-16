from enum import Enum
import glob
import os
import pdb
import random
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
import torchvision.transforms as T
from pytorch3d.io import load_obj
import pytorch_lightning as pl
from PIL import Image
import numpy as np

import polygen.utils.data_utils as data_utils


class ShapenetDataset(Dataset):
    def __init__(
        self,
        training_dir: str,
        default_shapenet: bool = True,
        all_files: Optional[List[str]] = None,
        label_dict: Dict[str, int] = None,
        num_input_points: int = 2048,
    ) -> None:
        """
        Args:
            training_dir: Root folder of shapenet dataset
        """
        self.training_dir = training_dir
        self.default_shapenet = default_shapenet
        self.num_input_points = num_input_points
        if default_shapenet:
            self.all_files = glob.glob(f"{self.training_dir}/*/*/models/model_normalized.obj")
            self.label_dict = {}
            for i, class_label in enumerate(os.listdir(training_dir)):
                self.label_dict[class_label] = i
        else:
            self.all_files = all_files
            self.label_dict = label_dict

    def _sample_point_cloud(self, vertices: torch.Tensor) -> torch.Tensor:
        """Samples a fixed-size point cloud from normalized mesh vertices."""
        num_vertices = vertices.shape[0]
        if num_vertices == 0:
            return torch.zeros([self.num_input_points, 3], dtype=torch.float32)
        if num_vertices >= self.num_input_points:
            indices = torch.randperm(num_vertices)[: self.num_input_points]
        else:
            extra = torch.randint(0, num_vertices, (self.num_input_points - num_vertices,))
            indices = torch.cat([torch.arange(num_vertices), extra], dim=0)
        return vertices[indices].to(torch.float32)

    def __len__(self) -> int:
        """Returns number of 3D objects"""
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns processed vertices, faces and class label of a mesh
        Args:
            idx: Which 3D object we're retrieving
        Returns:
            mesh_dict: Dictionary containing vertices, faces and class label
        """
        mesh_file = self.all_files[idx]
        vertices, faces, _ = load_obj(mesh_file)
        faces = faces.verts_idx
        vertices = vertices[:, [2, 0, 1]]
        vertices = data_utils.center_vertices(vertices)
        vertices = data_utils.normalize_vertices_scale(vertices)
        point_cloud = self._sample_point_cloud(vertices)
        vertices, faces, _ = data_utils.quantize_process_mesh(vertices, faces)
        faces = data_utils.flatten_faces(faces)
        vertices = vertices.to(torch.int32)
        faces = faces.to(torch.int32)
        if self.default_shapenet:
            class_label = self.label_dict[mesh_file.split("/")[-4]]
        else:
            class_label = self.label_dict[mesh_file]
        mesh_dict = {"vertices": vertices, "faces": faces, "class_label": class_label, "point_cloud": point_cloud}
        return mesh_dict


class ImageDataset(Dataset):
    def __init__(self, training_dir: str, image_extension: str = "jpeg") -> None:
        """Initializes Image Dataset

        Args:
            training_dir: Where model files along with renderings are located
            image_extension: Whether it's a .png or .jpeg or other type of file
        """
        self.training_dir = training_dir
        self.images = glob.glob(f"{self.training_dir}/*/*/renderings/*.{image_extension}")

        self.transforms = T.Compose([T.ToTensor(), T.Resize((256))])

    def __len__(self) -> int:
        """How many renderings we have"""
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Gets image object along with associated mesh

        Args:
            idx: Index of image to retrieve

        Returns:
            mesh_dict: Dictionary containing vertices, faces of .obj file and image tensor
        """
        img_file = self.images[idx]
        folder_path = "/".join(img_file.split("/")[:-2])
        model_file = os.path.sep.join([folder_path, "models", "model_normalized.obj"])
        verts, faces, _ = load_obj(model_file)
        faces = faces.verts_idx
        verts = verts[:, [2, 0, 1]]
        vertices = data_utils.center_vertices(verts)
        vertices = data_utils.normalize_vertices_scale(vertices)
        vertices, faces, _ = data_utils.quantize_process_mesh(vertices, faces)
        faces = data_utils.flatten_faces(faces)
        img = Image.open(img_file).convert("RGB")
        img = self.transforms(img)
        mesh_dict = {"vertices": vertices, "faces": faces, "image": img}
        return mesh_dict


def _count_vertices_in_obj(mesh_path: str) -> int:
    """Count vertex lines in an OBJ file (lines starting with 'v ')."""
    count = 0
    with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("v ") and not s.startswith("vn ") and not s.startswith("vt "):
                count += 1
    return count


class PairedObjXyzDataset(Dataset):
    """Dataset for paired building meshes and LiDAR point clouds.

    Expected directory structure:
      root/
        meshes/*.obj
        pointclouds/*.xyz
    where files are paired by stem, e.g. `1.obj` <-> `1.xyz`.
    """

    def __init__(
        self,
        root_dir: str,
        num_input_points: int = 2048,
        voxel_size: float = 0.01,
        max_xyz_abs: float = 1e6,
        max_retry: int = 20,
        bad_sample_log_file: str = "bad_xyz_samples.log",
        max_vertices_per_sample: Optional[int] = None,
    ) -> None:
        self.root_dir = root_dir
        self.mesh_dir = os.path.join(root_dir, "meshes")
        self.pointcloud_dir = os.path.join(root_dir, "pointclouds")
        self.num_input_points = num_input_points
        self.voxel_size = voxel_size
        self.max_xyz_abs = max_xyz_abs
        self.max_retry = max_retry
        self.bad_sample_log_file = os.path.join(root_dir, bad_sample_log_file)

        if (not os.path.isdir(self.mesh_dir)) or (not os.path.isdir(self.pointcloud_dir)):
            raise FileNotFoundError(
                f"Expected paired folders under {root_dir}: meshes/ and pointclouds/."
            )

        mesh_files = glob.glob(os.path.join(self.mesh_dir, "*.obj"))
        pc_files = glob.glob(os.path.join(self.pointcloud_dir, "*.xyz"))

        mesh_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mesh_files}
        pc_map = {os.path.splitext(os.path.basename(p))[0]: p for p in pc_files}

        shared_keys = sorted(set(mesh_map.keys()) & set(pc_map.keys()))
        if len(shared_keys) == 0:
            raise RuntimeError(f"No paired .obj/.xyz files found under {root_dir}.")

        self.pairs = [(mesh_map[k], pc_map[k]) for k in shared_keys]

        if max_vertices_per_sample is not None:
            filtered = []
            for mesh_path, xyz_path in self.pairs:
                try:
                    nv = _count_vertices_in_obj(mesh_path)
                    if nv <= max_vertices_per_sample:
                        filtered.append((mesh_path, xyz_path))
                except Exception:
                    continue
            self.pairs = filtered
            if len(self.pairs) == 0:
                raise RuntimeError(
                    f"No paired samples with <= {max_vertices_per_sample} vertices under {root_dir}."
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def _log_bad_sample(self, mesh_file: str, xyz_file: str, reason: str) -> None:
        with open(self.bad_sample_log_file, "a", encoding="utf-8") as f:
            f.write(f"mesh={mesh_file}\txyz={xyz_file}\treason={reason}\n")

    def _read_xyz_robust(self, xyz_file: str) -> torch.Tensor:
        valid_points = []
        bad_line_count = 0
        with open(xyz_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    bad_line_count += 1
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                except ValueError:
                    bad_line_count += 1
                    continue
                if (not np.isfinite(x)) or (not np.isfinite(y)) or (not np.isfinite(z)):
                    bad_line_count += 1
                    continue
                if (abs(x) > self.max_xyz_abs) or (abs(y) > self.max_xyz_abs) or (abs(z) > self.max_xyz_abs):
                    bad_line_count += 1
                    continue
                valid_points.append([x, y, z])
        if len(valid_points) == 0:
            raise RuntimeError("no valid xyz points after filtering")
        points = torch.tensor(valid_points, dtype=torch.float32)
        if bad_line_count > 0:
            # Keep sample but record that some lines were discarded.
            self._log_bad_sample("-", xyz_file, f"dropped_bad_lines={bad_line_count}")
        return points

    @staticmethod
    def _farthest_point_sample(points: torch.Tensor, target_n: int) -> torch.Tensor:
        n = points.shape[0]
        if n <= target_n:
            return points
        sampled_idx = torch.zeros(target_n, dtype=torch.long)
        distances = torch.full((n,), float("inf"), dtype=torch.float32)
        farthest = torch.randint(0, n, (1,), dtype=torch.long).item()
        for i in range(target_n):
            sampled_idx[i] = farthest
            centroid = points[farthest]
            dist = torch.sum((points - centroid) ** 2, dim=1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()
        return points[sampled_idx]

    def _voxel_downsample(self, points: torch.Tensor) -> torch.Tensor:
        if self.voxel_size <= 0:
            return points
        pts_np = points.cpu().numpy()
        voxel_idx = np.floor(pts_np / self.voxel_size).astype(np.int64)
        _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return points[torch.from_numpy(unique_idx).to(torch.long)]

    def _sample_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] == 0:
            return torch.zeros([self.num_input_points, 3], dtype=torch.float32)
        points = self._voxel_downsample(points)
        if points.shape[0] >= self.num_input_points:
            points = self._farthest_point_sample(points, self.num_input_points)
            return points.to(torch.float32)
        extra = torch.randint(0, points.shape[0], (self.num_input_points - points.shape[0],))
        points = torch.cat([points, points[extra]], dim=0)
        return points.to(torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        attempts = 0
        curr_idx = idx
        while attempts < self.max_retry:
            mesh_file, xyz_file = self.pairs[curr_idx]
            try:
                mesh_vertices, faces, _ = load_obj(mesh_file)
                faces = faces.verts_idx
                # Keep coordinate convention consistent with existing PolyGen pipeline.
                mesh_vertices = mesh_vertices[:, [2, 0, 1]]

                xyz_points = self._read_xyz_robust(xyz_file)[:, [2, 0, 1]]

                # Align both condition and target with the same mesh-derived normalization.
                vert_min, _ = torch.min(mesh_vertices, dim=0)
                vert_max, _ = torch.max(mesh_vertices, dim=0)
                center = 0.5 * (vert_min + vert_max)
                extents = vert_max - vert_min
                scale = torch.sqrt(torch.sum(extents ** 2))
                if torch.isclose(scale, torch.tensor(0.0, device=scale.device)):
                    scale = torch.tensor(1.0, device=scale.device)

                mesh_vertices = (mesh_vertices - center) / scale
                xyz_points = (xyz_points - center) / scale
                point_cloud = self._sample_point_cloud(xyz_points)

                mesh_vertices, faces, _ = data_utils.quantize_process_mesh(mesh_vertices, faces)
                faces = data_utils.flatten_faces(faces)
                mesh_vertices = mesh_vertices.to(torch.int32)
                faces = faces.to(torch.int32)

                # class_label is unused in point-cloud mode but kept for compatibility.
                return {
                    "vertices": mesh_vertices,
                    "faces": faces,
                    "class_label": 0,
                    "point_cloud": point_cloud,
                }
            except Exception as e:
                self._log_bad_sample(mesh_file, xyz_file, str(e))
                attempts += 1
                curr_idx = random.randint(0, len(self.pairs) - 1)
        raise RuntimeError(
            f"Exceeded max retries ({self.max_retry}) when loading valid paired sample. "
            f"See {self.bad_sample_log_file}"
        )


class CollateMethod(Enum):
    VERTICES = 1
    FACES = 2
    IMAGES = 3
    POINT_CLOUD = 4


class PolygenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        collate_method: CollateMethod,
        batch_size: int,
        training_split: float = 0.925,
        val_split: float = 0.025,
        default_shapenet: bool = True,
        quantization_bits: int = 8,
        use_image_dataset: bool = False,
        use_point_cloud_dataset: bool = False,
        img_extension: str = "jpeg",
        num_input_points: int = 2048,
        point_cloud_voxel_size: float = 0.01,
        point_cloud_max_xyz_abs: float = 1e6,
        point_cloud_max_retry: int = 20,
        point_cloud_bad_sample_log_file: str = "bad_xyz_samples.log",
        max_vertices_per_sample: Optional[int] = None,
        all_files: Optional[List[str]] = None,
        label_dict: Optional[Dict[str, int]] = None,
        apply_random_shift_vertices: bool = True,
        apply_random_shift_faces: bool = True,
        shuffle_vertices: bool = True,
    ) -> None:
        """
        Args:
            data_dir: Root directory for shapenet dataset
            collate_method: Whether to collate vertices or faces
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            default_shapenet: Whether or not we are using the default shapenet data structure
            quantization_bits: How many bits we are using to quantize the vertices
            use_image_dataset: Whether to use the image shapenet dataset or the regular shapenet dataset
            img_extension: Whether the images are .jpeg or .png files
            all_files: List of all .obj files (needs to be provided if default_shapnet = false)
            label_dict: Mapping of .obj file to class label (needs to be provided if default_shapnet = false)
            apply_random_shift_vertices: Whether or not we're applying random shift to vertices for vertex model
            apply_random_shift_faces: Whether or not we're applying random shift to vertices for face model
            shuffle_vertices: Whether or not we're shuffling the order of vertices during batch generation for face model
        """
        super().__init__()

        # If we are using the image dataset, then the collate method should not be for
        # class-conditioned vertices. It should be for image-conditioned vertices
        # or vertex-conditioned faces
        assert (not use_image_dataset) or (collate_method == CollateMethod.IMAGES)
        assert (not use_point_cloud_dataset) or (collate_method == CollateMethod.POINT_CLOUD)
        assert not (use_image_dataset and use_point_cloud_dataset)
        assert (training_split + val_split) <= 1.0

        self.data_dir = data_dir
        self.batch_size = batch_size

        if use_image_dataset:
            self.shapenet_dataset = ImageDataset(training_dir=self.data_dir, image_extension=img_extension)
        elif use_point_cloud_dataset and os.path.isdir(os.path.join(self.data_dir, "meshes")) and os.path.isdir(
            os.path.join(self.data_dir, "pointclouds")
        ):
            self.shapenet_dataset = PairedObjXyzDataset(
                root_dir=self.data_dir,
                num_input_points=num_input_points,
                voxel_size=point_cloud_voxel_size,
                max_xyz_abs=point_cloud_max_xyz_abs,
                max_retry=point_cloud_max_retry,
                bad_sample_log_file=point_cloud_bad_sample_log_file,
                max_vertices_per_sample=max_vertices_per_sample,
            )
        else:
            self.shapenet_dataset = ShapenetDataset(
                self.data_dir,
                default_shapenet=default_shapenet,
                all_files=all_files,
                label_dict=label_dict,
                num_input_points=num_input_points,
            )

        self.training_split = training_split
        self.val_split = val_split
        self.quantization_bits = quantization_bits
        self.apply_random_shift_vertices = apply_random_shift_vertices
        self.apply_random_shift_faces = apply_random_shift_faces
        self.shuffle_vertices = shuffle_vertices
        self.max_vertices_per_sample = max_vertices_per_sample

        # 标记是否已经将划分结果写入到 txt（避免重复写）
        self._split_files_written: bool = False

        if collate_method == CollateMethod.VERTICES:
            self.collate_fn = self.collate_vertex_model_batch
        elif collate_method == CollateMethod.FACES:
            self.collate_fn = self.collate_face_model_batch
        elif collate_method == CollateMethod.IMAGES:
            self.collate_fn = self.collate_img_model_batch
        elif collate_method == CollateMethod.POINT_CLOUD:
            self.collate_fn = self.collate_point_cloud_model_batch

    def collate_vertex_model_batch(self, ds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Applying padding to different length vertex sequences so we can batch them
        Args:
            ds: List of dictionaries where each dictionary has information about a 3D object
        Returns
            vertex_model_batch: A single dictionary which represents the whole batch
        """
        vertex_model_batch = {}
        num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
        max_vertices = max(num_vertices_list)
        if self.max_vertices_per_sample is not None:
            max_vertices = min(max_vertices, self.max_vertices_per_sample)
        num_elements = len(ds)
        vertex_tokens = torch.zeros([num_elements, max_vertices + 1], dtype=torch.int32)
        vertices_zyx = torch.zeros([num_elements, max_vertices, 3], dtype=torch.int32)
        class_labels = torch.zeros([num_elements], dtype=torch.int32)
        vertex_tokens_mask = torch.zeros_like(vertex_tokens, dtype=torch.int32)
        for i, element in enumerate(ds):
            vertices = element["vertices"]
            if self.apply_random_shift_vertices:
                vertices = data_utils.random_shift(vertices)
            initial_vertex_size = vertices.shape[0]
            n = min(initial_vertex_size, max_vertices)
            curr_vertices_zyx = torch.stack([vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1)
            vertices_zyx[i, :n] = curr_vertices_zyx[:n]
            vertex_tokens[i, :n] = 1
            class_labels[i] = torch.Tensor([element["class_label"]])
            vertex_tokens_mask[i, : n + 1] = 1
        vertex_model_batch["vertex_tokens"] = vertex_tokens
        vertex_model_batch["vertices_zyx"] = vertices_zyx
        vertex_model_batch["class_label"] = class_labels
        vertex_model_batch["vertex_tokens_mask"] = vertex_tokens_mask
        return vertex_model_batch

    def collate_face_model_batch(
        self,
        ds: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Applies padding to different length face sequences so we can batch them
        Args:
            ds: List of dictionaries with each dictionary containing info about a specific 3D object

        Returns:
            face_model_batch: A single dictionary which represents the whole face model batch
        """
        face_model_batch = {}
        num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
        max_vertices = max(num_vertices_list)
        num_faces_list = [shape_dict["faces"].shape[0] for shape_dict in ds]
        max_faces = max(num_faces_list)
        num_elements = len(ds)

        shuffled_faces = torch.zeros([num_elements, max_faces], dtype=torch.int32)
        face_vertices = torch.zeros([num_elements, max_vertices, 3])
        face_vertices_mask = torch.zeros([num_elements, max_vertices], dtype=torch.int32)
        faces_mask = torch.zeros_like(shuffled_faces, dtype=torch.int32)

        for i, element in enumerate(ds):
            vertices = element["vertices"]
            num_vertices = vertices.shape[0]
            if self.apply_random_shift_faces:
                vertices = data_utils.random_shift(vertices)

            if self.shuffle_vertices:
                permutation = torch.randperm(num_vertices)
                vertices = vertices[permutation]
                vertices = vertices.unsqueeze(0)
                face_permutation = torch.cat(
                    [
                        torch.Tensor([0, 1]).to(torch.int32),
                        torch.argsort(permutation).to(torch.int32) + 2,
                    ],
                    dim=0,
                )
                curr_faces = face_permutation[element["faces"].to(torch.int64)][None]
            else:
                curr_faces = element["faces"][None]

            vertex_padding_size = max_vertices - num_vertices
            initial_faces_size = curr_faces.shape[1]
            face_padding_size = max_faces - initial_faces_size
            shuffled_faces[i] = F.pad(curr_faces, [0, face_padding_size, 0, 0])
            curr_verts = data_utils.dequantize_verts(vertices, self.quantization_bits)
            face_vertices[i] = F.pad(curr_verts, [0, 0, 0, vertex_padding_size])
            face_vertices_mask[i] = torch.zeros_like(face_vertices[i][..., 0], dtype=torch.float32)
            face_vertices_mask[i, :num_vertices] = 1
            faces_mask[i] = torch.zeros_like(shuffled_faces[i], dtype=torch.float32)
            faces_mask[i, : initial_faces_size + 1] = 1
        face_model_batch["faces"] = shuffled_faces
        face_model_batch["vertices"] = face_vertices
        face_model_batch["vertices_mask"] = face_vertices_mask
        face_model_batch["faces_mask"] = faces_mask
        return face_model_batch

    def collate_img_model_batch(self, ds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Applies padding to different length vertex sequences and collects images for batching

        Args:
            ds: List of dictionaries where each dictionary has information about a 3D object

        Returns:
            img_vertex_model_batch: A single dictionary which represents the whole batch
        """
        img_vertex_model_batch = {}
        num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
        max_vertices = max(num_vertices_list)
        if self.max_vertices_per_sample is not None:
            max_vertices = min(max_vertices, self.max_vertices_per_sample)
        num_elements = len(ds)
        vertex_tokens = torch.zeros([num_elements, max_vertices + 1], dtype=torch.int32)
        vertices_zyx = torch.zeros([num_elements, max_vertices, 3], dtype=torch.int32)
        vertex_tokens_mask = torch.zeros_like(vertex_tokens, dtype=torch.int32)
        images = torch.zeros(
            [
                num_elements,
                ds[0]["image"].shape[0],
                ds[0]["image"].shape[1],
                ds[0]["image"].shape[2],
            ]
        )

        for i, element in enumerate(ds):
            vertices = element["vertices"]
            initial_vertex_size = vertices.shape[0]
            n = min(initial_vertex_size, max_vertices)
            vertices_zyx[i, :n] = torch.stack(
                [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
            )[:n]
            vertex_tokens[i, :n] = 1
            vertex_tokens_mask[i, : n + 1] = 1

            images[i] = element["image"]

        img_vertex_model_batch["vertex_tokens"] = vertex_tokens
        img_vertex_model_batch["vertices_zyx"] = vertices_zyx
        img_vertex_model_batch["vertex_tokens_mask"] = vertex_tokens_mask
        img_vertex_model_batch["image"] = images
        return img_vertex_model_batch

    def collate_point_cloud_model_batch(self, ds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Applies vertex padding and batches input point clouds. Caps vertex count to max_vertices_per_sample if set."""
        pc_vertex_model_batch = {}
        num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
        max_vertices = max(num_vertices_list)
        if self.max_vertices_per_sample is not None:
            max_vertices = min(max_vertices, self.max_vertices_per_sample)
        num_elements = len(ds)

        vertex_tokens = torch.zeros([num_elements, max_vertices + 1], dtype=torch.int32)
        vertices_zyx = torch.zeros([num_elements, max_vertices, 3], dtype=torch.int32)
        vertex_tokens_mask = torch.zeros_like(vertex_tokens, dtype=torch.int32)
        point_clouds = torch.stack([shape_dict["point_cloud"] for shape_dict in ds], dim=0).to(torch.float32)

        for i, element in enumerate(ds):
            vertices = element["vertices"]
            initial_vertex_size = vertices.shape[0]
            n = min(initial_vertex_size, max_vertices)
            vertices_zyx[i, :n] = torch.stack(
                [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
            )[:n]
            vertex_tokens[i, :n] = 1
            vertex_tokens_mask[i, : n + 1] = 1

        pc_vertex_model_batch["vertex_tokens"] = vertex_tokens
        pc_vertex_model_batch["vertices_zyx"] = vertices_zyx
        pc_vertex_model_batch["vertex_tokens_mask"] = vertex_tokens_mask
        pc_vertex_model_batch["point_cloud"] = point_clouds
        return pc_vertex_model_batch

    def setup(self, stage: Optional = None) -> None:
        """Pytorch Lightning Data Module setup method"""
        base_ds = self.shapenet_dataset

        # 1) 如果存在之前写出的 train/val/test txt，则按 txt 固定划分
        if not self._split_files_written:
            if self._load_split_from_files(base_ds):
                self._split_files_written = True
                return

        # 2) 否则按比例随机划分一次，并写出 txt 供之后复现
        num_files = len(base_ds)
        train_set_length = int(num_files * self.training_split)
        val_set_length = int(num_files * self.val_split)
        test_set_length = num_files - train_set_length - val_set_length
        self.train_set, self.val_set, self.test_set = random_split(
            base_ds, [train_set_length, val_set_length, test_set_length]
        )

        if not self._split_files_written:
            self._write_split_file_lists()
            self._split_files_written = True

    # ---- 划分结果写入 / 读回 ----

    def _write_split_file_lists(self) -> None:
        """将训练/验证/测试划分对应的文件名写入 txt."""
        base_ds = self.shapenet_dataset

        # 不同数据集类型的“文件名”获取方式不同
        if isinstance(base_ds, PairedObjXyzDataset):
            root_dir = base_ds.root_dir

            def _line_for_idx(i: int) -> str:
                mesh_path, xyz_path = base_ds.pairs[i]
                return f"{os.path.basename(mesh_path)}\t{os.path.basename(xyz_path)}"

        elif isinstance(base_ds, ShapenetDataset):
            root_dir = base_ds.training_dir

            def _line_for_idx(i: int) -> str:
                return base_ds.all_files[i]

        else:
            # 兜底：只写索引
            root_dir = getattr(self, "data_dir", ".")

            def _line_for_idx(i: int) -> str:
                return str(i)

        def _dump_subset(subset, filename: str) -> None:
            if subset is None or not hasattr(subset, "indices"):
                return
            out_path = os.path.join(root_dir, filename)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    for idx in subset.indices:
                        f.write(_line_for_idx(int(idx)) + "\n")
            except Exception:
                # 不影响训练流程，写失败时静默跳过
                pass

        _dump_subset(self.train_set, "train_files.txt")
        _dump_subset(self.val_set, "val_files.txt")
        _dump_subset(self.test_set, "test_files.txt")

    def _load_split_from_files(self, base_ds: Dataset) -> bool:
        """如果存在 train/val/test txt，则按照其中记录的样本恢复划分。

        返回:
            bool: True 表示成功从 txt 恢复划分；False 表示未恢复（将继续使用随机划分）。
        """
        # 根据数据集类型确定根目录与 key 生成方式
        if isinstance(base_ds, PairedObjXyzDataset):
            root_dir = base_ds.root_dir

            def _key_for_idx(i: int) -> str:
                mesh_path, xyz_path = base_ds.pairs[i]
                return f"{os.path.basename(mesh_path)}\t{os.path.basename(xyz_path)}"

        elif isinstance(base_ds, ShapenetDataset):
            root_dir = base_ds.training_dir

            def _key_for_idx(i: int) -> str:
                return base_ds.all_files[i]

        else:
            root_dir = getattr(self, "data_dir", ".")

            def _key_for_idx(i: int) -> str:
                return str(i)

        train_path = os.path.join(root_dir, "train_files.txt")
        val_path = os.path.join(root_dir, "val_files.txt")
        test_path = os.path.join(root_dir, "test_files.txt")
        if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            return False

        try:
            def _read_lines(p: str) -> List[str]:
                with open(p, "r", encoding="utf-8") as f:
                    return [ln.rstrip("\n") for ln in f if ln.strip()]

            train_keys = _read_lines(train_path)
            val_keys = _read_lines(val_path)
            test_keys = _read_lines(test_path)
        except Exception:
            return False

        # 为当前数据集构建 key -> index 映射
        index_map: Dict[str, int] = {}
        for i in range(len(base_ds)):
            k = _key_for_idx(i)
            # 如果 key 重复，只保留第一个
            if k not in index_map:
                index_map[k] = i

        def _map_keys_to_indices(keys: List[str]) -> List[int]:
            idxs: List[int] = []
            for k in keys:
                if k in index_map:
                    idxs.append(index_map[k])
            return idxs

        train_indices = _map_keys_to_indices(train_keys)
        val_indices = _map_keys_to_indices(val_keys)
        test_indices = _map_keys_to_indices(test_keys)

        # 至少要有训练集索引，否则认为恢复失败
        if not train_indices:
            return False

        from torch.utils.data import Subset

        self.train_set = Subset(base_ds, train_indices)
        self.val_set = Subset(base_ds, val_indices) if val_indices else None
        self.test_set = Subset(base_ds, test_indices) if test_indices else None
        return True

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            train_dataloader: Dataloader used to load training batches
        """
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=16,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            val_dataloader: Dataloader used to load validation batches
        """
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            test_dataloader: Dataloader used to load test batches
        """
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=8,
            persistent_workers=True,
        )

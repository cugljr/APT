from typing import Any, Dict, Optional

import torch

from polygen.modules.vertex_model import VertexModel, ImageToVertexModel, PointCloudToVertexModel
from polygen.modules.face_model import FaceModel, PointCloudToFaceModel
from polygen.modules.data_modules import PolygenDataModule, CollateMethod


class VertexModelConfig:
    def __init__(
        self,
        accelerator: str,
        dataset_path: str,
        batch_size: int,
        training_split: float,
        val_split: float,
        apply_random_shift: bool,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        class_conditional: bool,
        num_classes: int,
        max_num_input_verts: int,
        use_discrete_embeddings: bool,
        learning_rate: float,
        step_size: int,
        gamma: float,
        training_steps: int,
        image_model: bool = False,
        point_cloud_model: bool = False,
        num_input_points: int = 2048,
        num_context_tokens: int = 256,
        point_cloud_knn_scales: Optional[list] = None,
        geometric_loss_weight: float = 0.1,
        chamfer_max_points: int = 1024,
        stop_loss_weight: float = 1.0,
        length_loss_weight: float = 0.0,
        length_loss_type: str = "huber",
        length_huber_delta: float = 10.0,
        point_cloud_voxel_size: float = 0.01,
        point_cloud_max_xyz_abs: float = 1e6,
        point_cloud_max_retry: int = 20,
        point_cloud_bad_sample_log_file: str = "bad_xyz_samples.log",
        gpu_ids: Optional[list] = None,
        use_wandb: bool = False,
        wandb_project: str = "polygen",
        wandb_run_name: Optional[str] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
    ) -> None:
        """Initializes vertex model and vertex data module

        Args:
            accelerator: data parallel or distributed data parallel
            gpu_ids: GPU ids to use, e.g. [0, 1]. If None, use all visible GPUs.
            dataset_path: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            apply_random_shift: Whether or not we're applying random shift to vertices
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            class_conditional: If True, then condition on learned class embeddings
            num_classes: Number of classes to condition on
            max_num_input_verts:  Maximum number of vertices. Used for learned position embeddings.
            use_discrete_embeddings: Discrete embedding layers or linear layers for vertices
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
            training_steps: How many total steps we want to train for
            image_model: Whether we're training the image model or class-conditioned model
            point_cloud_model: Whether we're training point-cloud-conditioned vertex model
            num_input_points: Number of input points sampled per object for condition
            num_context_tokens: Number of condition tokens passed to decoder cross-attention
            point_cloud_knn_scales: kNN neighborhood sizes for multi-scale point-cloud encoding
            geometric_loss_weight: Weight for Chamfer geometric consistency loss
            chamfer_max_points: Number of condition points used in Chamfer computation
            stop_loss_weight: Weight for stop-token NLL term
            length_loss_weight: Weight for vertex-count consistency loss
            length_loss_type: Length loss type, one of {"huber", "l1"}
            length_huber_delta: Huber delta for length loss
            point_cloud_voxel_size: Voxel size for point-cloud downsampling in paired dataset
            point_cloud_max_xyz_abs: Absolute-value cap when filtering invalid xyz rows
            point_cloud_max_retry: Retry count for skipping corrupted samples
            point_cloud_bad_sample_log_file: Log file name for rejected xyz samples
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name (None for auto)
            early_stopping_patience: Early stopping patience on val_loss_mean
            early_stopping_min_delta: Minimum delta for early stopping
        """

        self.gpu_ids = gpu_ids
        self.num_gpus = torch.cuda.device_count() if gpu_ids is None else len(gpu_ids)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.accelerator = accelerator
        if accelerator.startswith("ddp"):
            self.batch_size = batch_size // self.num_gpus
        else:
            self.batch_size = batch_size

        if image_model:
            collate_method = CollateMethod.IMAGES
            self.vertex_model = ImageToVertexModel(
                decoder_config = decoder_config, 
                quantization_bits = quantization_bits,
                use_discrete_embeddings = use_discrete_embeddings,
                max_num_input_verts = max_num_input_verts,
                learning_rate = learning_rate,
                step_size = step_size,
                gamma = gamma,
            )
        elif point_cloud_model:
            collate_method = CollateMethod.POINT_CLOUD
            if point_cloud_knn_scales is None:
                point_cloud_knn_scales = [8, 16, 32]
            self.vertex_model = PointCloudToVertexModel(
                decoder_config=decoder_config,
                quantization_bits=quantization_bits,
                use_discrete_embeddings=use_discrete_embeddings,
                max_num_input_verts=max_num_input_verts,
                learning_rate=learning_rate,
                step_size=step_size,
                gamma=gamma,
                num_context_tokens=num_context_tokens,
                knn_scales=tuple(point_cloud_knn_scales),
                geometric_loss_weight=geometric_loss_weight,
                chamfer_max_points=chamfer_max_points,
                stop_loss_weight=stop_loss_weight,
                length_loss_weight=length_loss_weight,
                length_loss_type=length_loss_type,
                length_huber_delta=length_huber_delta,
            )
        else:
            collate_method = CollateMethod.VERTICES
            self.vertex_model = VertexModel(
                decoder_config=decoder_config,
                quantization_bits=quantization_bits,
                class_conditional=class_conditional,
                num_classes=num_classes,
                max_num_input_verts=max_num_input_verts,
                use_discrete_embeddings=use_discrete_embeddings,
                learning_rate=learning_rate,
                step_size=step_size,
                gamma=gamma,
            )


        self.vertex_data_module = PolygenDataModule(
            data_dir=dataset_path,
            batch_size=self.batch_size,
            collate_method=collate_method,
            training_split=training_split,
            val_split=val_split,
            quantization_bits=quantization_bits,
            use_image_dataset=image_model,
            use_point_cloud_dataset=point_cloud_model,
            num_input_points=num_input_points,
            point_cloud_voxel_size=point_cloud_voxel_size,
            point_cloud_max_xyz_abs=point_cloud_max_xyz_abs,
            point_cloud_max_retry=point_cloud_max_retry,
            point_cloud_bad_sample_log_file=point_cloud_bad_sample_log_file,
            max_vertices_per_sample=max_num_input_verts,
            apply_random_shift_vertices=(apply_random_shift and (not point_cloud_model)),
        )

        self.training_steps = training_steps

class FaceModelConfig:
    def __init__(
        self,
        accelerator: str,
        dataset_path: str,
        batch_size: int,
        training_split: float,
        val_split: float,
        apply_random_shift: bool,
        shuffle_vertices: bool,
        encoder_config: Dict,
        decoder_config: Dict,
        class_conditional: bool,
        num_classes: int,
        decoder_cross_attention: bool,
        use_discrete_vertex_embeddings: bool,
        quantization_bits: int,
        max_seq_length: int,
        learning_rate: float,
        step_size: int,
        gamma: float,
        training_steps: int,
        point_cloud_model: bool = False,
        num_input_points: int = 2048,
        num_context_tokens: int = 256,
        point_cloud_knn_scales: Optional[list] = None,
        point_cloud_voxel_size: float = 0.01,
        point_cloud_max_xyz_abs: float = 1e6,
        point_cloud_max_retry: int = 20,
        point_cloud_bad_sample_log_file: str = "bad_xyz_samples.log",
        gpu_ids: Optional[list] = None,
    ):
        """Initializes face model and face data module

        Args:
            accelerator: data parallel or distributed data parallel
            gpu_ids: GPU ids to use, e.g. [0, 1]. If None, use all visible GPUs.
            dataset_path: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            apply_random_shift: Whether or not we're applying random shift to vertices
            shuffle_vertices: Whether or not we are randomly shuffling the vertices during batch generation
            encoder_config: Dictionary representing config for PolygenEncoder
            decoder_config: Dictionary representing config for TransformerDecoder
            class_conditional: If we are using global context embeddings based on class labels
            num_classes: How many distinct classes in the dataset
            decoder_cross_attention: If we are using cross attention within the decoder
            use_discrete_vertex_embeddings: Are the inputted vertices quantized
            quantization_bits: How many bits are we using to encode the vertices
            max_seq_length: Max number of face indices we can generate
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
            training_steps: How many total steps we want to train for
            point_cloud_model: Whether to train face model with paired point-cloud condition
            num_input_points: Number of point-cloud samples per object
            num_context_tokens: Number of point-cloud context tokens for cross-attention
            point_cloud_knn_scales: kNN scales used in point-cloud encoder
            point_cloud_voxel_size: Voxel size for point-cloud downsampling
            point_cloud_max_xyz_abs: Absolute-value cap when filtering invalid xyz rows
            point_cloud_max_retry: Retry count for skipping corrupted samples
            point_cloud_bad_sample_log_file: Log file name for rejected xyz samples
        """

        self.gpu_ids = gpu_ids
        self.num_gpus = torch.cuda.device_count() if gpu_ids is None else len(gpu_ids)
        self.accelerator = accelerator
        if accelerator.startswith("ddp"):
            self.batch_size = batch_size // self.num_gpus
        else:
            self.batch_size = batch_size

        self.face_data_module = PolygenDataModule(
            data_dir = dataset_path,
            batch_size = self.batch_size,
            collate_method = CollateMethod.FACES,
            training_split = training_split,
            val_split = val_split,
            quantization_bits = quantization_bits,
            use_point_cloud_dataset = point_cloud_model,
            num_input_points = num_input_points,
            point_cloud_voxel_size = point_cloud_voxel_size,
            point_cloud_max_xyz_abs = point_cloud_max_xyz_abs,
            point_cloud_max_retry = point_cloud_max_retry,
            point_cloud_bad_sample_log_file = point_cloud_bad_sample_log_file,
            apply_random_shift_faces = apply_random_shift,
            shuffle_vertices = shuffle_vertices,
        )

        if point_cloud_model:
            if point_cloud_knn_scales is None:
                point_cloud_knn_scales = [8, 16, 32]
            self.face_model = PointCloudToFaceModel(
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                class_conditional=class_conditional,
                num_classes=num_classes,
                decoder_cross_attention=decoder_cross_attention,
                use_discrete_vertex_embeddings=use_discrete_vertex_embeddings,
                quantization_bits=quantization_bits,
                max_seq_length=max_seq_length,
                learning_rate=learning_rate,
                step_size=step_size,
                gamma=gamma,
                num_context_tokens=num_context_tokens,
                knn_scales=tuple(point_cloud_knn_scales),
            )
        else:
            self.face_model = FaceModel(
                encoder_config = encoder_config,
                decoder_config = decoder_config,
                class_conditional = class_conditional,
                num_classes = num_classes,
                decoder_cross_attention = decoder_cross_attention,
                use_discrete_vertex_embeddings = use_discrete_vertex_embeddings,
                quantization_bits = quantization_bits,
                max_seq_length = max_seq_length,
                learning_rate = learning_rate,
                step_size = step_size,
                gamma = gamma,
            )
        
        self.training_steps = training_steps


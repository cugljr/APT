from typing import Dict, Optional, Tuple, List, Any
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from polygen.utils.data_utils import dequantize_verts

from .polygen_decoder import TransformerDecoder
from .utils import top_k_logits, top_p_logits
from .image_encoder import PolygenResnet


class VertexModel(pl.LightningModule):
    """Autoregressive Generative Model of Quantized Mesh Vertices.
    Operates on vertex-position sequences with a stopping token:
    [VERT_0, VERT_1, ..., VERT_{n-1}, STOP]
    At each VERT position, predicts quantized coordinates hierarchically in-order:
    z -> y -> x.
    """

    def __init__(
        self,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        class_conditional: bool = False,
        num_classes: int = 55,
        max_num_input_verts: int = 2500,
        use_discrete_embeddings: bool = True,
        learning_rate: float = 3e-4,
        step_size: int = 5000,
        gamma: float = 0.9995,
    ) -> None:
        """Initializes VertexModel. The encoder can be a model with a Resnet backbone for image contexts and voxel contexts.
        However for class label context, the encoder is simply the class embedder.

        Args:
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            class_conditional: If True, then condition on learned class embeddings
            num_classes: Number of classes to condition on
            max_num_input_verts:  Maximum number of vertices. Used for learned position embeddings.
            use_discrete_embeddings: Discrete embedding layers or linear layers for vertices
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
        """

        super(VertexModel, self).__init__()
        self.decoder_config = decoder_config
        self.quantization_bits = quantization_bits
        self.embedding_dim = decoder_config["hidden_size"]
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.use_discrete_embeddings = use_discrete_embeddings
        self.decoder = TransformerDecoder(**decoder_config)
        self.class_embedder = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_dim)
        self.vertex_token_embedder = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)
        self.coord0_embedder = nn.Embedding(2 ** self.quantization_bits, self.embedding_dim)
        self.coord1_embedder = nn.Embedding(2 ** self.quantization_bits, self.embedding_dim)
        self.coord2_embedder = nn.Embedding(2 ** self.quantization_bits, self.embedding_dim)
        self.pos_embedder = nn.Embedding(num_embeddings=self.max_num_input_verts, embedding_dim=self.embedding_dim)
        self.stop_head = nn.Linear(self.embedding_dim, 2)
        self.z_head = nn.Linear(self.embedding_dim, 2 ** self.quantization_bits)
        self.y_head = nn.Linear(self.embedding_dim, 2 ** self.quantization_bits)
        self.x_head = nn.Linear(self.embedding_dim, 2 ** self.quantization_bits)

        zero_embeddings_tensor = torch.randn([1, 1, self.embedding_dim], device=self.device)
        self.zero_embed = nn.Parameter(zero_embeddings_tensor)

        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma

    def _embed_class_label(self, labels: torch.Tensor) -> torch.Tensor:
        """Embeds Class Label with learned embedding matrix

        Args:
            labels: A Tensor with shape [batch_size,]. Represents the class label for each sample in the batch.
        Returns:
            embeddings: A Tensor with shape [batch_size, embed_size].
        """
        return self.class_embedder(labels.to(torch.int64))

    def _prepare_context(self, context: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepares global context embedding

        Args:
            context: A dictionary that contains a key of class_label
        Returns:
            global_context_embeddings: A Tensor of shape [batch_size, embed_size]
            sequential_context_embeddings: None
        """
        if self.class_conditional:
            global_context_embedding = self._embed_class_label(context["class_label"])
        else:
            global_context_embedding = None
        return global_context_embedding, None

    def _embed_inputs(
        self,
        vertex_tokens: torch.Tensor,
        vertices_zyx: torch.Tensor,
        global_context_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Embeds vertex-position tokens and previous vertices.

        Args:
            vertex_tokens: A Tensor of shape [batch_size, sample_length].
            vertices_zyx: A Tensor of shape [batch_size, sample_length, 3]. Quantized z-y-x per position.
            global_context_embedding: A Tensor of shape [batch_size, embed_size]. Represents class label conditioning.
        Returns:
            embeddings: A Tensor of shape [sample_length + 1, batch_size]. Represents combination of embeddings with global context embeddings. The first and second
                        dimensions are transposed for the sake of the decoder.
        """
        input_shape = vertex_tokens.shape
        batch_size, seq_length = input_shape[0], input_shape[1]
        pos_embeddings = self.pos_embedder(torch.arange(seq_length, device=self.device))
        token_embeddings = self.vertex_token_embedder(vertex_tokens)
        vert_embeddings = (
            token_embeddings
            + self.coord0_embedder(vertices_zyx[..., 0])
            + self.coord1_embedder(vertices_zyx[..., 1])
            + self.coord2_embedder(vertices_zyx[..., 2])
            + pos_embeddings[None]
        )
        if global_context_embedding is None:
            zero_embed_tiled = torch.repeat_interleave(self.zero_embed, batch_size, dim=0)
        else:
            zero_embed_tiled = global_context_embedding[:, None].to(
                torch.float32
            )  # Zero embed tiled is of shape [batch_size, 1, embed_size]

        # Embeddings shape before concatenation is [batch_size, seq_length, embed_size], after concatenation it is [batch_size, seq_length + 1, embed_size]
        embeddings = torch.cat([zero_embed_tiled, vert_embeddings], dim=1)

        # Changing the dimension from [batch_size, seq_length, embed_size] to [seq_length, batch_size, embed_size] for TransformerDecoder
        return embeddings.transpose(0, 1)

    def _compute_heads(
        self,
        hidden: torch.Tensor,
        z_teacher: Optional[torch.Tensor] = None,
        y_teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs decoder states through hierarchical output heads.

        Args:
            hidden: Tensor of shape [batch_size, sequence_length, embed_size].
            z_teacher: Optional z targets for teacher-forced y prediction.
            y_teacher: Optional y targets for teacher-forced x prediction.
        Returns:
            stop_logits: Tensor of shape [batch_size, sequence_length, 2].
            z_logits: Tensor of shape [batch_size, sequence_length, q_levels].
            y_logits: Tensor of shape [batch_size, sequence_length, q_levels].
            x_logits: Tensor of shape [batch_size, sequence_length, q_levels].
        """
        stop_logits = self.stop_head(hidden)
        z_logits = self.z_head(hidden)

        if z_teacher is None:
            z_tokens = torch.argmax(z_logits, dim=-1)
        else:
            z_tokens = z_teacher
        y_hidden = hidden + self.coord0_embedder(z_tokens)
        y_logits = self.y_head(y_hidden)

        if y_teacher is None:
            y_tokens = torch.argmax(y_logits, dim=-1)
        else:
            y_tokens = y_teacher
        x_hidden = y_hidden + self.coord1_embedder(y_tokens)
        x_logits = self.x_head(x_hidden)
        return stop_logits, z_logits, y_logits, x_logits

    def _decode_hidden(
        self,
        vertex_tokens: torch.Tensor,
        vertices_zyx: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
        sequential_context_embedding: Optional[torch.Tensor] = None,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Decodes autoregressive hidden states.

        Args:
            vertex_tokens: A Tensor of shape [batch_size, sequence_length].
            vertices_zyx: A Tensor of shape [batch_size, sequence_length, 3].
            global_context_embedding: A Tensor of shape [batch_size, embed_size]. Represents conditioning on class labels.
            sequential_context_embeddings: A Tensor of shape [batch_size, context_seq_length, context_embed_size]. Represents conditioning on images or voxels.
            cache:  A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
        Returns:
            hidden: Decoder hidden states of shape [batch_size, sequence_length + 1, embedding_dim].
        """
        decoder_inputs = self._embed_inputs(
            vertex_tokens.to(torch.int64), vertices_zyx.to(torch.int64), global_context_embedding
        )
        if cache is not None:
            decoder_inputs = decoder_inputs[-1:, :]
        if sequential_context_embedding is not None:
            sequential_context_embedding = sequential_context_embedding.transpose(0, 1)
        hidden = self.decoder(
            decoder_inputs,
            sequential_context_embeddings=sequential_context_embedding,
            cache=cache,
        ).transpose(
            0, 1
        )  # Transpose to convert from [seq_length, batch_size, embedding_dim] to [batch_size, seq_length, embedding_dim]
        return hidden

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward method for Vertex Model
        Args:
            batch: A dictionary with keys vertex_tokens and vertices_zyx.
        Returns:
            logits: Dictionary of logits for stop and z/y/x heads.
        """
        global_context, seq_context = self._prepare_context(batch)
        vertex_tokens = batch["vertex_tokens"]
        vertices_zyx = batch["vertices_zyx"]
        z_teacher = F.pad(vertices_zyx[..., 0], [0, 1], value=0)
        y_teacher = F.pad(vertices_zyx[..., 1], [0, 1], value=0)
        hidden = self._decode_hidden(
            vertex_tokens[:, :-1],
            vertices_zyx,
            global_context_embedding=global_context,
            sequential_context_embedding=seq_context,
        )
        stop_logits, z_logits, y_logits, x_logits = self._compute_heads(
            hidden,
            z_teacher=z_teacher,
            y_teacher=y_teacher,
        )
        return {
            "stop_logits": stop_logits,
            "z_logits": z_logits,
            "y_logits": y_logits,
            "x_logits": x_logits,
        }

    def training_step(self, vertex_model_batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.float32:
        """Pytorch Lightning training step method

        Args:
            vertex_model_batch: A dictionary that contains the flat vertices
            batch_idx: Which batch are we processing

        Returns:
            vertex_loss: NLL loss for estimated categorical distribution
        """
        logits = self(vertex_model_batch)
        token_targets = vertex_model_batch["vertex_tokens"]
        token_mask = vertex_model_batch["vertex_tokens_mask"].to(torch.float32)
        coord_targets = vertex_model_batch["vertices_zyx"]
        coord_mask = ((token_targets[:, :-1] == 1).to(torch.float32) * token_mask[:, :-1]).to(torch.float32)

        stop_loss = -torch.sum(
            torch.distributions.categorical.Categorical(logits=logits["stop_logits"]).log_prob(token_targets)
            * token_mask
        )
        z_loss = -torch.sum(
            torch.distributions.categorical.Categorical(logits=logits["z_logits"][:, :-1]).log_prob(
                coord_targets[..., 0]
            )
            * coord_mask
        )
        y_loss = -torch.sum(
            torch.distributions.categorical.Categorical(logits=logits["y_logits"][:, :-1]).log_prob(
                coord_targets[..., 1]
            )
            * coord_mask
        )
        x_loss = -torch.sum(
            torch.distributions.categorical.Categorical(logits=logits["x_logits"][:, :-1]).log_prob(
                coord_targets[..., 2]
            )
            * coord_mask
        )
        vertex_loss = stop_loss + z_loss + y_loss + x_loss
        self.log("train_loss", vertex_loss)
        self.log("train_stop_loss", stop_loss)
        self.log("train_z_loss", z_loss)
        self.log("train_y_loss", y_loss)
        self.log("train_x_loss", x_loss)
        return vertex_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Method to create optimizer and learning rate scheduler

        Returns:
            dict: A dictionary with optimizer and learning rate scheduler
        """
        vertex_model_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        vertex_model_scheduler = torch.optim.lr_scheduler.StepLR(
            vertex_model_optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {
            "optimizer": vertex_model_optimizer,
            "lr_scheduler": vertex_model_scheduler,
        }

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.float32:
        """Validation step for Pytorch Lightning

        Args:
            val_batch: dictionary which contains batch to run validation on
            batch_idx: Which batch we are processing

        Returns:
            vertex_loss: NLL loss for estimated categorical distribution
        """
        with torch.no_grad():
            logits = self(val_batch)
            token_targets = val_batch["vertex_tokens"]
            token_mask = val_batch["vertex_tokens_mask"].to(torch.float32)
            coord_targets = val_batch["vertices_zyx"]
            coord_mask = ((token_targets[:, :-1] == 1).to(torch.float32) * token_mask[:, :-1]).to(torch.float32)

            stop_loss = -torch.sum(
                torch.distributions.categorical.Categorical(logits=logits["stop_logits"]).log_prob(token_targets)
                * token_mask
            )
            z_loss = -torch.sum(
                torch.distributions.categorical.Categorical(logits=logits["z_logits"][:, :-1]).log_prob(
                    coord_targets[..., 0]
                )
                * coord_mask
            )
            y_loss = -torch.sum(
                torch.distributions.categorical.Categorical(logits=logits["y_logits"][:, :-1]).log_prob(
                    coord_targets[..., 1]
                )
                * coord_mask
            )
            x_loss = -torch.sum(
                torch.distributions.categorical.Categorical(logits=logits["x_logits"][:, :-1]).log_prob(
                    coord_targets[..., 2]
                )
                * coord_mask
            )
            vertex_loss = stop_loss + z_loss + y_loss + x_loss
        self.log("val_loss", vertex_loss)
        self.log("val_stop_loss", stop_loss)
        self.log("val_z_loss", z_loss)
        self.log("val_y_loss", y_loss)
        self.log("val_x_loss", x_loss)
        return vertex_loss

    def sample(
        self,
        num_samples: int,
        max_sample_length: int = 50,
        context: Dict[str, torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        recenter_verts: bool = True,
        only_return_complete: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate vertices

        Args:
            num_samples: Number of samples to produce.
            context: A dictionary with the type of context to condition upon. This could be class labels or images or voxels.
            max_sample_length: Maximum length of sampled vertex samples. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            recenter_verts: If True, center vertex samples around origin. This should be used if model is trained using shift augmentations.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'completed': Boolean tensor of shape [num_samples]. If True then corresponding sample completed within max_sample_length.
                'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
                'num_vertices': Tensor indicating number of vertices for each example in padded vertex samples.
                'vertices_mask': Tensor of shape [num_samples, num_verts] that masks corresponding invalid elements in vertices.
        """
        global_context, seq_context = self._prepare_context(context)

        # limit context shape to number of samples desired
        if global_context is not None:
            num_samples = min(num_samples, global_context.shape[0])
            global_context = global_context[:num_samples]
            if seq_context is not None:
                seq_context = seq_context[:num_samples]
        elif seq_context is not None:
            num_samples = min(num_samples, seq_context.shape[0])
            seq_context = seq_context[:num_samples]

        def _loop_body(
            i: int,
            token_samples: torch.Tensor,
            vertex_samples_zyx: torch.Tensor,
            completed: torch.Tensor,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                token_samples: tensor of shape [num_samples, i].
                vertex_samples_zyx: tensor of shape [num_samples, i, 3].
                completed: tensor of shape [num_samples].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}.
                       Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                token_samples: tensor of shape [num_samples, i + 1].
                vertex_samples_zyx: tensor of shape [num_samples, i + 1, 3].
                completed: updated completion tensor.
            """
            hidden = self._decode_hidden(
                token_samples,
                vertex_samples_zyx,
                global_context_embedding=global_context,
                sequential_context_embedding=seq_context,
                cache=cache,
            )
            curr_hidden = hidden[:, -1:, :]
            stop_logits = self.stop_head(curr_hidden) / temperature
            stop_logits = top_k_logits(stop_logits, top_k)
            stop_logits = top_p_logits(stop_logits, top_p)
            next_token = torch.distributions.categorical.Categorical(logits=stop_logits).sample().to(torch.int32)
            next_token = torch.where(completed[:, None], torch.zeros_like(next_token), next_token)

            z_logits = self.z_head(curr_hidden) / temperature
            z_logits = top_k_logits(z_logits, top_k)
            z_logits = top_p_logits(z_logits, top_p)
            next_z = torch.distributions.categorical.Categorical(logits=z_logits).sample().to(torch.int32)

            y_hidden = curr_hidden + self.coord0_embedder(next_z.to(torch.int64))
            y_logits = self.y_head(y_hidden) / temperature
            y_logits = top_k_logits(y_logits, top_k)
            y_logits = top_p_logits(y_logits, top_p)
            next_y = torch.distributions.categorical.Categorical(logits=y_logits).sample().to(torch.int32)

            x_hidden = y_hidden + self.coord1_embedder(next_y.to(torch.int64))
            x_logits = self.x_head(x_hidden) / temperature
            x_logits = top_k_logits(x_logits, top_k)
            x_logits = top_p_logits(x_logits, top_p)
            next_x = torch.distributions.categorical.Categorical(logits=x_logits).sample().to(torch.int32)

            next_vertex_zyx = torch.cat([next_z, next_y, next_x], dim=-1)
            next_vertex_zyx = next_vertex_zyx * next_token.to(torch.int32)

            token_samples = torch.cat([token_samples, next_token], dim=1)
            vertex_samples_zyx = torch.cat([vertex_samples_zyx, next_vertex_zyx[:, None, :]], dim=1)
            completed = completed | (next_token[:, 0] == 0)
            return i + 1, token_samples, vertex_samples_zyx, completed

        def _stopping_cond(completed: torch.Tensor) -> bool:
            """
            Stopping condition for sampling while-loop. Looking for stop token (represented by 0)
            Args:
                completed: tensor of shape [num_samples].
            Returns:
                token_found: Boolean that represents if stop token has been found.
            """
            return torch.any(~completed)

        token_samples = torch.zeros([num_samples, 0], dtype=torch.int32, device=self.device)
        vertex_samples_zyx = torch.zeros([num_samples, 0, 3], dtype=torch.int32, device=self.device)
        completed = torch.zeros([num_samples], dtype=torch.bool, device=self.device)
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_num_input_verts
        j = 0
        while _stopping_cond(completed) and j < max_sample_length + 1:
            j, token_samples, vertex_samples_zyx, completed = _loop_body(
                j, token_samples, vertex_samples_zyx, completed, cache
            )

        completed_samples_boolean = token_samples == 0
        stop_index_completed = torch.argmax(completed_samples_boolean.to(torch.int32), dim=-1).to(torch.int32)
        stop_index_incomplete = max_sample_length * torch.ones_like(stop_index_completed)
        stop_index = torch.where(
            completed, stop_index_completed, stop_index_incomplete
        )
        num_vertices = stop_index

        max_vertices_sampled = int(torch.max(num_vertices).item()) if num_vertices.numel() > 0 else 0
        samples_zyx = vertex_samples_zyx[:, :max_vertices_sampled]
        verts_dequantized = dequantize_verts(samples_zyx, self.quantization_bits)
        vertices = verts_dequantized
        vertices = torch.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
        )  # Converts from z-y-x to x-y-z.

        # Pad samples such that samples of different lengths can be concatenated
        pad_size = max_sample_length - vertices.shape[1]
        vertices = F.pad(vertices, [0, 0, 0, pad_size, 0, 0])

        vertices_mask = (torch.arange(max_sample_length)[None] < num_vertices[:, None]).to(
            torch.float32
        )  # Provides a mask of which vertices to zero out as they were produced after stop token for that batch ended

        if recenter_verts:
            vert_max, _ = torch.max(vertices - 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True)
            vert_min, _ = torch.min(vertices + 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True)
            vert_centers = 0.5 * (vert_max + vert_min)
            vertices = vertices - vert_centers

        vertices = vertices * vertices_mask[..., None]  # Zeros out vertices produced after stop token

        if only_return_complete:
            vertices = vertices[completed]
            num_vertices = num_vertices[completed]
            vertices_mask = vertices_mask[completed]
            completed = completed[completed]

        outputs = {
            "completed": completed,
            "vertices": vertices,
            "num_vertices": num_vertices,
            "vertices_mask": vertices_mask.to(torch.int32),
        }
        return outputs


class ImageToVertexModel(VertexModel):
    def __init__(
        self,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        use_discrete_embeddings: bool = True,
        max_num_input_verts: int = 2500,
        learning_rate: float = 3e-4,
        step_size: int = 5000,
        gamma: float = 0.9995,
    ) -> None:
        """Initializes the resnet module along with an embedder

        Args:
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            use_discrete_embeddings: Whether or not we're working with quantized vertices
            max_num_input_verts: Maximum number of vertices. Used for learned position embeddings.
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
        """
        super(ImageToVertexModel, self).__init__(
            decoder_config=decoder_config,
            quantization_bits=quantization_bits,
            max_num_input_verts=max_num_input_verts,
            use_discrete_embeddings=use_discrete_embeddings,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
        )
        self.res_net = PolygenResnet()
        for param in self.res_net.parameters():
            param.requires_grad = False
        self.embedder = nn.Linear(2, self.embedding_dim)

    def _prepare_context(self, context: Dict[str, torch.Tensor]) -> Tuple[None, torch.Tensor]:
        """Creates image embeddings using resnet and flattened image

        Args:
            context: A dictionary that contains an image

        Returns:
            sequential_context_embeddings: Processed image embeddings
        """
        image_embeddings = self.res_net(context["image"] - 0.5)
        image_embeddings = image_embeddings.permute(0, 2, 3, 1)
        processed_image_resolution = image_embeddings.shape[1:3]
        x = torch.linspace(-1, 1, processed_image_resolution[0], device=self.device)
        y = torch.linspace(-1, 1, processed_image_resolution[1], device=self.device)
        image_coords = torch.stack(torch.meshgrid(x, y), dim=-1)
        image_coord_embeddings = self.embedder(image_coords)
        image_embeddings = image_embeddings + image_coord_embeddings[None]
        batch_size = image_embeddings.shape[0]
        sequential_context_embeddings = torch.reshape(image_embeddings, [batch_size, -1, self.embedding_dim])

        return None, sequential_context_embeddings

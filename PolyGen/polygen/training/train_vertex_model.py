import pdb

import torch
import pytorch_lightning as pl
import argparse

import hydra
from hydra.utils import instantiate

from polygen.polygen_config import VertexModelConfig


def main(config_name: str) -> None:
    with hydra.initialize_config_module(config_module="polygen.config"):
        cfg = hydra.compose(config_name=config_name)
        vertex_model_config = instantiate(cfg.VertexModelConfig)

    vertex_data_module = vertex_model_config.vertex_data_module
    vertex_model = vertex_model_config.vertex_model

    training_steps = vertex_model_config.training_steps
    batch_size = vertex_model_config.batch_size
    dataset_length = len(vertex_data_module.shapenet_dataset)
    num_epochs = training_steps * batch_size // (dataset_length)

    trainer = pl.Trainer(
        accelerator=vertex_model_config.accelerator,
        gpus=vertex_model_config.gpu_ids if vertex_model_config.gpu_ids is not None else vertex_model_config.num_gpus,
        max_epochs=num_epochs,
    )
    trainer.fit(model=vertex_model, datamodule=vertex_data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="point_cloud_model_config.yaml",
        help="Hydra config file under polygen/config.",
    )
    args = parser.parse_args()
    main(config_name=args.config_name)

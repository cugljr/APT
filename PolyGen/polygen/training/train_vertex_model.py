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
    if dataset_length <= 0:
        data_dir = getattr(vertex_data_module, "data_dir", "<unknown>")
        raise RuntimeError(
            "Dataset is empty (len(dataset) == 0). "
            f"Check your config `{config_name}` and dataset_path/data_dir. "
            f"Resolved data_dir={data_dir!r}. "
            "For point-cloud mode, the directory must contain paired "
            "`meshes/*.obj` and `pointclouds/*.xyz` with matching filename stems."
        )

    num_epochs = training_steps * batch_size // dataset_length
    num_epochs = max(1, int(num_epochs))

    # PyTorch Lightning >= 2.0 uses `devices` instead of `gpus`.
    cfg_accel = getattr(vertex_model_config, "accelerator", "auto")
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    strategy = "ddp" if str(cfg_accel).startswith("ddp") else "auto"

    if getattr(vertex_model_config, "gpu_ids", None) is not None:
        devices = vertex_model_config.gpu_ids
    else:
        devices = vertex_model_config.num_gpus if use_gpu else 1

    logger = None
    if getattr(vertex_model_config, "use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=getattr(vertex_model_config, "wandb_project", "polygen"),
            name=getattr(vertex_model_config, "wandb_run_name", None),
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=num_epochs,
        logger=logger,
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

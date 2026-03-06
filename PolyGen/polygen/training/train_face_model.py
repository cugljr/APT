import torch
import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate

from polygen.polygen_config import FaceModelConfig


def main() -> None:
    with hydra.initialize_config_module(config_module="polygen.config"):
        cfg = hydra.compose(config_name="face_model_config_1231.yaml")
        face_model_config = instantiate(cfg.FaceModelConfig)

    face_data_module = face_model_config.face_data_module
    face_model = face_model_config.face_model

    training_steps = face_model_config.training_steps
    batch_size = face_model_config.batch_size
    dataset_length = len(face_data_module.shapenet_dataset)
    num_epochs = training_steps * batch_size // (dataset_length)

    # PyTorch Lightning >= 2.0 uses `devices` instead of `gpus`.
    cfg_accel = getattr(face_model_config, "accelerator", "auto")
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    strategy = "ddp" if str(cfg_accel).startswith("ddp") else "auto"
    devices = face_model_config.num_gpus if use_gpu else 1
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, strategy=strategy, max_epochs=num_epochs)
    trainer.fit(face_model, face_data_module)


if __name__ == "__main__":
    main()

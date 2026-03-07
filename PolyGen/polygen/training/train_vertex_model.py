import csv
import os

import torch
import pytorch_lightning as pl
import argparse

import hydra
from hydra.utils import instantiate

from polygen.polygen_config import VertexModelConfig


class ValidationResultsCallback(pl.Callback):
    """每个 epoch 将验证指标追加到 CSV 文件。"""

    def __init__(self, save_dir: str = "lightning_logs"):
        self.save_dir = save_dir
        self.results_file = os.path.join(save_dir, "validation_results.csv")
        self._header_written = False

    def _write_header(self, val_keys: list):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + val_keys)
        self._header_written = True

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        val_keys = [k for k in sorted(metrics) if k.startswith("val_")]
        if not val_keys:
            return
        if not self._header_written:
            self._write_header(val_keys)
        def _to_scalar(v):
            if isinstance(v, torch.Tensor):
                return v.item()
            return v

        epoch = _to_scalar(metrics.get("epoch", trainer.current_epoch))
        row = [epoch] + [_to_scalar(metrics.get(k, "")) for k in val_keys]
        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


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

    ckpt_dir = "lightning_logs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 验证 loss 最低的 1 个权重
    best_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-val_loss={val_loss:.2f}-epoch={epoch:02d}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )
    # 2. 最新的 5 个权重（按 epoch 从大到小保留）
    latest_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="latest-epoch={epoch:02d}",
        monitor="epoch",
        mode="max",
        save_top_k=5,
        every_n_epochs=1,
    )
    # 3. 每个 epoch 保存验证结果到 CSV
    val_results_cb = ValidationResultsCallback(save_dir="lightning_logs")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[best_ckpt, latest_ckpt, val_results_cb],
        default_root_dir="lightning_logs",
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

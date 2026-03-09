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


class LatestKCheckpointsCallback(pl.Callback):
    """每个 epoch 保存一个 latest checkpoint，并只保留最近 K 个。"""

    def __init__(self, ckpt_dir: str, keep_last_k: int = 5) -> None:
        self.ckpt_dir = ckpt_dir
        self.keep_last_k = keep_last_k
        os.makedirs(self.ckpt_dir, exist_ok=True)

    @staticmethod
    def _parse_epoch_from_name(filename: str) -> int:
        # latest-epoch=0001.ckpt
        try:
            stem = os.path.splitext(os.path.basename(filename))[0]
            prefix = "latest-epoch="
            if not stem.startswith(prefix):
                return -1
            return int(stem[len(prefix) :])
        except Exception:
            return -1

    def _prune(self) -> None:
        files = []
        for name in os.listdir(self.ckpt_dir):
            if name.startswith("latest-epoch=") and name.endswith(".ckpt"):
                epoch = self._parse_epoch_from_name(name)
                if epoch >= 0:
                    files.append((epoch, os.path.join(self.ckpt_dir, name)))
        files.sort(key=lambda x: x[0])
        if len(files) <= self.keep_last_k:
            return
        for _, path in files[: -self.keep_last_k]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # 只在 rank0 保存/删除，避免 DDP 冲突
        if not trainer.is_global_zero:
            return
        epoch = int(trainer.current_epoch)
        ckpt_path = os.path.join(self.ckpt_dir, f"latest-epoch={epoch:04d}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        self._prune()


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

    # 本地日志：每次 fit 自动创建 lightning_logs/<name>/version_x/
    csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs", name="vertex_pointcloud")
    log_dir = csv_logger.log_dir
    ckpt_dir = os.path.join(log_dir, "checkpoints")

    # 1) 验证 loss 最低的 1 个权重
    best_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-epoch={epoch:04d}-val_loss_mean={val_loss_mean:.4f}",
        monitor="val_loss_mean",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    # 2) 最新的 5 个权重（每个 epoch 存一份，然后裁剪只保留最近 5 个）
    latest_k_ckpt = LatestKCheckpointsCallback(ckpt_dir=ckpt_dir, keep_last_k=5)
    # 3) 每个 epoch 保存验证结果到 CSV（写到当前 run 目录）
    val_results_cb = ValidationResultsCallback(save_dir=log_dir)

    # logger：本地 CSV + （可选）wandb
    if logger is None:
        loggers = csv_logger
    else:
        loggers = [csv_logger, logger]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=num_epochs,
        logger=loggers,
        callbacks=[best_ckpt, latest_k_ckpt, val_results_cb],
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

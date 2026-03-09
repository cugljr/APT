import argparse
import csv
import os
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate

from polygen.polygen_config import VertexModelConfig
from polygen.utils.data_utils import dequantize_verts


def _load_model_from_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _gt_vertices_xyz(batch: Dict[str, torch.Tensor], sample_idx: int, quantization_bits: int) -> torch.Tensor:
    token_mask = batch["vertex_tokens"][sample_idx, :-1] == 1
    verts_zyx = batch["vertices_zyx"][sample_idx][token_mask]
    verts_xyz = dequantize_verts(verts_zyx, quantization_bits)
    verts_xyz = torch.stack([verts_xyz[..., 2], verts_xyz[..., 1], verts_xyz[..., 0]], dim=-1)
    return verts_xyz.to(torch.float32)


def _chamfer(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return float("inf")
    d = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    return (torch.min(d, dim=1).values.mean() + torch.min(d, dim=0).values.mean()).item()


def _save_xyz(points: torch.Tensor, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in points.detach().cpu().tolist():
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def _plot_triplet(input_pc: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, title: str, out_path: str) -> None:
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    data = [input_pc, gt, pred]
    names = ["Input Point Cloud", "GT Vertices", "Pred Vertices"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for ax, pts, name, color in zip(axes, data, names, colors):
        if pts.shape[0] > 0:
            p = pts.detach().cpu().numpy()
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, c=color, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=35)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual validation script for PolyGen vertex model.")
    parser.add_argument("--config_name", type=str, default="point_cloud_model_config.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to lightning checkpoint (.ckpt).")
    parser.add_argument("--out_dir", type=str, default="vertex_vis")
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--num_samples_per_batch", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    with hydra.initialize_config_module(config_module="polygen.config", version_base=None):
        cfg = hydra.compose(config_name=args.config_name)
        vertex_model_config: VertexModelConfig = instantiate(cfg.VertexModelConfig)

    datamodule = vertex_model_config.vertex_data_module
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()

    model = _load_model_from_checkpoint(vertex_model_config.vertex_model, args.ckpt_path, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_rows: List[Dict[str, float]] = []
    global_sample_idx = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_batches:
            break

        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        batch_size = batch["vertex_tokens"].shape[0]
        run_n = min(batch_size, args.num_samples_per_batch)

        context = {"point_cloud": batch["point_cloud"][:run_n]}
        with torch.no_grad():
            pred = model.sample(
                num_samples=run_n,
                context=context,
                max_sample_length=model.max_num_input_verts,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                recenter_verts=False,
                only_return_complete=False,
            )

        for i in range(run_n):
            sample_dir = os.path.join(args.out_dir, f"sample_{global_sample_idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            input_pc = batch["point_cloud"][i].detach().cpu()
            gt_xyz = _gt_vertices_xyz(batch, i, quantization_bits=model.quantization_bits).detach().cpu()
            pred_mask = pred["vertices_mask"][i] > 0
            pred_xyz = pred["vertices"][i][pred_mask].detach().cpu()

            chamfer = _chamfer(pred_xyz, gt_xyz)
            v_err = abs(int(pred_xyz.shape[0]) - int(gt_xyz.shape[0]))

            _save_xyz(input_pc, os.path.join(sample_dir, "input_point_cloud.xyz"))
            _save_xyz(gt_xyz, os.path.join(sample_dir, "gt_vertices.xyz"))
            _save_xyz(pred_xyz, os.path.join(sample_dir, "pred_vertices.xyz"))
            _plot_triplet(
                input_pc=input_pc,
                gt=gt_xyz,
                pred=pred_xyz,
                title=f"sample={global_sample_idx} chamfer={chamfer:.6f} |v_pred-v_gt|={v_err}",
                out_path=os.path.join(sample_dir, "compare.png"),
            )

            metrics_rows.append(
                {
                    "sample_id": global_sample_idx,
                    "batch_id": batch_idx,
                    "index_in_batch": i,
                    "chamfer_l2": chamfer,
                    "n_pred_vertices": int(pred_xyz.shape[0]),
                    "n_gt_vertices": int(gt_xyz.shape[0]),
                    "vertex_count_abs_error": v_err,
                    "completed": bool(pred["completed"][i].item()),
                }
            )
            global_sample_idx += 1

    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "batch_id",
                "index_in_batch",
                "chamfer_l2",
                "n_pred_vertices",
                "n_gt_vertices",
                "vertex_count_abs_error",
                "completed",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    if len(metrics_rows) > 0:
        mean_chamfer = sum(r["chamfer_l2"] for r in metrics_rows if r["chamfer_l2"] != float("inf")) / max(
            1, sum(1 for r in metrics_rows if r["chamfer_l2"] != float("inf"))
        )
        mean_v_err = sum(r["vertex_count_abs_error"] for r in metrics_rows) / len(metrics_rows)
        print(f"Saved visualization to: {args.out_dir}")
        print(f"Saved metrics to: {metrics_csv}")
        print(f"Mean Chamfer-L2: {mean_chamfer:.6f}")
        print(f"Mean |V_pred - V_gt|: {mean_v_err:.3f}")
    else:
        print("No samples processed. Check num_batches/num_samples_per_batch and validation dataloader.")


if __name__ == "__main__":
    main()


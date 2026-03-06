import argparse
import glob
import os
from typing import Dict, List, Tuple

import torch
from pytorch3d.io import load_obj


def _sample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points.to(torch.float32)
    idx = torch.randperm(points.shape[0])[:max_points]
    return points[idx].to(torch.float32)


def compute_point_set_distances(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    max_points: int = 20000,
) -> Dict[str, float]:
    """Compute Chamfer and Hausdorff distances between two point sets."""
    pred = _sample_points(pred_points, max_points)
    gt = _sample_points(gt_points, max_points)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"chamfer_l2": float("inf"), "hausdorff_l2": float("inf")}

    dmat = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    min_pred_to_gt = torch.min(dmat, dim=1).values
    min_gt_to_pred = torch.min(dmat, dim=0).values

    chamfer_l2 = (torch.mean(min_pred_to_gt) + torch.mean(min_gt_to_pred)).item()
    hausdorff_l2 = torch.max(torch.max(min_pred_to_gt), torch.max(min_gt_to_pred)).item()
    return {"chamfer_l2": chamfer_l2, "hausdorff_l2": hausdorff_l2}


def _faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = []
    for face in faces.tolist():
        if len(face) != 3:
            continue
        a, b, c = face
        edges.append(sorted([a, b]))
        edges.append(sorted([b, c]))
        edges.append(sorted([c, a]))
    if len(edges) == 0:
        return torch.zeros([0, 2], dtype=torch.int64)
    return torch.tensor(edges, dtype=torch.int64)


def compute_mesh_quality_metrics(vertices: torch.Tensor, faces: torch.Tensor, degenerate_eps: float = 1e-12) -> Dict[str, float]:
    """Compute mesh quality metrics from triangles."""
    metrics = {
        "num_vertices": float(vertices.shape[0]),
        "num_faces": float(faces.shape[0]),
        "degenerate_face_ratio": 0.0,
        "non_manifold_edge_ratio": 0.0,
        "boundary_edge_ratio": 0.0,
    }
    if faces.shape[0] == 0:
        return metrics

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    area2 = torch.sum(cross * cross, dim=-1)
    degenerate = area2 <= degenerate_eps
    metrics["degenerate_face_ratio"] = torch.mean(degenerate.to(torch.float32)).item()

    edges = _faces_to_edges(faces)
    if edges.shape[0] == 0:
        return metrics
    unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
    non_manifold = counts > 2
    boundary = counts == 1
    metrics["non_manifold_edge_ratio"] = torch.mean(non_manifold.to(torch.float32)).item()
    metrics["boundary_edge_ratio"] = torch.mean(boundary.to(torch.float32)).item()
    metrics["num_unique_edges"] = float(unique_edges.shape[0])
    return metrics


def evaluate_obj_pair(pred_obj: str, gt_obj: str, max_points: int = 20000) -> Dict[str, float]:
    """Evaluate one predicted OBJ against one GT OBJ."""
    pred_vertices, pred_faces_obj, _ = load_obj(pred_obj)
    gt_vertices, gt_faces_obj, _ = load_obj(gt_obj)
    pred_faces = pred_faces_obj.verts_idx.to(torch.long)
    gt_faces = gt_faces_obj.verts_idx.to(torch.long)

    dist_metrics = compute_point_set_distances(pred_vertices, gt_vertices, max_points=max_points)
    quality_metrics = compute_mesh_quality_metrics(pred_vertices, pred_faces)
    gt_quality_metrics = compute_mesh_quality_metrics(gt_vertices, gt_faces)

    out = {
        "pred_obj": pred_obj,
        "gt_obj": gt_obj,
        **dist_metrics,
        **quality_metrics,
        "gt_num_vertices": gt_quality_metrics["num_vertices"],
        "gt_num_faces": gt_quality_metrics["num_faces"],
    }
    return out


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, (float, int))]
    out = {}
    for k in scalar_keys:
        vals = [float(r[k]) for r in rows]
        out[k] = sum(vals) / max(1, len(vals))
    return out


def _collect_pairs(pred_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    pred_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(pred_dir, "*.obj"))}
    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(gt_dir, "*.obj"))}
    shared = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    return [(pred_map[k], gt_map[k]) for k in shared]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mesh reconstruction metrics on OBJ pairs.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory of predicted OBJ meshes.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of GT OBJ meshes.")
    parser.add_argument("--max_points", type=int, default=20000, help="Max points used for Chamfer/Hausdorff.")
    args = parser.parse_args()

    pairs = _collect_pairs(args.pred_dir, args.gt_dir)
    if len(pairs) == 0:
        raise RuntimeError("No paired OBJ files found by matching filename stem.")

    rows = [evaluate_obj_pair(pred_obj=p, gt_obj=g, max_points=args.max_points) for p, g in pairs]
    summary = _mean_metrics(rows)

    print(f"Pairs evaluated: {len(rows)}")
    for k in sorted(summary.keys()):
        print(f"{k}: {summary[k]:.6f}")


if __name__ == "__main__":
    main()


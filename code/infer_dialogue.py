import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from base import config as cfg_loader
from dataset.seamless import get_seamless_dataloaders
from seq2seq_pretrain import SLMFT


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config_dualtalk.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    cfg = cfg_loader.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = cfg_loader.merge_cfg_from_list(cfg, args.opts)
    cfg.config_path = args.config
    return cfg


def _resolve_manifest_for_inference(cfg):
    test_manifest = getattr(cfg, "test_manifest_path", None)
    if test_manifest:
        return Path(test_manifest)
    val_manifest = getattr(cfg, "val_manifest_path", None)
    if val_manifest:
        return Path(val_manifest)
    manifest = getattr(cfg, "manifest_path", None)
    if manifest:
        return Path(manifest)
    raise ValueError("No manifest path configured for inference.")


def load_manifest_map(cfg):
    manifest_path = _resolve_manifest_for_inference(cfg)
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    item_map = {}
    for item in items:
        sample_id = item.get("sample_id")
        if sample_id:
            item_map[sample_id] = item
    if not item_map:
        raise ValueError(f"No sample_id found in manifest: {manifest_path}")
    return item_map


def _resolve_path(data_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return data_root / p


def _load_shape(shape_path, flame_path):
    if shape_path and Path(shape_path).exists():
        shape = np.load(shape_path).astype(np.float32).reshape(-1)
    else:
        data = np.load(flame_path)
        if "shape_params" in data:
            shape_params = np.asarray(data["shape_params"])
            if shape_params.ndim == 1:
                shape = shape_params.astype(np.float32)
            else:
                shape = shape_params.reshape(shape_params.shape[0], -1)[0].astype(np.float32)
        else:
            shape = np.zeros((100,), dtype=np.float32)
    if shape.shape[0] > 100:
        shape = shape[:100]
    elif shape.shape[0] < 100:
        shape = np.pad(shape, (0, 100 - shape.shape[0]), mode="constant")
    return shape.astype(np.float32)


def _build_mask(lengths, max_len, device):
    mask = torch.zeros((len(lengths), max_len), dtype=torch.bool, device=device)
    for idx, seq_len in enumerate(lengths):
        mask[idx, : int(seq_len)] = True
    return mask


def _load_flame_runner(cfg, device):
    flowtalker_root = Path(
        getattr(cfg, "flowtalker_root", "/home/caizhuoqiang/Code/experiment/FlowTalker")
    ).resolve()
    if not flowtalker_root.exists():
        raise FileNotFoundError(f"FlowTalker path does not exist: {flowtalker_root}")
    sys.path.insert(0, str(flowtalker_root))
    try:
        from flowtalker.models import FLAME, FLAMEConfig
        from flowtalker.utils.vertices import motion_to_vertices
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to import FLAME from FlowTalker ({flowtalker_root}): {e}") from e

    flame = FLAME(FLAMEConfig).to(device)
    flame.eval()
    return flame, motion_to_vertices


def _to_motion54(motion, target_dim=54):
    if motion.shape[1] == target_dim:
        return motion
    if motion.shape[1] > target_dim:
        return motion[:, :target_dim]
    pad = np.zeros((motion.shape[0], target_dim - motion.shape[1]), dtype=motion.dtype)
    return np.concatenate([motion, pad], axis=1)


def main():
    cfg = parse_cfg()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    speaker_cfg = getattr(cfg, "speaker_config", cfg.config_path)
    listener_cfg = getattr(cfg, "listener_config", cfg.config_path)
    model = SLMFT(config_speaker_pth=speaker_cfg, config_listener_pth=listener_cfg).to(device)

    ckpt_path = getattr(cfg, "finetune_ckpt", "best_vico_causal.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Finetune checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    dataset = get_seamless_dataloaders(cfg)
    loader = dataset.get("test", dataset["valid"])
    manifest_map = load_manifest_map(cfg)

    flame, motion_to_vertices = _load_flame_runner(cfg, device)
    flame_batch_size = int(getattr(cfg, "flame_batch_size", 512))
    audio_dim = int(getattr(cfg, "audio_dim", 768))
    target_fps = int(getattr(cfg, "fps", 30))
    target_steps = int(getattr(cfg, "inference_steps", 15))
    default_dataset_name = str(getattr(cfg, "dialogue_dataset_name", "seamless_mini"))
    output_root = Path(getattr(cfg, "dialogue_output_root", "results/dialogue"))
    data_root = Path(getattr(cfg, "data_root", "/home/caizhuoqiang/Data"))
    output_root.mkdir(parents=True, exist_ok=True)

    shape_cache = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Dialogue inference"):
            src, tgt, src_len, _, sample_ids = batch
            src = src.to(device)
            tgt = tgt.to(device)
            visual_dim = src.shape[2] - audio_dim
            src_s_v, src_s_a = torch.split(src, [visual_dim, audio_dim], dim=2)
            mask = _build_mask(src_len, src.shape[1], device=device)

            tic = time.perf_counter()
            _, _, pred_cont = model(src_s_v, tgt, src_s_a, mask, mode="val")
            infer_seconds = time.perf_counter() - tic

            for idx, sample_id in enumerate(sample_ids):
                if sample_id not in manifest_map:
                    raise KeyError(f"sample_id not found in manifest: {sample_id}")
                item = manifest_map[sample_id]
                seq_len = int(src_len[idx])
                pred_seq = pred_cont[idx][: max(seq_len - 1, 0), :].detach().cpu().numpy()
                init_frame = tgt[idx, 0, :].detach().cpu().numpy().reshape(1, -1)
                motion = np.concatenate([init_frame, pred_seq], axis=0).astype(np.float32)
                if motion.shape[0] > seq_len:
                    motion = motion[:seq_len]
                elif motion.shape[0] < seq_len:
                    if motion.shape[0] == 0:
                        motion = np.zeros((seq_len, init_frame.shape[1]), dtype=np.float32)
                    else:
                        pad = np.repeat(motion[-1:, :], seq_len - motion.shape[0], axis=0)
                        motion = np.concatenate([motion, pad], axis=0)
                motion = _to_motion54(motion, target_dim=54)

                listener_flame = item.get("listener_flame")
                if not listener_flame:
                    raise KeyError(
                        "listener_flame path is missing in manifest item. "
                        "Re-run preprocessing with updated seamless_preprocessing.py."
                    )
                listener_flame = _resolve_path(data_root, listener_flame)

                shape_key = item.get("listener_shape", "") or str(listener_flame)
                if shape_key in shape_cache:
                    shape = shape_cache[shape_key]
                else:
                    shape_path = item.get("listener_shape", "")
                    if shape_path:
                        shape_path = _resolve_path(data_root, shape_path)
                    shape = _load_shape(shape_path, listener_flame)
                    shape_cache[shape_key] = shape

                motion_t = torch.from_numpy(motion).unsqueeze(0).to(device)
                shape_t = torch.from_numpy(shape).unsqueeze(0).to(device)
                vertices = motion_to_vertices(
                    motion_t,
                    shape_t,
                    flame,
                    flame_batch_size=flame_batch_size,
                    use_neck_pose=True,
                )
                vertices = vertices.squeeze(0).detach().cpu().numpy().astype(np.float32)

                dataset_name = str(item.get("dataset_name", default_dataset_name))
                dialog_id = str(item["dialog_id"])
                target_partner = str(item["listener_id"])
                out_dir = output_root / dataset_name / dialog_id / target_partner
                out_dir.mkdir(parents=True, exist_ok=True)

                np.save(out_dir / "0.npy", vertices)
                timing = {
                    "total_seconds": float(infer_seconds),
                    "per_rep_seconds": [float(infer_seconds)],
                    "steps": int(target_steps),
                    "n_repetitions": 1,
                    "frames": int(vertices.shape[0]),
                    "audio_seconds": float(vertices.shape[0] / max(target_fps, 1)),
                    "target_fps": int(target_fps),
                }
                with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
                    json.dump(timing, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dual-talk dyadic dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/caizhuoqiang/Data",
        help="Root directory of the raw dataset",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="dataset_jsons/dualtalk_splits/train.json",
        help="Train split JSON path relative to data_root",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="dataset_jsons/dualtalk_splits/val.json",
        help="Validation split JSON path relative to data_root",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="dataset_jsons/dualtalk_splits/test.json",
        help="Test split JSON path relative to data_root",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Legacy single JSON path (relative to data_root). If provided and split JSONs are empty, it is treated as train split.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/dualtalk_processed",
        help="Output directory for processed files (relative to repo root)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="seamless_mini",
        help="Name recorded in manifest, also used by dialogue inference output",
    )
    parser.add_argument("--exp_dim", type=int, default=50)
    parser.add_argument("--jaw_dim", type=int, default=1)
    parser.add_argument("--neck_dim", type=int, default=3)
    parser.add_argument("--shape_dim", type=int, default=100)
    parser.add_argument("--audio_dim", type=int, default=768)
    parser.add_argument("--hubert_sr", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for HuBERT feature extraction",
    )
    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="Skip HuBERT extraction and write zeros for audio features",
    )
    return parser.parse_args()


def _normalize_field(arr: np.ndarray, target_dim: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] > target_dim:
        arr = arr[:, :target_dim]
    elif arr.shape[1] < target_dim:
        pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    return arr


def _resolve_path(root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return root / path


def _split_json_paths(args) -> Dict[str, str]:
    if args.json_path:
        return {"train": args.json_path}
    split_paths = {
        "train": args.train_json,
        "val": args.val_json,
        "test": args.test_json,
    }
    split_paths = {k: v for k, v in split_paths.items() if v}
    if split_paths:
        return split_paths
    raise ValueError("No valid JSON path provided. Please set split JSON paths or --json_path.")


def load_motion_and_shape(
    npz_path: Path,
    exp_dim: int,
    jaw_dim: int,
    neck_dim: int,
    shape_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    exp = _normalize_field(data["expression_params"], exp_dim).astype(np.float32)
    jaw = _normalize_field(data["jaw_params"], jaw_dim).astype(np.float32)
    neck = _normalize_field(data["pose_params"], neck_dim).astype(np.float32)
    motion = np.concatenate([exp, jaw, neck], axis=-1).astype(np.float32)

    shape_all = _normalize_field(data["shape_params"], max(shape_dim, 1)).astype(np.float32)
    shape_vec = shape_all[0, :shape_dim]
    if shape_vec.shape[0] < shape_dim:
        shape_vec = np.pad(shape_vec, (0, shape_dim - shape_vec.shape[0]), mode="constant")
    return motion, shape_vec.astype(np.float32)


def downsample_to_len(array: np.ndarray, target_len: int) -> np.ndarray:
    if array.shape[0] == target_len:
        return array
    if target_len <= 0:
        return array[:0]
    tensor = torch.from_numpy(array).unsqueeze(0).permute(0, 2, 1)  # 1,C,T
    tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
    tensor = tensor.permute(0, 2, 1).squeeze(0)
    return tensor.cpu().numpy()


def extract_hubert(
    model,
    wav_path: Path,
    target_len: int,
    device: str,
    target_sr: int,
    audio_dim: int,
) -> np.ndarray:
    if torchaudio is None:
        raise ImportError("torchaudio is required for audio processing. Install torchaudio or run with --skip_audio.")
    wav, sr = torchaudio.load(wav_path)
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.to(device)
    wav_len = torch.LongTensor([wav.shape[1]]).to(device)
    with torch.no_grad():
        hs, _ = model(wav, wav_len)
        feats = hs[-1].squeeze(0).cpu().numpy()
    feats = downsample_to_len(feats, target_len).astype(np.float32)
    if feats.shape[1] > audio_dim:
        feats = feats[:, :audio_dim]
    elif feats.shape[1] < audio_dim:
        pad = np.zeros((feats.shape[0], audio_dim - feats.shape[1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=1)
    return feats


def _build_hubert(args):
    if args.skip_audio:
        return None
    try:
        from s3prl.nn import S3PRLUpstream
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"s3prl is not available. Please install it or use --skip_audio. Error: {e}") from e
    hubert = S3PRLUpstream("hubert").to(args.device)
    hubert.eval()
    return hubert


def _process_split(
    split_name: str,
    json_path: Path,
    data_root: Path,
    out_root: Path,
    args,
    hubert,
) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    manifest = []
    split_out_root = out_root / split_name
    split_out_root.mkdir(parents=True, exist_ok=True)

    for dialog_id, partners in tqdm(meta.items(), desc=f"{split_name} dialogs"):
        if not isinstance(partners, dict) or len(partners) < 2:
            continue
        partner_ids = list(partners.keys())
        if len(partner_ids) != 2:
            continue

        cached = {}
        for pid in partner_ids:
            info = partners[pid]
            flame_path = _resolve_path(data_root, str(info["flame_coeff_save_path"]))
            audio_path = _resolve_path(data_root, str(info["audio_path"]))
            motion, shape = load_motion_and_shape(
                flame_path,
                exp_dim=args.exp_dim,
                jaw_dim=args.jaw_dim,
                neck_dim=args.neck_dim,
                shape_dim=args.shape_dim,
            )
            valid_frames = int(info.get("valid_frames_num", motion.shape[0]))
            valid_frames = min(valid_frames, motion.shape[0])
            motion = motion[:valid_frames]

            if args.skip_audio:
                audio_feats = np.zeros((motion.shape[0], args.audio_dim), dtype=np.float32)
            else:
                audio_feats = extract_hubert(
                    hubert,
                    audio_path,
                    motion.shape[0],
                    args.device,
                    args.hubert_sr,
                    args.audio_dim,
                )

            cached[pid] = {
                "motion": motion,
                "audio": audio_feats,
                "shape": shape,
                "flame_path": flame_path,
                "audio_path": audio_path,
                "state_path": _resolve_path(data_root, str(info["state_path"])) if info.get("state_path") else None,
            }

        T = min(
            cached[partner_ids[0]]["motion"].shape[0],
            cached[partner_ids[1]]["motion"].shape[0],
            cached[partner_ids[0]]["audio"].shape[0],
            cached[partner_ids[1]]["audio"].shape[0],
        )
        if T <= 1:
            continue

        out_dir = split_out_root / dialog_id
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = {}
        for pid in partner_ids:
            motion = cached[pid]["motion"][:T]
            audio = cached[pid]["audio"][:T]
            shape = cached[pid]["shape"]

            motion_out = out_dir / f"{pid}_motion54.npy"
            audio_out = out_dir / f"{pid}_audio_hubert.npy"
            shape_out = out_dir / f"{pid}_shape100.npy"
            np.save(motion_out, motion.astype(np.float32))
            np.save(audio_out, audio.astype(np.float32))
            np.save(shape_out, shape.astype(np.float32))
            saved[pid] = {
                "motion": motion_out.resolve(),
                "audio": audio_out.resolve(),
                "shape": shape_out.resolve(),
            }

        for speaker_id, listener_id in [(partner_ids[0], partner_ids[1]), (partner_ids[1], partner_ids[0])]:
            sample_id = f"{dialog_id}_{speaker_id}_to_{listener_id}"
            manifest.append(
                {
                    "dataset_name": args.dataset_name,
                    "split_name": split_name,
                    "dialog_id": dialog_id,
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "listener_id": listener_id,
                    "fps": args.fps,
                    "T": int(T),
                    "speaker_motion": str(saved[speaker_id]["motion"]),
                    "listener_motion": str(saved[listener_id]["motion"]),
                    "speaker_audio": str(saved[speaker_id]["audio"]),
                    "speaker_shape": str(saved[speaker_id]["shape"]),
                    "listener_shape": str(saved[listener_id]["shape"]),
                    "speaker_flame": str(cached[speaker_id]["flame_path"].resolve()),
                    "listener_flame": str(cached[listener_id]["flame_path"].resolve()),
                    "speaker_audio_path": str(cached[speaker_id]["audio_path"].resolve()),
                    "listener_audio_path": str(cached[listener_id]["audio_path"].resolve()),
                    "speaker_state_path": str(cached[speaker_id]["state_path"].resolve())
                    if cached[speaker_id]["state_path"] is not None
                    else "",
                    "listener_state_path": str(cached[listener_id]["state_path"].resolve())
                    if cached[listener_id]["state_path"] is not None
                    else "",
                }
            )
    return manifest


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_jsons = _split_json_paths(args)
    hubert = _build_hubert(args)

    all_counts = {}
    for split_name, json_rel in split_jsons.items():
        json_path = _resolve_path(data_root, json_rel)
        manifest = _process_split(split_name, json_path, data_root, out_root, args, hubert)
        manifest_path = out_root / f"manifest_{split_name}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        all_counts[split_name] = len(manifest)
        print(f"[{split_name}] saved {len(manifest)} samples to {manifest_path}")

    if "train" in split_jsons and len(split_jsons) == 1:
        single_manifest = out_root / "manifest_train.json"
        compat_manifest = out_root / "manifest.json"
        with open(single_manifest, "r", encoding="utf-8") as src, open(
            compat_manifest, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())
        print(f"[compat] copied {single_manifest} -> {compat_manifest}")

    total = sum(all_counts.values())
    print(f"Finished preprocessing {total} directional samples across splits: {all_counts}")


if __name__ == "__main__":
    main()

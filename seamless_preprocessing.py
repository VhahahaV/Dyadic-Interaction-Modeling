import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Seamless dyadic dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/caizhuoqiang/Data",
        help="Root directory of the raw dataset",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="dataset_jsons/seamless_mini.json",
        help="Path to dataset json relative to data_root",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/seamless_processed",
        help="Output directory for processed files (relative to repo root)",
    )
    parser.add_argument("--exp_dim", type=int, default=50)
    parser.add_argument("--jaw_dim", type=int, default=1)
    parser.add_argument("--audio_sr", type=int, default=16000)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for any shuffling (if needed)",
    )
    return parser.parse_args()


def load_motion(npz_path: Path, exp_dim: int, jaw_dim: int) -> np.ndarray:
    npz = np.load(npz_path)
    exp = npz["expression_params"][:, 0, :exp_dim].astype(np.float32)
    jaw = npz["jaw_params"][:, 0, :jaw_dim].astype(np.float32)
    return np.concatenate([exp, jaw], axis=-1)


def downsample_to_len(array: np.ndarray, target_len: int) -> np.ndarray:
    if array.shape[0] == target_len:
        return array
    if target_len <= 0:
        return array[:0]
    tensor = torch.from_numpy(array).unsqueeze(0).permute(0, 2, 1)  # 1,C,T
    tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=True)
    tensor = tensor.permute(0, 2, 1).squeeze(0)
    return tensor.cpu().numpy()


def extract_hubert(model, wav_path: Path, target_len: int, device: str, target_sr: int) -> np.ndarray:
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
    feats = downsample_to_len(feats, target_len)
    return feats.astype(np.float32)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    json_path = data_root / args.json_path
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        meta = json.load(f)

    hubert = None
    if not args.skip_audio:
        try:
            from s3prl.nn import S3PRLUpstream
            hubert = S3PRLUpstream("hubert").to(args.device)
            hubert.eval()
        except ImportError as e:
            raise ImportError(f"s3prl is not available. Please install it or use --skip_audio. Error: {e}")

    manifest = []

    for dialog_id, partners in tqdm(meta.items(), desc="Dialogs"):
        if len(partners) != 2:
            continue
        partner_ids = list(partners.keys())
        cached = {}

        # load and cache per-partner motion/audio
        for pid in partner_ids:
            info = partners[pid]
            motion_path = data_root / info["flame_coeff_save_path"]
            motion = load_motion(motion_path, args.exp_dim, args.jaw_dim)
            valid_frames = int(info.get("valid_frames_num", motion.shape[0]))
            motion = motion[:valid_frames]

            if args.skip_audio:
                audio_feats = np.zeros((motion.shape[0], 768), dtype=np.float32)
            else:
                audio_path = data_root / info["audio_path"]
                audio_feats = extract_hubert(
                    hubert, audio_path, motion.shape[0], args.device, args.audio_sr
                )
            cached[pid] = {"motion": motion, "audio": audio_feats}

        # align both partners to the same length
        T = min(
            cached[partner_ids[0]]["motion"].shape[0],
            cached[partner_ids[1]]["motion"].shape[0],
            cached[partner_ids[0]]["audio"].shape[0],
            cached[partner_ids[1]]["audio"].shape[0],
        )
        if T <= 0:
            continue

        out_dir = out_root / dialog_id
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = {}
        for pid in partner_ids:
            motion = cached[pid]["motion"][:T]
            audio = cached[pid]["audio"][:T]

            motion_out = out_dir / f"{pid}_motion51.npy"
            audio_out = out_dir / f"{pid}_audio_hubert.npy"
            np.save(motion_out, motion)
            np.save(audio_out, audio)
            saved[pid] = (motion_out.resolve(), audio_out.resolve())

        # two directional samples per dialog
        for speaker_id, listener_id in [(partner_ids[0], partner_ids[1]), (partner_ids[1], partner_ids[0])]:
            sample_id = f"{dialog_id}_{speaker_id}_to_{listener_id}"
            manifest.append(
                {
                    "dialog_id": dialog_id,
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "listener_id": listener_id,
                    "fps": args.fps,
                    "T": T,
                    "speaker_motion": str(saved[speaker_id][0]),
                    "listener_motion": str(saved[listener_id][0]),
                    "speaker_audio": str(saved[speaker_id][1]),
                }
            )

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} samples to {manifest_path}")


if __name__ == "__main__":
    main()

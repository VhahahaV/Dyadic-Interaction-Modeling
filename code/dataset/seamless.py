import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils import data


def _split_by_dialog(items, val_ratio=0.05, seed=42):
    dialog_to_items = {}
    for it in items:
        dialog_to_items.setdefault(it["dialog_id"], []).append(it)
    dialog_ids = list(dialog_to_items.keys())
    rng = random.Random(seed)
    rng.shuffle(dialog_ids)
    num_val = max(1, int(len(dialog_ids) * val_ratio)) if dialog_ids else 0
    val_ids = set(dialog_ids[:num_val])
    train_items, val_items = [], []
    for did in dialog_ids:
        if did in val_ids:
            val_items.extend(dialog_to_items[did])
        else:
            train_items.extend(dialog_to_items[did])
    return train_items, val_items


def _build_id_map(items):
    ids = set()
    for it in items:
        ids.add(it["speaker_id"])
        ids.add(it["listener_id"])
    return {sid: idx for idx, sid in enumerate(sorted(ids))}


class SeamlessDyadicDataset(data.Dataset):
    def __init__(self, items, window_frames=150, is_train=True, audio_dim=768, id_map=None):
        self.items = items
        self.window_frames = int(window_frames)
        self.is_train = is_train
        self.audio_dim = int(audio_dim)
        self.id_map = id_map or _build_id_map(items)

    def __len__(self):
        return len(self.items)

    def _pick_window(self, T):
        if not self.is_train or self.window_frames <= 0 or self.window_frames >= T:
            return 0, T
        start = random.randint(0, T - self.window_frames)
        return start, start + self.window_frames

    def __getitem__(self, idx):
        it = self.items[idx]
        T = int(it["T"])
        start, end = self._pick_window(T)

        speaker_motion = np.load(it["speaker_motion"], mmap_mode="r")[start:end]
        listener_motion = np.load(it["listener_motion"], mmap_mode="r")[start:end]

        if "speaker_audio" in it and it["speaker_audio"]:
            speaker_audio = np.load(it["speaker_audio"], mmap_mode="r")[start:end]
        else:
            speaker_audio = np.zeros((end - start, self.audio_dim), dtype=np.float32)

        combined = np.concatenate([speaker_motion, speaker_audio], axis=-1)

        speaker_id = self.id_map.get(it["speaker_id"], 0)
        listener_id = self.id_map.get(it["listener_id"], 0)
        sentiment = 0

        return (
            torch.from_numpy(combined).float(),
            torch.from_numpy(listener_motion).float(),
            it["sample_id"],
            speaker_id,
            listener_id,
            sentiment,
        )


class SeamlessVQDataset(data.Dataset):
    def __init__(self, items, role="listener", window_frames=150, is_train=True):
        if role not in ("speaker", "listener"):
            raise ValueError(f"role must be speaker or listener, got {role}")
        self.items = items
        self.role = role
        self.window_frames = int(window_frames)
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def _pick_window(self, T):
        if not self.is_train or self.window_frames <= 0 or self.window_frames >= T:
            return 0, T
        start = random.randint(0, T - self.window_frames)
        return start, start + self.window_frames

    def __getitem__(self, idx):
        it = self.items[idx]
        T = int(it["T"])
        start, end = self._pick_window(T)
        motion_key = "speaker_motion" if self.role == "speaker" else "listener_motion"
        motion = np.load(it[motion_key], mmap_mode="r")[start:end]
        return torch.from_numpy(motion).float(), it["sample_id"]


def get_seamless_dataloaders(cfg):
    manifest_path = Path(cfg.manifest_path)
    with open(manifest_path, "r") as f:
        items = json.load(f)

    val_ratio = float(getattr(cfg, "val_ratio", 0.05))
    seed = int(getattr(cfg, "seed", 42))
    train_items, val_items = _split_by_dialog(items, val_ratio=val_ratio, seed=seed)

    window_frames = int(getattr(cfg, "window_frames", 150))
    audio_dim = int(getattr(cfg, "audio_dim", 768))

    id_map = _build_id_map(items)

    train_ds = SeamlessDyadicDataset(
        train_items,
        window_frames=window_frames,
        is_train=True,
        audio_dim=audio_dim,
        id_map=id_map,
    )
    val_ds = SeamlessDyadicDataset(
        val_items if val_items else train_items,
        window_frames=window_frames,
        is_train=True,  # Also use random windows for validation to save memory
        audio_dim=audio_dim,
        id_map=id_map,
    )

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=pad_collate,
    )
    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=pad_collate,
    )
    return {"train": train_loader, "valid": val_loader}


def get_seamless_vq_dataloaders(cfg):
    manifest_path = Path(cfg.manifest_path)
    with open(manifest_path, "r") as f:
        items = json.load(f)

    val_ratio = float(getattr(cfg, "val_ratio", 0.05))
    seed = int(getattr(cfg, "seed", 42))
    train_items, val_items = _split_by_dialog(items, val_ratio=val_ratio, seed=seed)

    window_frames = int(getattr(cfg, "window_frames", 150))
    role = getattr(cfg, "vq_role", "listener")

    train_ds = SeamlessVQDataset(
        train_items, role=role, window_frames=window_frames, is_train=True
    )
    val_ds = SeamlessVQDataset(
        val_items if val_items else train_items,
        role=role,
        window_frames=window_frames,
        is_train=False,
    )

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
    )
    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size_val if hasattr(cfg, "batch_size_val") else 1,
        shuffle=False,
        num_workers=cfg.workers,
    )
    return {"train": train_loader, "valid": val_loader}


def pad_collate(batch):
    (xx, yy, zz, speaker_ids, listener_ids, sentiment) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    zz_names = [z for z in zz]
    speaker_ids = torch.LongTensor(speaker_ids)
    listener_ids = torch.LongTensor(listener_ids)
    sentiment = torch.LongTensor(sentiment)
    return xx_pad, yy_pad, x_lens, (speaker_ids, listener_ids), zz_names

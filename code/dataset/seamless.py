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


def _load_manifest(path):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if not isinstance(data_list, list):
        raise ValueError(f"Manifest must be a list: {p}")
    return data_list


def _resolve_split_items(cfg):
    train_manifest_path = getattr(cfg, "train_manifest_path", None)
    val_manifest_path = getattr(cfg, "val_manifest_path", None)
    test_manifest_path = getattr(cfg, "test_manifest_path", None)

    use_split_manifests = any([train_manifest_path, val_manifest_path, test_manifest_path])
    if use_split_manifests:
        train_items = _load_manifest(train_manifest_path)
        val_items = _load_manifest(val_manifest_path)
        test_items = _load_manifest(test_manifest_path)

        if not train_items:
            if val_items:
                train_items = val_items
            elif test_items:
                train_items = test_items
            else:
                raise ValueError("No data found in split manifests.")
        if not val_items:
            val_items = train_items
        return train_items, val_items, test_items

    manifest_path = getattr(cfg, "manifest_path", None)
    if manifest_path is None:
        raise ValueError(
            "No seamless manifest path configured. Set manifest_path or split manifests."
        )
    items = _load_manifest(manifest_path)
    val_ratio = float(getattr(cfg, "val_ratio", 0.05))
    seed = int(getattr(cfg, "seed", 42))
    train_items, val_items = _split_by_dialog(items, val_ratio=val_ratio, seed=seed)
    if not val_items:
        val_items = train_items
    return train_items, val_items, []


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
    train_items, val_items, test_items = _resolve_split_items(cfg)
    window_frames = int(getattr(cfg, "window_frames", 150))
    audio_dim = int(getattr(cfg, "audio_dim", 768))
    workers = int(getattr(cfg, "workers", 0))
    batch_size = int(getattr(cfg, "batch_size", 1))
    batch_size_val = int(getattr(cfg, "batch_size_val", batch_size))
    batch_size_test = int(getattr(cfg, "batch_size_test", 1))
    val_random_window = bool(getattr(cfg, "val_random_window", True))

    id_map = _build_id_map(train_items + val_items + test_items)

    train_ds = SeamlessDyadicDataset(
        train_items,
        window_frames=window_frames,
        is_train=True,
        audio_dim=audio_dim,
        id_map=id_map,
    )
    val_ds = SeamlessDyadicDataset(
        val_items,
        window_frames=window_frames,
        is_train=val_random_window,
        audio_dim=audio_dim,
        id_map=id_map,
    )

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=pad_collate,
    )
    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=workers,
        collate_fn=pad_collate,
    )
    loaders = {"train": train_loader, "valid": val_loader}

    if test_items:
        test_ds = SeamlessDyadicDataset(
            test_items,
            window_frames=window_frames,
            is_train=False,
            audio_dim=audio_dim,
            id_map=id_map,
        )
        test_loader = data.DataLoader(
            dataset=test_ds,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            collate_fn=pad_collate,
        )
        loaders["test"] = test_loader

    return loaders


def get_seamless_vq_dataloaders(cfg):
    train_items, val_items, test_items = _resolve_split_items(cfg)
    window_frames = int(getattr(cfg, "window_frames", 150))
    role = getattr(cfg, "vq_role", "listener")
    workers = int(getattr(cfg, "workers", 0))
    batch_size = int(getattr(cfg, "batch_size", 1))
    batch_size_val = int(getattr(cfg, "batch_size_val", 1))
    batch_size_test = int(getattr(cfg, "batch_size_test", 1))

    train_ds = SeamlessVQDataset(
        train_items, role=role, window_frames=window_frames, is_train=True
    )
    val_ds = SeamlessVQDataset(
        val_items,
        role=role,
        window_frames=window_frames,
        is_train=False,
    )

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=workers,
    )
    loaders = {"train": train_loader, "valid": val_loader}

    if test_items:
        test_ds = SeamlessVQDataset(
            test_items,
            role=role,
            window_frames=window_frames,
            is_train=False,
        )
        test_loader = data.DataLoader(
            dataset=test_ds,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
        )
        loaders["test"] = test_loader

    return loaders


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

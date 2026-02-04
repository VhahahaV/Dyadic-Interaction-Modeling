import argparse
import os
import torch
import torch.nn as nn
from dataset.seamless import get_seamless_dataloaders
from dataset.data_loader import get_candor_dataloaders, get_vico_dataloaders
from dataset.l2l import get_lm_listener_dataloaders
from x_engine_pt import train_epoch, evaluate_finetune_epoch, evaluate_test_epoch
from seq2seq_pretrain import SLMFT, SpeakerSLMFT

from tqdm import tqdm
import numpy as np
from piq import FID
import torch.distributed as dist
import builtins
import pickle5 as pickle
from mymetrics import print_metrics, print_metrics_full
import pickle5 as pickle
from base import config as cfg_loader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_dist():
    env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }

    ngpus_per_node = torch.cuda.device_count()

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="nccl")
    crank = int(env_dict['RANK'])

    if(crank!=0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print(f'Init succesfully rank {crank}')
    return crank

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    cfg = cfg_loader.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = cfg_loader.merge_cfg_from_list(cfg, args.opts)
    cfg.config_path = args.config
    return cfg


def main():
    cfg = parse_cfg()
    # crank = initialize_dist()
    crank = 0
    device = torch.device("cuda:{}".format(crank))

    speaker_cfg = getattr(cfg, "speaker_config", cfg.config_path)
    listener_cfg = getattr(cfg, "listener_config", cfg.config_path)
    model = SLMFT(config_speaker_pth=speaker_cfg, config_listener_pth=listener_cfg).to(device)

    ckpt_path = getattr(cfg, "finetune_ckpt", "best_vico_causal.pt")
    if os.path.exists(ckpt_path):
        d = torch.load(ckpt_path)
        model.load_state_dict(d)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = None

    dataset_name = str(getattr(cfg, "dataset", "vico")).lower()
    if dataset_name == "seamless":
        dataset = get_seamless_dataloaders(cfg)
    elif dataset_name == "lm_listener":
        dataset = get_lm_listener_dataloaders(batch_size=cfg.batch_size)
    elif dataset_name == "candor":
        dataset = get_candor_dataloaders(batch_size=cfg.batch_size)
    else:
        dataset = get_vico_dataloaders(batch_size=cfg.batch_size)

    train_loader = dataset["train"]
    val_loader = dataset["valid"]

    y_true, y_pred, x, data_ids = evaluate_test_epoch(model, val_loader, device)
    print_metrics(y_true, y_pred, x)
    print_metrics_full(y_true, y_pred, x)

    # save predictions
    d = {"y_true": y_true, "y_pred": y_pred, "data_ids": data_ids}
    with open("l2l_listener_predictions.pkl", "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

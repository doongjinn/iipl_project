#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime

import torch

from config import get_cfg
from model.SOFS import SOFS
from tools.metrics_eval import eval_non_resize_loader_metrics
from tools.train import train as train_func
from utils import seed_everything
from utils.load_dataset import get_datasets


def _build_cfg(cfg_path: str, opts: list) -> "CfgNode":
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    if opts:
        cfg.merge_from_list(opts)

    # Mirror utils/parser_.py OUTPUT_DIR behavior.
    method = cfg.TRAIN.method
    cfg.OUTPUT_DIR = "_".join([cfg.OUTPUT_DIR, cfg.DATASET.name, method, str(cfg.RNG_SEED)])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "checkpoints"), exist_ok=True)
    return cfg


def _load_model(cfg, ckpt_path: str) -> torch.nn.Module:
    model = SOFS(cfg=cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.cuda(cfg.DEVICE if hasattr(cfg, "DEVICE") else 0)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_train", required=True)
    ap.add_argument("--counts", default="1,2,4,8,16")
    ap.add_argument("--test_count", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=None, help="Override TRAIN_SETUPS.epochs (optional)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=54)
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    counts = [int(x.strip()) for x in args.counts.split(",") if x.strip()]
    assert all(c > 0 for c in counts)

    rows = []
    for n_train in counts:
        # Train config (disable in-training test to keep runtime focused).
        base_out = f"./log_SWEEP_DUSAN_TRAINCOUNT_{n_train}"
        opts = [
            "NUM_GPUS", "1",
            "DEVICE", str(int(args.device.split(",")[0])),
            "RNG_SEED", str(args.seed),
            "OUTPUT_DIR", base_out,
            "DATASET.custom_train_count", str(n_train),
            "DATASET.custom_test_count", str(args.test_count),
            "TRAIN_SETUPS.TEST_SETUPS.test_state", "False",
        ]
        if args.epochs is not None:
            opts += ["TRAIN_SETUPS.epochs", str(args.epochs)]

        cfg = _build_cfg(args.cfg_train, opts)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        seed_everything(cfg.RNG_SEED)
        train_func(cfg=cfg)

        ckpt_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "last.pth")
        if not os.path.isfile(ckpt_path):
            # fallback to any .pth
            cands = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            if not cands:
                raise RuntimeError(f"No checkpoint found under {ckpt_dir}")
            ckpt_path = cands[0]

        # Build eval model once.
        model = _load_model(cfg, ckpt_path)

        # Evaluate held-out test split.
        cfg_test_eval = cfg.clone()
        cfg_test_eval.defrost()
        cfg_test_eval.TRAIN.enable = False
        cfg_test_eval.TEST.enable = True
        cfg_test_eval.freeze()

        test_eval_ds = get_datasets(cfg=cfg_test_eval, mode="test")[0]
        test_eval_loader = torch.utils.data.DataLoader(
            test_eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.TRAIN_SETUPS.num_workers,
            pin_memory=True,
        )
        m_test = eval_non_resize_loader_metrics(test_eval_loader, model, cfg_test_eval)

        rows.append({
            "train_count": n_train,
            "test_count": args.test_count,
            "ckpt_path": ckpt_path,
            "test_pixel_acc": m_test.pixel_acc,
            "test_mIoU": m_test.miou,
            "test_dice": m_test.dice,
        })

    out_csv = args.output_csv
    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"./sweep_metrics_dusan_{ts}.csv"
    out_csv = os.path.abspath(out_csv)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote: {out_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()



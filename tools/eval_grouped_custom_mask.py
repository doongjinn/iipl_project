#!/usr/bin/env python3
"""
Grouped evaluation for CUSTOM_MASK_ND-style datasets.

Outputs (per shot):
  - Overall metrics on the evaluation loader
  - Metrics grouped by type index (parsed from filename)
  - Metrics grouped by status (normal/abnormal)

Assumes batch_size=1.
"""

import os
import sys

# Ensure repo root is on PYTHONPATH when this script is executed directly as a file.
# (When running `python tools/xxx.py`, Python puts `tools/` on sys.path, not the repo root.)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from config import get_cfg
from model.SOFS import SOFS
from tools.metrics_eval import BinarySegMetrics, _accumulate_confusion, _finalize
from utils.common import network_output2original_result, produce_qualitative_result
from utils.load_dataset import get_datasets


def _get_scalar(x):
    # DataLoader collates ints into tensors; strings into list[str]
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def _new_conf():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


@torch.no_grad()
def eval_grouped(val_loader, model, cfg) -> Tuple[
    Dict[str, BinarySegMetrics],
    Dict[str, Dict[str, BinarySegMetrics]],
    Dict[str, Dict[str, BinarySegMetrics]],
]:
    """
    Returns:
      overall_by_split: split -> metrics, where split in {"all","train","test","unknown"}
      by_type_by_split: split -> (type_key -> metrics)
      by_status_by_split: split -> (status_key -> metrics)
    """
    model.eval()
    model_ref = model.module if hasattr(model, "module") else model
    if hasattr(model_ref, "backbone"):
        model_ref.backbone.eval()

    overall_by_split = {"all": _new_conf()}
    by_type_by_split = {"all": {}}
    by_status_by_split = {"all": {}}

    it = val_loader
    if hasattr(val_loader, "__len__"):
        it = tqdm(val_loader, desc="eval", total=len(val_loader))
    for _, data in enumerate(it):
        s_input = data["support_image"].cuda(non_blocking=True)
        s_mask = data["support_mask"].cuda(non_blocking=True)
        input = data["query_image"].cuda(non_blocking=True)
        target = data["query_mask"].cuda(non_blocking=True)

        query_original_shape = data["query_original_shape"][0]
        query_input_shape = data["query_input_shape"][0]
        query_crop_shape = data["query_crop_shape"][0]
        img_position_list = data["img_position_list"]

        # meta (for grouping)
        type_idx = _get_scalar(data.get("query_type_idx", -1))
        status = _get_scalar(data.get("query_status", "unknown"))
        split = _get_scalar(data.get("query_split", "unknown"))
        type_key = f"type{int(type_idx)}" if int(type_idx) >= 0 else "type_unknown"
        status_key = str(status)
        split_key = str(split)
        if split_key not in overall_by_split:
            overall_by_split[split_key] = _new_conf()
            by_type_by_split[split_key] = {}
            by_status_by_split[split_key] = {}

        assert input.shape[0] == 1, "evaluation assumes batch_size=1"

        original_heatmap = None
        if input.dim() == 4:
            output = model(s_x=s_input, s_y=s_mask, x=input)
            output_abs = output.max(1)[1][0]
            output_heatmap = output[:, 1, ...][0]
            original_output, original_heatmap = network_output2original_result(
                query_input_shape=query_input_shape,
                query_original_shape=query_original_shape,
                output_absolute_val=output_abs,
                output_heatmap=output_heatmap,
            )
        else:
            # patching / ND merge (same as eval_non_resize_loader_metrics)
            grid_size = input.shape[1]
            original_output = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_position = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_heatmap = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))

            multiple_ND = grid_size // cfg.TEST_SETUPS.ND_batch_size
            multiple_mod = grid_size % cfg.TEST_SETUPS.ND_batch_size
            iter_num = multiple_ND if multiple_mod == 0 else multiple_ND + 1

            output_total = []
            for pointer_idx in range(iter_num):
                init_val = pointer_idx * cfg.TEST_SETUPS.ND_batch_size
                _, gs, c, h, w = input.shape
                multiple_bs = cfg.TEST_SETUPS.ND_batch_size
                if multiple_ND == 0:
                    multiple_bs = gs
                if pointer_idx == multiple_ND:
                    multiple_bs = multiple_mod

                support_input_m = s_input.repeat(multiple_bs, 1, 1, 1, 1)
                support_mask_m = s_mask.repeat(multiple_bs, 1, 1, 1, 1)
                query_input_m = input[:, init_val : init_val + multiple_bs, ...].reshape(1 * multiple_bs, c, h, w)

                tmp_output = model(s_x=support_input_m, s_y=support_mask_m, x=query_input_m)
                output_total.append(tmp_output)

            output_total = torch.concat(output_total, dim=0)

            for query_idx in range(grid_size):
                top, down, left, right = [int(i) for i in img_position_list[query_idx]]
                output = output_total[query_idx].unsqueeze(0)
                output_abs = output.max(1)[1][0]
                output_heatmap = output[:, 1, ...][0]

                tuple_output_shape = tuple(torch.as_tensor(output_abs.shape).cpu().numpy())
                tuple_query_input_shape = tuple(query_input_shape[query_idx].cpu().numpy())
                tuple_query_crop_shape = tuple(query_crop_shape[query_idx].cpu().numpy())

                if not (tuple_output_shape == tuple_query_input_shape and tuple_query_input_shape == tuple_query_crop_shape):
                    output_abs, output_heatmap = network_output2original_result(
                        query_input_shape=query_input_shape[query_idx],
                        query_original_shape=query_crop_shape[query_idx],
                        output_absolute_val=output_abs,
                        output_heatmap=output_heatmap,
                    )

                original_output[query_idx, top:down, left:right] = output_abs.cpu()
                original_heatmap[query_idx, top:down, left:right] = output_heatmap.cpu()
                original_position[query_idx, top:down, left:right] = 1

            original_output = torch.sum(original_output, dim=0) / torch.sum(original_position, dim=0)
            original_heatmap = torch.sum(original_heatmap, dim=0) / torch.sum(original_position, dim=0)

        original_mask = target.squeeze(0).squeeze(0)[: int(query_original_shape[0]), : int(query_original_shape[1])]

        pred = original_output
        if pred.dtype != torch.long:
            pred = (pred > 0.5).to(torch.uint8)
        if original_mask.dtype != torch.uint8:
            original_mask = (original_mask > 0.5).to(torch.uint8)

        if pred.shape != original_mask.shape:
            h = min(pred.shape[-2], original_mask.shape[-2])
            w = min(pred.shape[-1], original_mask.shape[-1])
            pred = pred[:h, :w]
            original_mask = original_mask[:h, :w]

        tp, fp, fn, tn = _accumulate_confusion(pred.cpu(), original_mask.cpu())

        # Accumulate into both 'all' and the sample's split bucket.
        for sk in ("all", split_key):
            for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                overall_by_split[sk][k] += v

            if type_key not in by_type_by_split[sk]:
                by_type_by_split[sk][type_key] = _new_conf()
            by_type_by_split[sk][type_key]["tp"] += tp
            by_type_by_split[sk][type_key]["fp"] += fp
            by_type_by_split[sk][type_key]["fn"] += fn
            by_type_by_split[sk][type_key]["tn"] += tn

            if status_key not in by_status_by_split[sk]:
                by_status_by_split[sk][status_key] = _new_conf()
            by_status_by_split[sk][status_key]["tp"] += tp
            by_status_by_split[sk][status_key]["fp"] += fp
            by_status_by_split[sk][status_key]["fn"] += fn
            by_status_by_split[sk][status_key]["tn"] += tn

    overall_m = {k: _finalize(**v) for k, v in overall_by_split.items()}
    by_type_m = {sk: {k: _finalize(**v) for k, v in sorted(d.items(), key=lambda kv: kv[0])} for sk, d in by_type_by_split.items()}
    by_status_m = {sk: {k: _finalize(**v) for k, v in sorted(d.items(), key=lambda kv: kv[0])} for sk, d in by_status_by_split.items()}
    return overall_m, by_type_m, by_status_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Test config yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint (.pth)")
    ap.add_argument("--dino_ckpt", default=None, help="Override TRAIN.backbone_checkpoint (optional)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=54)
    ap.add_argument("--shots", default="1,2,4,8,16", help="Comma-separated shots")
    ap.add_argument("--out_prefix", default=None, help="Output prefix (default: ./dataset_final_metrics_<timestamp>)")
    ap.add_argument("--num_workers", type=int, default=None, help="Override DataLoader num_workers (optional)")
    ap.add_argument("--save_figures", action="store_true", help="Save visualization images (pred/gt/heatmap)")
    ap.add_argument("--figure_sample_prob", type=float, default=1.0, help="Probability to save each sample figure")
    ap.add_argument("--figure_dir", default=None, help="Figure output dir (default: <out_prefix>_figures)")
    args = ap.parse_args()

    shots = [int(x.strip()) for x in args.shots.split(",") if x.strip()]
    assert all(s > 0 for s in shots)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = args.out_prefix or f"./dataset_final_metrics_{ts}"
    out_prefix = os.path.abspath(out_prefix)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # Build base cfg once.
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    opts = ["NUM_GPUS", "1", "DEVICE", str(int(args.device.split(',')[0])), "RNG_SEED", str(args.seed)]
    if args.dino_ckpt:
        opts += ["TRAIN.backbone_checkpoint", str(args.dino_ckpt)]
    cfg.merge_from_list(opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Build dataset + loader once per shot (shot affects support sampling).
    overall_rows = []
    type_rows = []
    status_rows = []

    def _metrics_row(m: BinarySegMetrics) -> Dict[str, float]:
        # Only expose the requested metrics
        return {
            "pixel_acc": float(m.pixel_acc),
            "miou": float(m.miou),
            "dice": float(m.dice),  # Dice coefficient == F1 for binary segmentation
        }

    for shot in shots:
        print(f"[INFO] Evaluating shot={shot} ckpt={os.path.abspath(args.ckpt)}")
        cfg_s = cfg.clone()
        cfg_s.defrost()
        cfg_s.DATASET.shot = int(shot)
        cfg_s.freeze()

        ds_list = get_datasets(cfg=cfg_s, mode="test")
        # CUSTOM_MASK_ND uses cfg.DATASET.sub_datasets, but in this repo it defaults to ["original"].
        ds = ds_list[0]
        num_workers = cfg_s.TRAIN_SETUPS.num_workers if args.num_workers is None else int(args.num_workers)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = SOFS(cfg=cfg_s)
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model = model.cuda(cfg_s.DEVICE if hasattr(cfg_s, "DEVICE") else 0)
        # IMPORTANT: ensure inference path (forward without y) is used
        model.eval()
        model_ref = model.module if hasattr(model, "module") else model
        if hasattr(model_ref, "backbone"):
            model_ref.backbone.eval()

        # If saving figures, run a second pass that produces qualitative results, while still computing metrics.
        # (Keeps eval_grouped simple and avoids threading extra args through it.)
        if args.save_figures:
            fig_root = args.figure_dir or (out_prefix + "_figures")
            fig_root = os.path.abspath(fig_root)
            fig_save_path = os.path.join(fig_root, f"split_{int(cfg_s.DATASET.split)}_shot_{int(shot)}")
            os.makedirs(fig_save_path, exist_ok=True)

            test_num = 0
            with torch.no_grad():
                for data in tqdm(
                    loader,
                    desc=f"figures shot={shot}",
                    total=len(loader) if hasattr(loader, "__len__") else None,
                ):
                    if args.figure_sample_prob < 1.0 and torch.rand(1).item() > float(args.figure_sample_prob):
                        continue
                    # Run forward and recover original_output + heatmap (same logic as eval_grouped).
                    s_input = data["support_image"].cuda(non_blocking=True)
                    s_mask = data["support_mask"].cuda(non_blocking=True)
                    input = data["query_image"].cuda(non_blocking=True)
                    target = data["query_mask"].cuda(non_blocking=True)
                    query_original_shape = data["query_original_shape"][0]
                    query_input_shape = data["query_input_shape"][0]
                    query_crop_shape = data["query_crop_shape"][0]
                    img_position_list = data["img_position_list"]
                    ocf = _get_scalar(data["query_object_category_filename"])
                    sip = _get_scalar(data["support_img_path"])

                    if input.dim() == 4:
                        output = model(s_x=s_input, s_y=s_mask, x=input)
                        output_abs = output.max(1)[1][0]
                        output_heatmap = output[:, 1, ...][0]
                        original_output, original_heatmap = network_output2original_result(
                            query_input_shape=query_input_shape,
                            query_original_shape=query_original_shape,
                            output_absolute_val=output_abs,
                            output_heatmap=output_heatmap,
                        )
                    else:
                        grid_size = input.shape[1]
                        original_output = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
                        original_heatmap = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
                        original_position = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))

                        multiple_ND = grid_size // cfg_s.TEST_SETUPS.ND_batch_size
                        multiple_mod = grid_size % cfg_s.TEST_SETUPS.ND_batch_size
                        iter_num = multiple_ND if multiple_mod == 0 else multiple_ND + 1

                        output_total = []
                        for pointer_idx in range(iter_num):
                            init_val = pointer_idx * cfg_s.TEST_SETUPS.ND_batch_size
                            _, gs, c, h, w = input.shape
                            multiple_bs = cfg_s.TEST_SETUPS.ND_batch_size
                            if multiple_ND == 0:
                                multiple_bs = gs
                            if pointer_idx == multiple_ND:
                                multiple_bs = multiple_mod

                            support_input_m = s_input.repeat(multiple_bs, 1, 1, 1, 1)
                            support_mask_m = s_mask.repeat(multiple_bs, 1, 1, 1, 1)
                            query_input_m = input[:, init_val : init_val + multiple_bs, ...].reshape(1 * multiple_bs, c, h, w)
                            tmp_output = model(s_x=support_input_m, s_y=support_mask_m, x=query_input_m)
                            output_total.append(tmp_output)
                        output_total = torch.concat(output_total, dim=0)

                        for query_idx in range(grid_size):
                            top, down, left, right = [int(i) for i in img_position_list[query_idx]]
                            output = output_total[query_idx].unsqueeze(0)
                            output_abs = output.max(1)[1][0]
                            output_heatmap = output[:, 1, ...][0]

                            tuple_output_shape = tuple(torch.as_tensor(output_abs.shape).cpu().numpy())
                            tuple_query_input_shape = tuple(query_input_shape[query_idx].cpu().numpy())
                            tuple_query_crop_shape = tuple(query_crop_shape[query_idx].cpu().numpy())

                            if not (tuple_output_shape == tuple_query_input_shape and tuple_query_input_shape == tuple_query_crop_shape):
                                output_abs, output_heatmap = network_output2original_result(
                                    query_input_shape=query_input_shape[query_idx],
                                    query_original_shape=query_crop_shape[query_idx],
                                    output_absolute_val=output_abs,
                                    output_heatmap=output_heatmap,
                                )

                            original_output[query_idx, top:down, left:right] = output_abs.cpu()
                            original_heatmap[query_idx, top:down, left:right] = output_heatmap.cpu()
                            original_position[query_idx, top:down, left:right] = 1

                        original_output = torch.sum(original_output, dim=0) / torch.sum(original_position, dim=0)
                        original_heatmap = torch.sum(original_heatmap, dim=0) / torch.sum(original_position, dim=0)

                    original_mask = target.squeeze(0).squeeze(0)[: int(query_original_shape[0]), : int(query_original_shape[1])].cpu()

                    # Parse query info for produce_qualitative_result
                    query_object, query_category, query_filename = str(ocf).split("^")
                    test_num += 1
                    try:
                        produce_qualitative_result(
                            original_mask=original_mask,
                            oav=original_output,
                            ohm=original_heatmap,
                            source_path=cfg_s.TRAIN.dataset_path,
                            query_object=query_object,
                            query_filename=query_filename,
                            fig_save_path=fig_save_path,
                            test_num=test_num,
                            support_img_path=str(sip),
                        )
                    except Exception as e:
                        print(f"[WARN] figure_save failed for {query_filename}: {e}")

        overall_m, by_type_m, by_status_m = eval_grouped(loader, model, cfg_s)

        # overall_m: split -> metrics
        for split_key, m in overall_m.items():
            overall_rows.append({"shot": shot, "split": split_key, **_metrics_row(m)})
        for split_key, d in by_type_m.items():
            for k, m in d.items():
                type_rows.append({"shot": shot, "split": split_key, "type": k, **_metrics_row(m)})
        for split_key, d in by_status_m.items():
            for k, m in d.items():
                status_rows.append({"shot": shot, "split": split_key, "status": k, **_metrics_row(m)})

        m_all = overall_m.get("all")
        m_train = overall_m.get("train")
        m_test = overall_m.get("test")
        if m_all is not None:
            print(f"[OK] shot={shot} ALL  pixel_acc={m_all.pixel_acc:.6f} dice={m_all.dice:.6f} mIoU={m_all.miou:.6f}")
        if m_train is not None:
            print(f"[OK] shot={shot} TRAIN pixel_acc={m_train.pixel_acc:.6f} dice={m_train.dice:.6f} mIoU={m_train.miou:.6f}")
        if m_test is not None:
            print(f"[OK] shot={shot} TEST pixel_acc={m_test.pixel_acc:.6f} dice={m_test.dice:.6f} mIoU={m_test.miou:.6f}")

    def _write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    overall_csv = out_prefix + "_overall.csv"
    type_csv = out_prefix + "_by_type.csv"
    status_csv = out_prefix + "_by_status.csv"
    _write_csv(overall_csv, overall_rows)
    _write_csv(type_csv, type_rows)
    _write_csv(status_csv, status_rows)

    print(f"[OK] Wrote:\n  {overall_csv}\n  {type_csv}\n  {status_csv}")


if __name__ == "__main__":
    main()



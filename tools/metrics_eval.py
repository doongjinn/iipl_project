import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from utils.common import network_output2original_result

LOGGER = logging.getLogger(__name__)


@dataclass
class BinarySegMetrics:
    pixel_acc: float
    miou: float
    dice: float
    iou_fg: float
    iou_bg: float
    tp: int
    fp: int
    fn: int
    tn: int


def _accumulate_confusion(pred01: torch.Tensor, gt01: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    pred01, gt01: 2D tensors with values in {0,1}
    """
    pred01 = pred01.to(torch.int64)
    gt01 = gt01.to(torch.int64)
    tp = int(((pred01 == 1) & (gt01 == 1)).sum().item())
    fp = int(((pred01 == 1) & (gt01 == 0)).sum().item())
    fn = int(((pred01 == 0) & (gt01 == 1)).sum().item())
    tn = int(((pred01 == 0) & (gt01 == 0)).sum().item())
    return tp, fp, fn, tn


def _finalize(tp: int, fp: int, fn: int, tn: int) -> BinarySegMetrics:
    eps = 1e-10
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = (iou_fg + iou_bg) / 2.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return BinarySegMetrics(
        pixel_acc=float(pixel_acc),
        miou=float(miou),
        dice=float(dice),
        iou_fg=float(iou_fg),
        iou_bg=float(iou_bg),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


@torch.no_grad()
def eval_non_resize_loader_metrics(val_loader, model, cfg) -> BinarySegMetrics:
    """
    Evaluate a loader from BASE_DATASET_FSSS_ND-style datasets (supports query_image dim 4 or 5).
    Computes metrics on the *foreground* class plus standard pixel acc and mean IoU over {bg, fg}.
    """
    model.eval()
    model_ref = model.module if hasattr(model, "module") else model
    if hasattr(model_ref, "backbone"):
        model_ref.backbone.eval()

    tp = fp = fn = tn = 0

    for _, data in enumerate(val_loader):
        s_input = data["support_image"].cuda(non_blocking=True)
        s_mask = data["support_mask"].cuda(non_blocking=True)
        input = data["query_image"].cuda(non_blocking=True)
        target = data["query_mask"].cuda(non_blocking=True)

        query_original_shape = data["query_original_shape"][0]
        query_input_shape = data["query_input_shape"][0]
        query_crop_shape = data["query_crop_shape"][0]
        img_position_list = data["img_position_list"]

        assert input.shape[0] == 1, "evaluation assumes batch_size=1"

        if input.dim() == 4:
            output = model(s_x=s_input, s_y=s_mask, x=input)
            output_abs = output.max(1)[1][0]
            output_heatmap = output[:, 1, ...][0]
            original_output, _ = network_output2original_result(
                query_input_shape=query_input_shape,
                query_original_shape=query_original_shape,
                output_absolute_val=output_abs,
                output_heatmap=output_heatmap,
            )
        else:
            # patching / ND merge
            grid_size = input.shape[1]
            original_output = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))
            original_position = torch.zeros((grid_size, int(query_original_shape[0]), int(query_original_shape[1])))

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
                    output_abs, _ = network_output2original_result(
                        query_input_shape=query_input_shape[query_idx],
                        query_original_shape=query_crop_shape[query_idx],
                        output_absolute_val=output_abs,
                        output_heatmap=output_heatmap,
                    )

                original_output[query_idx, top:down, left:right] = output_abs.cpu()
                original_position[query_idx, top:down, left:right] = 1

            original_output = torch.sum(original_output, dim=0) / torch.sum(original_position, dim=0)

        original_mask = target.squeeze(0).squeeze(0)[: int(query_original_shape[0]), : int(query_original_shape[1])]

        pred = original_output
        if pred.dtype != torch.long:
            pred = (pred > 0.5).to(torch.uint8)
        if original_mask.dtype != torch.uint8:
            original_mask = (original_mask > 0.5).to(torch.uint8)

        # Align shapes defensively.
        if pred.shape != original_mask.shape:
            if pred.ndim == 2 and original_mask.ndim == 2 and pred.shape == tuple(reversed(original_mask.shape)):
                pred = pred.t()
            h = min(pred.shape[-2], original_mask.shape[-2])
            w = min(pred.shape[-1], original_mask.shape[-1])
            pred = pred[:h, :w]
            original_mask = original_mask[:h, :w]

        _tp, _fp, _fn, _tn = _accumulate_confusion(pred.cpu(), original_mask.cpu())
        tp += _tp
        fp += _fp
        fn += _fn
        tn += _tn

    return _finalize(tp, fp, fn, tn)



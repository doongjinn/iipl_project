import os
import logging
import re
from glob import glob
from typing import List, Tuple

import numpy as np
import cv2
import PIL.Image as PIL_Image

from datasets.base_dataset_fsss_ND import BASE_DATASET_FSSS_ND
from datasets.utilis_data import generate_category_filename

LOGGER = logging.getLogger(__name__)


def _list_images(image_dir: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob(os.path.join(image_dir, ext)))
    files = sorted(list(set(files)))
    return files


def _default_mask_path(mask_dir: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(mask_dir, f"{stem}_mask.png")

_DATASET_FINAL_NAME_RE = re.compile(r"^type(?P<type_idx>\d+)_(?P<status>normal|abnormal)_(?P<split>train|test)_")


def _parse_meta_from_filename(image_path: str) -> Tuple[int, str, str]:
    """
    Parse metadata from filename pattern:
      type{type_idx}_{normal|abnormal}_{train|test}_{orig_name}.ext
    Returns: (type_idx, status, split). If not matched, returns (-1, "unknown", "unknown").
    """
    base = os.path.basename(image_path)
    m = _DATASET_FINAL_NAME_RE.match(base)
    if not m:
        return -1, "unknown", "unknown"
    return int(m.group("type_idx")), str(m.group("status")), str(m.group("split"))


class CUSTOM_MASK_FSSS_ND(BASE_DATASET_FSSS_ND):
    """
    Custom dataset for few-shot semantic segmentation with binary masks.

    Expected inputs (configured via yaml):
      - TRAIN.dataset_path: images directory (e.g., /path/to/images)
      - TRAIN.mask_path: masks directory (e.g., /path/to/masks)
      - TEST.dataset_path / TEST.mask_path similarly

    Mask format:
      - One mask per image
      - Filename: <image_stem>_mask.png
      - Values: {0,255} or any >0 treated as foreground
    """

    def __init__(self, cfg, mode="train", **kwargs):
        super().__init__(cfg=cfg, mode=mode, kwargs=kwargs)
        # Optional status filtering for training-time data selection.
        # Example: status_filter=["abnormal"] to exclude normal images from episodic training.
        # Status is parsed from filename pattern: type{idx}_{normal|abnormal}_{train|test}_*.*
        status_filter = None
        if isinstance(kwargs, dict):
            status_filter = kwargs.get("status_filter", None)
        if status_filter is not None and not isinstance(status_filter, (list, tuple, set)):
            status_filter = [status_filter]
        self.status_filter = set([str(s) for s in status_filter]) if status_filter is not None else None

        self.object_name = cfg.DATASET.custom_object_name
        self.train_ratio = float(getattr(cfg.DATASET, "custom_train_ratio", 0.8))
        self.train_count = int(getattr(cfg.DATASET, "custom_train_count", 0) or 0)
        self.test_count = int(getattr(cfg.DATASET, "custom_test_count", 0) or 0)
        # If True, split train/test using filename tags (e.g., produced by dataset_final/split_samples.py)
        # Example filenames: type0_abnormal_train_*.jpg, type0_abnormal_test_*.jpg
        self.split_by_filename_tag = bool(getattr(cfg.DATASET, "custom_split_by_filename_tag", False))
        self.train_tag = str(getattr(cfg.DATASET, "custom_split_train_tag", "_train_"))
        self.test_tag = str(getattr(cfg.DATASET, "custom_split_test_tag", "_test_"))
        self.eval_use_all_pairs = bool(getattr(cfg.DATASET, "custom_eval_use_all_pairs", False))
        self.mask_source = cfg.TRAIN.mask_path if mode in ["train", "val"] else cfg.TEST.mask_path

        if not self.mask_source:
            raise ValueError(
                "CUSTOM_MASK_ND requires TRAIN.mask_path / TEST.mask_path to be set in the config."
            )

        all_images = _list_images(self.source)
        pairs: List[Tuple[str, str]] = []
        for img_path in all_images:
            mpath = _default_mask_path(self.mask_source, img_path)
            if os.path.exists(mpath):
                pairs.append((img_path, mpath))

        if len(pairs) < 2:
            raise RuntimeError(
                f"Not enough image/mask pairs found. images={self.source}, masks={self.mask_source}. "
                f"Need at least 2 paired samples, found {len(pairs)}."
            )

        # Split strategy:
        # 1) If filename-tag split is enabled, use that (train_tag/test_tag).
        # 2) Else fall back to deterministic split (sorted list) using counts or ratio.
        n_total = len(pairs)
        if self.split_by_filename_tag:
            train_pairs = []
            test_pairs = []
            for img_path, mask_path in pairs:
                base = os.path.basename(img_path)
                if self.train_tag in base:
                    train_pairs.append((img_path, mask_path))
                elif self.test_tag in base:
                    test_pairs.append((img_path, mask_path))
            if len(train_pairs) == 0 or len(test_pairs) == 0:
                raise RuntimeError(
                    "CUSTOM_MASK_ND: custom_split_by_filename_tag=True but could not find both "
                    f"train and test samples. train_tag={self.train_tag!r} test_tag={self.test_tag!r} "
                    f"train={len(train_pairs)} test={len(test_pairs)} total_pairs={n_total}."
                )
            n_train = len(train_pairs)
            n_test = len(test_pairs)
        else:
            # Deterministic split (sorted list).
            if self.train_count > 0 or self.test_count > 0:
                if self.train_count <= 0 or self.test_count <= 0:
                    raise ValueError(
                        "CUSTOM_MASK_ND: when using explicit counts, both DATASET.custom_train_count "
                        "and DATASET.custom_test_count must be > 0."
                    )
                if self.train_count + self.test_count > n_total:
                    raise RuntimeError(
                        f"CUSTOM_MASK_ND: requested train({self.train_count})+test({self.test_count}) "
                        f"> available_pairs({n_total})."
                    )
                n_train = self.train_count
                n_test = self.test_count
            else:
                n_train = max(1, int(n_total * self.train_ratio))
                n_train = min(n_train, n_total - 1)  # keep at least 1 sample for test
                n_test = n_total - n_train

            train_pairs = pairs[:n_train]
            test_pairs = pairs[n_train : n_train + n_test]

        if mode == "train":
            use_pairs = train_pairs
        else:
            # val/test share the held-out split by default, unless explicitly overridden.
            if self.eval_use_all_pairs:
                use_pairs = train_pairs + test_pairs
            elif bool(getattr(cfg.DATASET, "custom_eval_use_train_pairs", False)):
                use_pairs = train_pairs
            else:
                use_pairs = test_pairs

        # Apply optional status filter (typically used to separate abnormal vs normal streams in training).
        if self.status_filter is not None:
            filtered_pairs: List[Tuple[str, str]] = []
            for img_path, mask_path in use_pairs:
                _, status, _ = _parse_meta_from_filename(img_path)
                if status in self.status_filter:
                    filtered_pairs.append((img_path, mask_path))
            use_pairs = filtered_pairs

        if len(use_pairs) == 0:
            raise RuntimeError(
                f"[CUSTOM_MASK_ND] After status_filter={self.status_filter}, no pairs remain. "
                f"Consider disabling filtering or adjusting the split."
            )
        if len(use_pairs) < 2 and mode == "train":
            LOGGER.warning(
                f"[CUSTOM_MASK_ND] After status_filter={self.status_filter}, only {len(use_pairs)} train pair(s) remain. "
                "Episode support sampling may fall back to using the query image itself."
            )

        LOGGER.info(
            f"[CUSTOM_MASK_ND] mode={mode} total={n_total} train={n_train} test={n_test} "
            f"using={len(use_pairs)} object={self.object_name}"
        )

        # Build SOFS internal indexing structures.
        # Store image_path as the key; store mask_path in attributes to avoid loading masks in memory at init.
        filename_segmentation_category = {}
        filename_meta = {}
        for img_path, mask_path in use_pairs:
            filename_segmentation_category[img_path] = {
                "mask_path": mask_path,
                "category_sum": 1,
                "category": [0],
            }
            t_idx, status, split = _parse_meta_from_filename(img_path)
            filename_meta[img_path] = {"type_idx": t_idx, "status": status, "split": split}

        self.object_filename = {self.object_name: filename_segmentation_category}
        self._filename_meta = filename_meta
        self.object_category_filename = {
            self.object_name: generate_category_filename(1, {k: {"category": [0]} for k in filename_segmentation_category})
        }
        self.object_category_filename_list = [
            "^".join([self.object_name, "0", img_path]) for img_path in filename_segmentation_category.keys()
        ]

        # Optional: during eval, keep query from held-out split but sample support from training split.
        self.custom_support_from_train = bool(getattr(cfg.DATASET, "custom_support_from_train", False))
        self.support_object_category_filename = None
        self.support_object_filename = None
        if self.custom_support_from_train and mode != "train":
            support_filename_segmentation_category = {}
            support_filename_meta = {}
            for img_path, mask_path in train_pairs:
                support_filename_segmentation_category[img_path] = {
                    "mask_path": mask_path,
                    "category_sum": 1,
                    "category": [0],
                }
                t_idx, status, split = _parse_meta_from_filename(img_path)
                support_filename_meta[img_path] = {"type_idx": t_idx, "status": status, "split": split}
            self.support_object_category_filename = {
                self.object_name: generate_category_filename(
                    1, {k: {"category": [0]} for k in support_filename_segmentation_category}
                )
            }
            # For reading support masks from train split without KeyError.
            self.support_object_filename = {self.object_name: support_filename_segmentation_category}
            self._support_filename_meta = support_filename_meta
        else:
            self._support_filename_meta = None

    def __getitem__(self, idx):
        """
        Override BASE_DATASET_FSSS_ND to support: query from held-out split, support from train split (optional).
        """
        # Keep BASE_DATASET_FSSS_ND behavior for query selection (including test_sample_repeated_multiple logic).
        if self.mode == "train":
            tmp_idx = idx
        else:
            tmp_idx = idx // self.test_sample_repeated_multiple

        current_sample = self.object_category_filename_list[tmp_idx]
        query_object, query_category, query_filename = current_sample.split("^")
        query_category = int(query_category)

        support_category = query_category

        if self.support_object_category_filename is not None:
            sample_filename_list = self.support_object_category_filename[query_object][support_category]
        else:
            sample_filename_list = self.object_category_filename[query_object][support_category]

        # Robust support sampling:
        # - Prefer the configured support pool (train split when custom_support_from_train=True).
        # - Avoid choosing the query image as support when possible.
        # - If the pool collapses to only the query (e.g., a type has 1 train sample),
        #   fall back to the query pool (which may include test samples), and finally
        #   allow using the query itself to avoid infinite loops.
        def _sample_support(pool: list) -> list:
            pool_wo = [p for p in pool if p != query_filename]
            if len(pool_wo) >= self.shot:
                return random.sample(pool_wo, self.shot)
            if len(pool_wo) > 0:
                return random.choices(pool_wo, k=self.shot)
            # last resort
            return [query_filename] * self.shot

        acquire_k_shot_support = _sample_support(sample_filename_list)
        if query_filename in acquire_k_shot_support and self.support_object_category_filename is not None:
            # fallback to query pool (often contains both train+test when eval_use_all_pairs=True)
            fallback_pool = self.object_category_filename[query_object][support_category]
            acquire_k_shot_support = _sample_support(fallback_pool)

        support_img_path = [str(support_category)] + [i.replace("/", "_") for i in acquire_k_shot_support]

        generate_defect_state = False
        tmp_defect_mode = None

        support_image_list = []
        support_mask_list = []
        support_defect_status = []

        for each_support_sample in acquire_k_shot_support:
            input_image, mask_defect, defect_status = self.support_mode_generate_image_mask(
                tmp_filename=each_support_sample,
                tmp_object=query_object,
                tmp_category=support_category,
                defect_generation_state=generate_defect_state,
                tmp_defect_mode=tmp_defect_mode,
            )
            support_image_list.append(input_image)
            support_mask_list.append(mask_defect)
            support_defect_status.append(defect_status)

        support_image = torch.concat(support_image_list, dim=0)
        support_mask = torch.concat(support_mask_list, dim=0)
        support_defect_status_resize = sum(support_defect_status) == len(support_defect_status)

        sub_mode = "scale" if self.mode == "train" else "original"
        query_image, query_mask, query_original_shape, query_crop_shape, query_input_shape, img_position_list = self.generate_image_mask_(
            tmp_filename=query_filename,
            tmp_object=query_object,
            tmp_category=query_category,
            support_defect_status_resize=support_defect_status_resize,
            sub_mode=sub_mode,
            defect_generation_state=generate_defect_state,
            tmp_defect_mode=tmp_defect_mode,
        )

        # Expose sample meta for grouped evaluation (type/status/train-test split).
        meta = self._filename_meta.get(query_filename, {"type_idx": -1, "status": "unknown", "split": "unknown"})

        return {
            "query_image": query_image,
            "query_mask": query_mask,
            "query_original_shape": query_original_shape,
            "query_crop_shape": query_crop_shape,
            "query_input_shape": query_input_shape,
            "img_position_list": img_position_list,
            "support_image": support_image,
            "support_mask": support_mask,
            "query_object_category_filename": current_sample,
            "support_img_path": "_".join(support_img_path),
            "query_type_idx": meta["type_idx"],
            "query_status": meta["status"],
            "query_split": meta["split"],
        }

    def _read_mask_uint8(self, image_path: str, tmp_object: str) -> np.ndarray:
        # During eval, support may come from train split while query dict is built from test split.
        obj_dict = self.object_filename.get(tmp_object, {})
        if image_path in obj_dict:
            attr = obj_dict[image_path]
        elif self.support_object_filename is not None and image_path in self.support_object_filename.get(tmp_object, {}):
            attr = self.support_object_filename[tmp_object][image_path]
        else:
            raise KeyError(image_path)
        mask_path = attr["mask_path"]
        mask = np.array(PIL_Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def generate_image_mask_(
        self,
        tmp_filename,
        tmp_object,
        tmp_category,
        support_defect_status_resize,
        defect_generation_state,
        tmp_defect_mode,
        sub_mode="scale",
    ):
        # tmp_filename is an absolute image path for CUSTOM_MASK_ND
        file_name = tmp_filename
        temp_mask = self._read_mask_uint8(file_name, tmp_object)

        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_image_h, original_image_w = img.shape[:2]
        original_img_shape = img.shape[:2]

        processed_mask = np.array((temp_mask / 255) > 0).astype(np.uint8)
        defect_area_ratio = np.sum(processed_mask) / (temp_mask.shape[0] * temp_mask.shape[1])

        if (defect_area_ratio > self.area_resize_ratio and self.mode == "train") or (
            support_defect_status_resize and self.mode != "train"
        ):
            if self.mode == "train":
                img = np.array(self.first_step_transform_train(cv2_to_pil(img)))
                img, temp_mask = self.second_step_transform_train(img, temp_mask)
                temp_mask = temp_mask.astype(np.uint8)

            input_pil_image = self.transform_original_image.image_convert_pilimage(img.astype(np.uint8))
            input_image_torch = self.transform_function(input_pil_image)
            input_image = self.preprocess(input_image_torch, self.image_longest_size)

            original_img_shape = img.shape[:2]
            crop_img_shape = img.shape[:2]
            img_position_list = [1]
            input_image_shape = tuple(input_image_torch.shape[-2:])

            if sub_mode == "scale":
                current_mask_transform = self.transform_mask.apply_image(temp_mask)
                current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])
                mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")
            else:
                current_mask_torch = torch.as_tensor(temp_mask[None, :, :])
                mask_defect = self.preprocess(
                    current_mask_torch, self.test_unified_mask_longest_size, mode="gray"
                )
        else:
            if self.mode == "train":
                random_crop_ratio = random.uniform(*self.crop_ratio)
                crop_size = int(self.crop_size * random_crop_ratio)
            else:
                crop_size = self.crop_size
            target_size_h, target_size_w = min(crop_size, original_image_h), min(crop_size, original_image_w)

            if sub_mode == "scale":
                if random.uniform(0, 1) > self.normal_sample_sampling_prob:
                    condition_mask_h, condition_mask_w = np.where(processed_mask > 0)
                else:
                    condition_mask_h, condition_mask_w = np.where(processed_mask == 0)
                len_processed_mask = len(condition_mask_h)
                # If mask is empty (normal sample) and we happened to sample from >0 pixels,
                # fall back to background pixels to avoid crashing.
                if len_processed_mask == 0 and random.uniform(0, 1) > self.normal_sample_sampling_prob:
                    condition_mask_h, condition_mask_w = np.where(processed_mask == 0)
                    len_processed_mask = len(condition_mask_h)
                if len_processed_mask <= 0:
                    center_pixel = (original_image_h // 2, original_image_w // 2)
                else:
                    center_pixel_idx = random.randint(0, len_processed_mask - 1)
                    center_pixel = (condition_mask_h[center_pixel_idx], condition_mask_w[center_pixel_idx])
                residual_h, residual_w = original_image_h - center_pixel[0], original_image_w - center_pixel[1]

                mask_down_boundary_random, mask_right_boundary_random = random.randint(0, crop_size), random.randint(
                    0, crop_size
                )
                real_mask_down = min(residual_h, mask_down_boundary_random)
                real_mask_right = min(residual_w, mask_right_boundary_random)

                down_boundary = center_pixel[0] + real_mask_down
                right_boundary = center_pixel[1] + real_mask_right
                top_boundary = down_boundary - target_size_h
                left_boundary = right_boundary - target_size_w

                if top_boundary < 0:
                    top_boundary = 0
                    down_boundary = target_size_h

                if left_boundary < 0:
                    left_boundary = 0
                    right_boundary = target_size_w

                crop_img = img[top_boundary:down_boundary, left_boundary:right_boundary, :]
                crop_mask = temp_mask[top_boundary:down_boundary, left_boundary:right_boundary]

                if self.mode == "train":
                    crop_img = np.array(self.first_step_transform_train(cv2_to_pil(crop_img)))
                    crop_img, crop_mask = self.second_step_transform_train(crop_img, crop_mask)
                    crop_mask = crop_mask.astype(np.uint8)

                input_pil_image = self.transform_original_image.image_convert_pilimage(crop_img.astype(np.uint8))
                input_image_torch = self.transform_function(input_pil_image)
                input_image = self.preprocess(input_image_torch, self.image_longest_size)

                current_mask_transform = self.transform_mask.apply_image(crop_mask)
                current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])
                mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")

                crop_img_shape = crop_img.shape[:2]
                input_image_shape = tuple(input_image_torch.shape[-2:])
                img_position_list = [1]
            else:
                # test/original mode patching
                h_num = original_image_h // target_size_h
                w_num = original_image_w // target_size_w

                h_remainder = original_image_h % target_size_h
                w_remainder = original_image_w % target_size_w

                if h_remainder != 0:
                    h_num += 1
                if w_remainder != 0:
                    w_num += 1

                img_list = []
                crop_img_shape_list = []
                input_image_shape_list = []
                img_position_list = []

                for temp_h in range(h_num):
                    for temp_w in range(w_num):
                        if temp_h != h_num - 1 and temp_w != w_num - 1:
                            tmp_img = img[
                                temp_h * target_size_h : (temp_h + 1) * target_size_h,
                                temp_w * target_size_w : (temp_w + 1) * target_size_w,
                                :,
                            ]
                            img_position_list.append(
                                (
                                    temp_h * target_size_h,
                                    (temp_h + 1) * target_size_h,
                                    temp_w * target_size_w,
                                    (temp_w + 1) * target_size_w,
                                )
                            )
                        elif temp_h == h_num - 1 and temp_w != w_num - 1:
                            tmp_img = img[
                                original_image_h - target_size_h : original_image_h,
                                temp_w * target_size_w : (temp_w + 1) * target_size_w,
                                :,
                            ]
                            img_position_list.append(
                                (
                                    original_image_h - target_size_h,
                                    original_image_h,
                                    temp_w * target_size_w,
                                    (temp_w + 1) * target_size_w,
                                )
                            )
                        elif temp_w == w_num - 1 and temp_h != h_num - 1:
                            tmp_img = img[
                                temp_h * target_size_h : (temp_h + 1) * target_size_h,
                                original_image_w - target_size_w : original_image_w,
                                :,
                            ]
                            img_position_list.append(
                                (
                                    temp_h * target_size_h,
                                    (temp_h + 1) * target_size_h,
                                    original_image_w - target_size_w,
                                    original_image_w,
                                )
                            )
                        else:
                            tmp_img = img[
                                original_image_h - target_size_h : original_image_h,
                                original_image_w - target_size_w : original_image_w,
                                :,
                            ]
                            img_position_list.append(
                                (
                                    original_image_h - target_size_h,
                                    original_image_h,
                                    original_image_w - target_size_w,
                                    original_image_w,
                                )
                            )

                        input_pil_image = self.transform_original_image.image_convert_pilimage(tmp_img.astype(np.uint8))
                        input_image_torch = self.transform_function(input_pil_image)
                        input_image = self.preprocess(input_image_torch, self.image_longest_size)

                        crop_img_shape_list.append(torch.as_tensor(tmp_img.shape[:2]))
                        input_image_shape_list.append(torch.as_tensor(tuple(input_image_torch.shape[-2:])))
                        img_list.append(input_image)

                input_image = torch.stack(img_list, dim=0)
                crop_img_shape = torch.stack(crop_img_shape_list, dim=0)
                input_image_shape = torch.stack(input_image_shape_list, dim=0)

                current_mask_torch = torch.as_tensor(temp_mask[None, :, :])
                mask_defect = self.preprocess(current_mask_torch, self.test_unified_mask_longest_size, mode="gray")

        mask_defect = (mask_defect > 0.1).float()
        return (
            input_image,
            mask_defect,
            torch.as_tensor(original_img_shape),
            torch.as_tensor(crop_img_shape),
            torch.as_tensor(input_image_shape),
            img_position_list,
        )

    def support_mode_generate_image_mask(
        self,
        tmp_filename,
        tmp_object,
        tmp_category,
        defect_generation_state,
        tmp_defect_mode,
    ):
        file_name = tmp_filename
        temp_mask = self._read_mask_uint8(file_name, tmp_object)

        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_image_h, original_image_w = img.shape[:2]

        processed_mask = np.array((temp_mask / 255) > 0).astype(np.uint8)
        defect_area_ratio = np.sum(processed_mask) / (temp_mask.shape[0] * temp_mask.shape[1])
        defect_status = defect_area_ratio > self.area_resize_ratio

        if defect_area_ratio > self.area_resize_ratio:
            if self.mode == "train":
                img = np.array(self.first_step_transform_train(cv2_to_pil(img)))
                img, temp_mask = self.second_step_transform_train(img, temp_mask)
                temp_mask = temp_mask.astype(np.uint8)

            input_pil_image = self.transform_original_image.image_convert_pilimage(img.astype(np.uint8))
            input_image_torch = self.transform_function(input_pil_image)
            input_image = self.preprocess(input_image_torch, self.image_longest_size)
            input_image = input_image.unsqueeze(0).repeat(self.s_in_shot, 1, 1, 1)

            current_mask_transform = self.transform_mask.apply_image(temp_mask)
            current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])
            mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")
            mask_defect = mask_defect.unsqueeze(0).repeat(self.s_in_shot, 1, 1, 1)
        else:
            # If the mask is empty (normal sample), fall back to sampling from background pixels
            # so we can still create a valid support crop without crashing.
            condition_mask_h, condition_mask_w = np.where(processed_mask > 0)
            if len(condition_mask_h) == 0:
                condition_mask_h, condition_mask_w = np.where(processed_mask == 0)
            len_processed_mask = len(condition_mask_h)

            s_in_shot_img_list = []
            s_in_shot_mask_list = []
            for _ in range(self.s_in_shot):
                if self.mode == "train":
                    random_crop_ratio = random.uniform(*self.crop_ratio)
                    crop_size = int(self.crop_size * random_crop_ratio)
                else:
                    crop_size = self.crop_size
                target_size_h, target_size_w = min(crop_size, original_image_h), min(crop_size, original_image_w)

                if len_processed_mask <= 0:
                    # Extremely defensive fallback; should not happen because background exists.
                    center_pixel = (original_image_h // 2, original_image_w // 2)
                else:
                    center_pixel_idx = random.randint(0, len_processed_mask - 1)
                    center_pixel = (condition_mask_h[center_pixel_idx], condition_mask_w[center_pixel_idx])
                residual_h, residual_w = original_image_h - center_pixel[0], original_image_w - center_pixel[1]

                mask_down_boundary_random, mask_right_boundary_random = random.randint(0, crop_size), random.randint(
                    0, crop_size
                )
                real_mask_down = min(residual_h, mask_down_boundary_random)
                real_mask_right = min(residual_w, mask_right_boundary_random)

                down_boundary = center_pixel[0] + real_mask_down
                right_boundary = center_pixel[1] + real_mask_right
                top_boundary = down_boundary - target_size_h
                left_boundary = right_boundary - target_size_w

                if top_boundary < 0:
                    top_boundary = 0
                    down_boundary = target_size_h
                if left_boundary < 0:
                    left_boundary = 0
                    right_boundary = target_size_w

                crop_img = img[top_boundary:down_boundary, left_boundary:right_boundary, :]
                crop_mask = temp_mask[top_boundary:down_boundary, left_boundary:right_boundary]

                if self.mode == "train":
                    crop_img = np.array(self.first_step_transform_train(cv2_to_pil(crop_img)))
                    crop_img, crop_mask = self.second_step_transform_train(crop_img, crop_mask)
                    crop_mask = crop_mask.astype(np.uint8)

                input_pil_image = self.transform_original_image.image_convert_pilimage(crop_img.astype(np.uint8))
                input_image_torch = self.transform_function(input_pil_image)
                input_image = self.preprocess(input_image_torch, self.image_longest_size)

                current_mask_transform = self.transform_mask.apply_image(crop_mask)
                current_mask_torch = torch.as_tensor(current_mask_transform[None, :, :])
                mask_defect = self.preprocess(current_mask_torch, self.mask_longest_size, mode="gray")

                s_in_shot_img_list.append(input_image)
                s_in_shot_mask_list.append(mask_defect)

            input_image = torch.stack(s_in_shot_img_list, dim=0)
            mask_defect = torch.stack(s_in_shot_mask_list, dim=0)

        mask_defect = (mask_defect > 0.1).float()
        return input_image, mask_defect, defect_status


# Local imports used by the copied augmentation logic above.
import torch  # noqa: E402
import random  # noqa: E402
from torchvision.transforms.functional import to_pil_image as cv2_to_pil  # noqa: E402



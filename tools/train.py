import os.path
import random
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback

from utils import init_distributed_training, is_master_proc, seed_everything, \
    get_datasets, setup_logging

from model.SOFS import SOFS

from tools.epoch_train_eval_ss import epoch_train_ss, epoch_validate_ss, epoch_validate_non_resize_ss

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler

LOGGER = logging.getLogger(__name__)

from datasets.custom_mask_fsss_ND import CUSTOM_MASK_FSSS_ND  # noqa: E402


def train(cfg):
    """
    include data loader load, model load, optimizer, training and test.
    """
    # set up environment
    init_distributed_training(cfg)

    if torch.distributed.is_initialized():
        temp_rank = torch.distributed.get_rank()
    else:
        temp_rank = 0
    # Set random seed from configs.
    seed_everything(cfg.RNG_SEED + temp_rank)

    # setup logging
    # only for multiple GPUs
    # todo for a single GPU
    if is_master_proc():
        setup_logging(cfg)
        LOGGER.info(pprint.pformat(cfg))

    try:
        LOGGER.info("start main training!")
        cur_device = torch.cuda.current_device()

        LOGGER.info("load dataset!")
        # get train dataloader (include each category)
        train_datasets = get_datasets(cfg=cfg, mode='train')
        # if is_master_proc():
        if cfg.TRAIN_SETUPS.TEST_SETUPS.test_state:
            if cfg.TRAIN_SETUPS.TEST_SETUPS.val_state:
                val_datasets = get_datasets(cfg=cfg, mode='val')
            test_datasets = get_datasets(cfg=cfg, mode='test')

        LOGGER.info("load complete!")

        result_collect = {"AUROC": [],
                          "Pixel-AUROC": [],
                          "per-region-overlap (PRO)": []}

        # training process
        for idx, individual_datasets in enumerate(train_datasets):
            LOGGER.info("current dataset is {}.".format(individual_datasets.name))
            LOGGER.info("the data in current dataset {} are {}.".format(individual_datasets.name,
                                                                        len(individual_datasets)))

            # start training
            torch.cuda.empty_cache()

            normal_train_loader = None
            # Special handling for CUSTOM_MASK_ND:
            # - Use abnormal-only episodes for the main defect-localization loss.
            # - Use normal-only samples for Mixed Normal Dice Loss (MNDL) as an auxiliary regularizer.
            if cfg.DATASET.name == "CUSTOM_MASK_ND" and bool(getattr(cfg.TRAIN.SOFS, "exclude_normal_from_episode", False)):
                LOGGER.info("[CUSTOM_MASK_ND] exclude_normal_from_episode=True: building abnormal-only train loader")
                individual_datasets = CUSTOM_MASK_FSSS_ND(cfg=cfg, mode="train", status_filter=["abnormal"])
                individual_datasets.name = cfg.DATASET.name + "_abnormal_only"
                try:
                    normal_dataset = CUSTOM_MASK_FSSS_ND(cfg=cfg, mode="train", status_filter=["normal"])
                    normal_dataset.name = cfg.DATASET.name + "_normal_only"
                    if cfg.NUM_GPUS > 1:
                        normal_sampler = torch.utils.data.distributed.DistributedSampler(normal_dataset)
                        normal_train_loader = torch.utils.data.DataLoader(
                            normal_dataset,
                            batch_size=cfg.TRAIN_SETUPS.batch_size,
                            shuffle=(normal_sampler is None),
                            num_workers=cfg.TRAIN_SETUPS.num_workers,
                            pin_memory=True,
                            sampler=normal_sampler,
                            drop_last=True,
                        )
                    else:
                        normal_train_loader = torch.utils.data.DataLoader(
                            normal_dataset,
                            batch_size=cfg.TRAIN_SETUPS.batch_size,
                            shuffle=True,
                            num_workers=cfg.TRAIN_SETUPS.num_workers,
                            pin_memory=True,
                            drop_last=True,
                        )
                    LOGGER.info(
                        f"[CUSTOM_MASK_ND] normal-only loader enabled for MNDL, n={len(normal_dataset)} "
                        f"mndl_weight={float(getattr(cfg.TRAIN.SOFS, 'mndl_weight', 0.0) or 0.0)}"
                    )
                except Exception as e:
                    LOGGER.warning(f"[CUSTOM_MASK_ND] normal-only loader disabled (could not build): {e}")

            if cfg.NUM_GPUS > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(individual_datasets)
                train_loader = torch.utils.data.DataLoader(
                    individual_datasets, batch_size=cfg.TRAIN_SETUPS.batch_size, shuffle=(train_sampler is None),
                    num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            else:
                train_loader = torch.utils.data.DataLoader(
                    individual_datasets, batch_size=cfg.TRAIN_SETUPS.batch_size, shuffle=True,
                    num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True)

            # 多卡测试
            # if is_master_proc():
            if cfg.TRAIN_SETUPS.TEST_SETUPS.test_state:
                if cfg.TRAIN_SETUPS.TEST_SETUPS.val_state:
                    val_loader_list = []
                    val_sampler_list = []

                    for test_idx in range(len(cfg.DATASET.sub_datasets)):
                        individual_val_datasets = val_datasets[test_idx]
                        LOGGER.info("current dataset is {}.".format(individual_val_datasets.name))
                        LOGGER.info("the val data in current dataset {} are {}.".format(individual_val_datasets.name,
                                                                                        len(individual_val_datasets)))
                        if cfg.NUM_GPUS > 1:
                            val_sampler = torch.utils.data.distributed.DistributedSampler(individual_val_datasets)
                            val_loader = torch.utils.data.DataLoader(
                                individual_val_datasets, batch_size=cfg.TEST_SETUPS.batch_size, shuffle=False,
                                num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True, sampler=val_sampler,
                                drop_last=False)
                            val_sampler_list.append(val_sampler)
                        else:
                            val_loader = torch.utils.data.DataLoader(
                                individual_val_datasets, batch_size=cfg.TEST_SETUPS.batch_size, shuffle=False,
                                num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True, drop_last=False)

                        val_loader_list.append(val_loader)

                test_loader_list = []
                test_sampler_list = []

                for test_idx in range(len(cfg.DATASET.sub_datasets)):
                    individual_test_datasets = test_datasets[test_idx]
                    LOGGER.info("current dataset is {}.".format(individual_test_datasets.name))
                    LOGGER.info("the test data in current dataset {} are {}.".format(individual_test_datasets.name,
                                                                                     len(individual_test_datasets)))
                    if cfg.NUM_GPUS > 1:
                        test_sampler = torch.utils.data.distributed.DistributedSampler(individual_test_datasets)
                        test_loader = torch.utils.data.DataLoader(
                            individual_test_datasets, batch_size=cfg.TEST_SETUPS.batch_size, shuffle=False,
                            num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True, sampler=test_sampler,
                            drop_last=False)
                        test_sampler_list.append(test_sampler)
                    else:
                        test_loader = torch.utils.data.DataLoader(
                            individual_test_datasets, batch_size=cfg.TEST_SETUPS.batch_size, shuffle=False,
                            num_workers=cfg.TRAIN_SETUPS.num_workers, pin_memory=True, drop_last=False)

                    test_loader_list.append(test_loader)

            LOGGER.info("load model!")
            if cfg.TRAIN.method == "SOFS":
                model = SOFS(cfg=cfg)
            else:
                raise NotImplementedError("train method is not in the target list!")

            if cfg.TRAIN.load_checkpoint:
                save_checkpoint = torch.load(cfg.TRAIN.load_model_path,
                                             map_location="cpu")
                model.load_state_dict(save_checkpoint)
                LOGGER.info("load main model successful!")

            LOGGER.info("load optimizer!")
            if cfg.TRAIN.method == "SOFS":
                LR = cfg.TRAIN_SETUPS.learning_rate

                target_params_id = []
                for name, para in model.named_parameters():
                    if "query_semantic_transformer" in name:
                        target_params_id.append(id(para))

                target_params_id = list(set(target_params_id))
                backbone_params = list(map(id, model.backbone.parameters()))
                base_params = filter(lambda p: id(
                    p) not in target_params_id + backbone_params,
                                     model.parameters())
                target_params = filter(lambda p: id(
                    p) in target_params_id,
                                       model.parameters())

                params = [{'params': target_params},
                          {'params': base_params, 'lr': LR * cfg.TRAIN_SETUPS.lr_multiple}]

                optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=cfg.TRAIN_SETUPS.weight_decay,
                                              betas=(0.9, 0.999))
            else:
                raise NotImplementedError("train method is not in the target list!")

            model = model.cuda(cur_device)

            if cfg.NUM_GPUS > 1:
                # Make model replica operate on the current device
                model = torch.nn.parallel.DistributedDataParallel(
                    module=model,
                    device_ids=[cur_device],
                    output_device=cur_device,
                    find_unused_parameters=False,
                )
            else:
                # todo a single gpu
                pass

            # start training!
            for epoch in range(1, cfg.TRAIN_SETUPS.epochs + 1):
                if cfg.NUM_GPUS > 1:
                    train_sampler.set_epoch(epoch)

                if epoch % cfg.TRAIN_SETUPS.TEST_SETUPS.train_miou == 0:
                    validate_each_class = True
                else:
                    validate_each_class = False

                if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND", "CUSTOM_MASK_ND", "ECCV_Contest_ND"]:
                    if cfg.TRAIN.method in ["SOFS"]:
                        epoch_train_ss(
                            train_loader=train_loader,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            cfg=cfg,
                            validate_each_class=validate_each_class,
                            normal_loader=normal_train_loader
                        )
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                # Save a rolling checkpoint each epoch so evaluation failures don't lose the trained weights.
                if cfg.TRAIN.save_model and is_master_proc():
                    base_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
                    os.makedirs(base_path, exist_ok=True)
                    last_name = os.path.join(base_path, "last.pth")
                    model_ref = model.module if hasattr(model, "module") else model
                    state_dict = model_ref.state_dict()
                    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
                    torch.save(cpu_state_dict, last_name)

                # if is_master_proc():
                if cfg.TRAIN_SETUPS.TEST_SETUPS.test_state:
                    """
                                                    prediction
                                                ______1________0____
                                              1 |    TP   |   FN   |
                                ground truth  0 |    FP   |   TN   |

                                ACC = (TP + TN) / (TP + FP + FN + TN)

                                precision = TP / (TP + FP)

                                recall (TPR) = TP / (TP + FN)

                                FPR（False Positive Rate）= FP / (FP + TN)
                    """
                    if epoch % cfg.TRAIN_SETUPS.TEST_SETUPS.epoch_test == 0:
                        # Also save a snapshot at evaluation epochs for convenience.
                        if cfg.TRAIN.save_model and is_master_proc():
                            base_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
                            os.makedirs(base_path, exist_ok=True)
                            snap_name = os.path.join(base_path, f"epoch_{epoch}.pth")
                            model_ref = model.module if hasattr(model, "module") else model
                            state_dict = model_ref.state_dict()
                            cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
                            torch.save(cpu_state_dict, snap_name)

                        for test_idx in range(len(cfg.DATASET.sub_datasets)):
                            if cfg.TRAIN_SETUPS.TEST_SETUPS.val_state:
                                tmp_val_loader = val_loader_list[test_idx]
                                if cfg.NUM_GPUS > 1:
                                    tmp_val_sampler = val_sampler_list[test_idx]
                                    tmp_val_sampler.set_epoch(cfg.RNG_SEED)

                            tmp_test_loader = test_loader_list[test_idx]
                            if cfg.NUM_GPUS > 1:
                                tmp_test_sampler = test_sampler_list[test_idx]
                                tmp_test_sampler.set_epoch(cfg.RNG_SEED)

                            if cfg.DATASET.name in ["VISION_V1", "VISION_V1_ND", "CUSTOM_MASK_ND"]:
                                if cfg.TRAIN.method in ["SOFS"]:
                                    if cfg.DATASET.name in ["VISION_V1_ND", "CUSTOM_MASK_ND"]:
                                        if cfg.TRAIN_SETUPS.TEST_SETUPS.val_state:
                                            epoch_validate_non_resize_ss(
                                                val_loader=tmp_val_loader,
                                                model=model,
                                                epoch=epoch,
                                                cfg=cfg,
                                                rand_seed=cfg.RNG_SEED,
                                                mode="val"
                                            )
                                        epoch_validate_non_resize_ss(
                                            val_loader=tmp_test_loader,
                                            model=model,
                                            epoch=epoch,
                                            cfg=cfg,
                                            rand_seed=cfg.RNG_SEED,
                                            mode="test"
                                        )
                                    elif cfg.DATASET.name in ["VISION_V1"]:
                                        epoch_validate_ss(
                                            val_loader=tmp_test_loader,
                                            model=model,
                                            epoch=epoch,
                                            cfg=cfg,
                                            rand_seed=cfg.RNG_SEED
                                        )
                                    else:
                                        raise NotImplementedError
                                else:
                                    raise NotImplementedError
                            else:
                                raise NotImplementedError

            # save model
            if cfg.TRAIN.save_model:
                if is_master_proc():
                    base_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints")

                    save_name = "_".join(
                        ["best", "method", cfg.TRAIN.method, cfg.DATASET.name,
                         "split_" + str(cfg.DATASET.split), ".pth"])
                    while os.path.isfile(os.path.join(base_path, save_name)):
                        save_name = "new_" + save_name
                    save_name = os.path.join(base_path, save_name)

                    model_ref = model.module if hasattr(model, "module") else model
                    if cfg.TRAIN.method == "SOFS_ada":
                        state_dict = model_ref.backbone.state_dict()
                    else:
                        state_dict = model_ref.state_dict()
                    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
                    torch.save(cpu_state_dict, save_name)
                    LOGGER.info("Model save in {}".format(save_name))

        LOGGER.info("Method training phase complete!")
    except Exception as e:
        LOGGER.error("error：")
        LOGGER.error(e)
        LOGGER.error("\n" + traceback.format_exc())

import os
from typing import List, Optional, Tuple

import hydra
import mmcv
import pyrootutils
import pytorch_lightning as pl
import torch
from mmcv.utils import Config, get_git_hash
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.loggers import LightningLoggerBase

# import torch

# torch.nn.utils.clip_grad_norm_
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from lightnings import utils
from lightnings.lightning_dc_data_module import  BaseDCDataModuleV2
from lightnings.lightning_model_module import BaseModelModuleV2

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #
from mmseg import __version__
from mmcd import __version__
from mmseg.apis import init_random_seed
from mmseg.utils import get_device, setup_multi_processes

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def train(args: DictConfig) -> Tuple[dict, dict]:

    cfg_path = args.cfg_path

    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(
            hydra.utils.get_original_cwd(), cfg_path
        )
    cfg = Config.fromfile(cfg_path)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args.paths.output_dir
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    cfg.TRAIN = OmegaConf.to_object(args.TRAIN)

    if args.ckpt_path is not None:
        cfg.resume_from = args.resume_from

    # set seed for random number generators in pytorch, numpy and python.random
    # set random seeds
    cfg.device = get_device()
    seed = args.get('seed', None)
    if seed is None:
        seed = init_random_seed(args.seed, device=cfg.device)

    log.info(f'Set random seed to {seed}')
    cfg.seed = seed
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating dataModule <>")
    
    # 防止 直接修改 cfg。data ,保证dump下的是函数之前的
    
    cfg.dump(os.path.join(cfg.work_dir,
             os.path.basename(args.cfg_path)))
    datamodule: LightningDataModule = BaseDCDataModuleV2(cfg.data)
    datamodule.setup()
    # add an attribute for visualization convenience
    cfg.CLASSES = datamodule.val_dataset.CLASSES
    cfg.ignore_index = datamodule.train_dataset.ignore_index

    log.info(f"Instantiating modelModule <>")
    cfg.load_from = args.ckpt_path

    # ----
    evaluate_dataset = datamodule.val_dataset
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    eval_hook_args = [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule', 'by_epoch'
    ]
    for key in eval_hook_args:
        eval_kwargs.pop(key, None)

    lightning_model: LightningModule = BaseModelModuleV2(
        cfg, args, evaluate_dataset, eval_kwargs, CKPT=cfg.get("load_from"))
    log.info(lightning_model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        args.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(
        args.get("logger"))

    log.info(f"Instantiating trainer <{args.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        args.trainer, callbacks=callbacks, logger=logger)
    args.datamodule = cfg.data.to_dict()
    args.model = cfg.model.to_dict()
    object_dict = {
        "cfg": args,
        "datamodule": datamodule,
        "model": lightning_model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting training!")
    cfg.resume_from = args.resume_from
    trainer.fit(model=lightning_model, datamodule=datamodule,
                ckpt_path=cfg.get("resume_from", None))

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    log.info(f"Best ckpt path: {ckpt_path}")

    if ckpt_path is not None and os.path.exists(ckpt_path):
        best_ckpt = torch.load(ckpt_path, map_location='cuda:0')
        best_ckpt_state_dict = best_ckpt['state_dict']
        lightning_model.model.load_state_dict(best_ckpt_state_dict)

    if args.get("test"):
        log.info("Starting testing!")
        trainer.test(model=lightning_model,
                     datamodule=datamodule)

    train_metrics = trainer.callback_metrics
    return train_metrics, object_dict


@hydra.main(version_base="1.2", config_path=root / "hydra_configs", config_name="train.yaml")
def main(args: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(args)
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=args.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

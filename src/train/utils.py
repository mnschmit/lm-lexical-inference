from typing import Callable
import argparse
from pathlib import Path
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning import LightningModule


def add_generic_args(parser) -> None:
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=True,
        help="The checkpoint directory where the checkpoints will be written.",
    )

    parser.add_argument('--experiment_name', required=True, default='default')

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val",
                        default=1.0, type=float, help="Max gradient norm")

    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument('--seed', type=int, default=47110815)
    parser.add_argument('--gpus', type=int, nargs='+', default=[])


def generic_train(
        model_cls: Callable[[argparse.Namespace], LightningModule],
        args: argparse.Namespace
):
    pl.seed_everything(args.seed)

    # init model
    model = model_cls(args)

    cdir = Path(os.path.join(
        model.hparams.checkpoint_dir, args.experiment_name))
    cdir.mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=cdir, prefix="checkpoint",
        monitor="AUC", mode="max", save_top_k=1
    )
    lr_logger_callback = LearningRateMonitor(logging_interval='epoch')
    logger = TestTubeLogger('tt_logs', name=args.experiment_name)

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if len(args.gpus) > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[lr_logger_callback],
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        **train_params
    )

    trainer.fit(model)

    return trainer, model

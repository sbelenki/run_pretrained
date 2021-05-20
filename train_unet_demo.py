"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import torch
from pathlib import Path
import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule, NestedUnetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = None
    if args.unet_module == "unet":
        model = UnetModule(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=int(args.chans),
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            optimizer=args.optmizer,
        )
    elif args.unet_module == "nestedunet":
        model = NestedUnetModule(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            optimizer = args.optmizer,
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("The requested cuda device isn't available please set --device cpu")

    pretrained_dict = torch.load(args.state_dict_file, map_location=args.device)
    model_dict = model.unet.state_dict()
    if args.unet_module == "unet":
        model_dict = { k: pretrained_dict["classy_state_dict"]["base_model"]["model"]["trunk"]["_feature_blocks.unetblock." + k] for k, _ in model_dict.items()}
    elif args.unet_module == "nestedunet":
        model_dict = {k: pretrained_dict["classy_state_dict"]["base_model"]["model"]["trunk"]["_feature_blocks.nublock." + k] for k, v in model_dict.items()}

    model.unet.load_state_dict(model_dict)

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    output_filename = f"fine_tuned_{args.unet_module}.torch"
    output_model_filepath = f"{args.output_path}/{output_filename}"
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
        print(f"Saving model: {output_model_filepath}")
        torch.save(model.state_dict(), output_model_filepath)
        print("DONE!")
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()
    batch_size = 1

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # unet module arguments
    parser.add_argument(
        "--unet_module",
        default = "unet",
        choices = ("unet", "nestedunet"),
        type = str,
        help = "Unet module to run with",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--output_path",
        type=Path, # directory for logs and checkpoints
        default=Path("./fine_tuning"),
        help="Path for saving reconstructions",
    )

    # unet specific
    parser.add_argument(
        "--in_chans",
        default = 1,
        type = int,
        help = "number of input channels to U-Net",
    )
    parser.add_argument(
        "--out_chans",
        default = 1,
        type = int,
        help = "number of output chanenls to U-Net",
    )
    parser.add_argument(
        "--chans",
        default = 32,
        type = int,
        help = "number of top-level U-Net channels",
    )
    # RMSProp parameters
    parser.add_argument(
        "--opt_drop_prob",
        default = 0.0,
        type = float,
        help = "dropout probability",
    )
    parser.add_argument(
        "--opt_lr",
        default = 0.001,
        type = float,
        help = "RMSProp learning rate",
    )
    parser.add_argument(
        "--opt_lr_step_size",
        default = 10,
        type = int,
        help = "epoch at which to decrease learning rate",
    )
    parser.add_argument(
        "--opt_lr_gamma",
        default = 0.1,
        type = float,
        help = "extent to which to decrease learning rate",
    )
    parser.add_argument(
        "--opt_weight_decay",
        default = 0.0,
        type = float,
        help = "weight decay regularization strength",
    )
    parser.add_argument(
        "--opt_optimizer",
        choices = ("RMSprop", "Adam"),
        default = "RMSprop",
        type = str,
        help = "optimizer (RMSprop, Adam)",
    )

    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path="/home/ec2-user/mri", batch_size=batch_size, test_path=None)

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=0,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        max_epochs=50,  # max number of epochs
        unet_module="unet", # "unet" or "nestedunet"
    )

    args = parser.parse_args()

    # module config
    if args.unet_module == "unet":
        parser = UnetModule.add_model_specific_args(parser)
        parser.set_defaults(
            num_pool_layers=4,  # number of U-Net pooling layers
            drop_prob=args.opt_drop_prob,  # dropout probability
            lr=args.opt_lr,  # RMSProp learning rate
            lr_step_size= args.opt_lr_step_size,  # epoch at which to decrease learning rate
            lr_gamma= args.opt_lr_gamma,  # extent to which to decrease learning rate
            weight_decay=args.opt_weight_decay,  # weight decay regularization strength
            optmizer=args.opt_optimizer, # optimizer (RMSprop, Adam)
            accelerator = "ddp_cpu" if args.device == "cpu" else "ddp",
        )
    elif args.unet_module == "nestedunet":
        parser = NestedUnetModule.add_model_specific_args(parser)
        parser.set_defaults(
            num_pool_layers=4,  # number of U-Net pooling layers
            drop_prob = args.opt_drop_prob,  # dropout probability
            lr = args.opt_lr,  # RMSProp learning rate
            lr_step_size = args.opt_lr_step_size,  # epoch at which to decrease learning rate
            lr_gamma = args.opt_lr_gamma,  # extent to which to decrease learning rate
            weight_decay = args.opt_weight_decay,  # weight decay regularization strength
            optmizer = args.opt_optimizer,  # optimizer (RMSprop, Adam)
            accelerator = "ddp_cpu" if args.device == "cpu" else "ddp",
        )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.output_path / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_path / "checkpoints",
        save_top_k=True,
        verbose=True,
        monitor="validation_loss",
        mode="min",
        prefix="",
    )

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()

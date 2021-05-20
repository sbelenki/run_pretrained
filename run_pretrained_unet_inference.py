"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import fastmri
import fastmri.data.transforms as T
from fastmri.data.subsample import create_mask_for_mask_type
import numpy as np
import requests
import torch
from fastmri.data import SliceDataset
from fastmri.models import Unet
from fastmri.models import NestedUnet
from tqdm import tqdm

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

    progress_bar.close()


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, device, mask, in_chans, out_chans, chans):
    if args.unet_module == "unet":
        model = Unet(in_chans = in_chans, out_chans = out_chans, chans = chans, num_pool_layers = 4, drop_prob = 0.0)
    elif args.unet_module == "nestedunet":
        model = NestedUnet(in_chans = in_chans, out_chans = out_chans, chans = chans, num_pool_layers = 4, drop_prob = 0.0)

    pretrained_dict = torch.load(state_dict_file, map_location=device)
    model_dict = model.state_dict()
    if args.fine_tuned_model:
        if 'state_dict' in pretrained_dict.keys():
            model_dict = { k: pretrained_dict["state_dict"][f"unet.{k}"] for k, _ in model_dict.items()} # load from .ckpt
        elif 'unet' in pretrained_dict.keys():
            model_dict = {k: pretrained_dict["unet." + k] for k, v in model_dict.items()} # load from .torch
        else:
            model_dict = {k: pretrained_dict[k] for k, v in model_dict.items()}  # load from .pt
    else:
        if args.unet_module == "unet":
            model_dict = {k: pretrained_dict["classy_state_dict"]["base_model"]["model"]["trunk"][
                "_feature_blocks.unetblock." + k] for k, _ in model_dict.items()}
        elif args.unet_module == "nestedunet":
            model_dict = {
                k: pretrained_dict["classy_state_dict"]["base_model"]["model"]["trunk"]["_feature_blocks.nublock." + k]
                for k, v in model_dict.items()}

    model.load_state_dict(model_dict)
    model = model.eval()

    # data loader setup
    if "_mc" in challenge:
        data_transform = T.UnetDataTransform(which_challenge="multicoil", mask_func=mask)
    else:
        data_transform = T.UnetDataTransform(which_challenge="singlecoil", mask_func=mask)

    if "_mc" in challenge:
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            challenge="multicoil",
        )
    else:
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            challenge="singlecoil",
        )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)

        outputs[fname].append((slice_num, output))

    # save outputs
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="unet_knee_sc",
        choices=(
            "unet_knee_sc",
            "unet_knee_mc",
            "unet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--fine_tuned_model",
        action="store_true",
        help = "Convert on loading VISSL model",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )
    
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
    # unet module arguments
    parser.add_argument(
        "--unet_module",
        default = "unet",
        choices = ("unet", "nestedunet"),
        type = str,
        help = "Unet module to run with",
    )

    args = parser.parse_args()

    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
        mask,
        args.in_chans,
        args.out_chans,
        args.chans
    )

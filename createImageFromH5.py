import h5py
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

def produceImages(h5file, outputpath, index):
    if not outputpath.exists():
        outputpath.mkdir()
    
    hf = h5py.File(h5file, "r")
    
    data  = hf['reconstruction'][()]
    
    indexes = [index] if index > 0 else range(data.shape[0])
    for i in indexes:
        tmp = data[i,0,:,:]
        tmp = ( tmp * 255 / np.max(tmp)).astype('uint8')
        Image.fromarray(tmp).save(outputpath / f"{h5file.stem}_index{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--h5file",
        default=None,
        type=Path,
        required=True,
        help="The path to a single h5 file"
    )
    
    parser.add_argument(
        "--output_path",
        default="./images",
        type=Path,
        help="The path where the images will be stored, defailts to ./images"
    )
    
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="The index in the MRI stack to convert. -1 indicates all.  defaults to -1"
    )
    
    args = parser.parse_args()
    
    produceImages(
        args.h5file,
        args.output_path,
        args.index
    )
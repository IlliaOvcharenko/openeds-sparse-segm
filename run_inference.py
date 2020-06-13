import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import torchvision

import pandas as pd
import numpy as np

from fire import Fire
from pathlib import Path
from tqdm.cli import tqdm
from src.config import ConfigParser
from src.utils import set_global_seed


def run_inferece(
    config_filename,
    checkpoint_filename,
    output_folder,
    out_shape=(640, 400),
):
    set_global_seed(42)
    output_folder = Path(output_folder)
    
    config_parser = ConfigParser(
        config_filename, False,
        **{
            "checkpoint.filename": checkpoint_filename,
            "checkpoint.model": True,
        })
    config = config_parser()
    dataloader = config.dataloaders.test
    device = config.device
    model = config.model
    model.eval();
    
    print("Inference stage")
    filenames = []
    with torch.no_grad():
        for imgs, pos in tqdm(dataloader):
            imgs = imgs.to(device)
            outs = model(imgs)
            outs = outs.argmax(1).cpu()

            seqs = pos[0].long().tolist()
            orders = pos[1].long().tolist()

            for out, seq, order in zip(outs, seqs, orders):
                out = out.numpy().astype(np.uint8)
                out = cv2.resize(out, out_shape)
                filename = f"S_{seq}/{order}.npy"
                path = output_folder / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, out)
                filenames.append(filename)
    with open(output_folder / "output.txt", "w") as output_file:
        output_file.writelines("\n".join(filenames))

if __name__ == "__main__":
    Fire(run_inferece)

import cv2
import torch
import torchvision
import random

import pandas as pd
import numpy as np
import seaborn as sns 
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from functools import partial
from pathlib import Path
from tqdm.cli import tqdm
from sklearn.model_selection import StratifiedKFold


all_rows = partial(pd.option_context, 'display.max_rows', None, 'display.max_columns', None)

def fprint(df):
    """ full print for pandas dataframes 
    """
    with all_rows():
        print(df)

def percent_of(scores, cl=1):
    return (scores == cl).sum() / len(scores)


def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    

def load_splits(
    folds_folder,
    val_folds=[0],
    train_folds=None,
    only_train=None,
    only_val=None,
):
    folds = [int(fn.stem.split('_')[-1]) for fn in folds_folder.glob("fold_?.csv")]
    
    if train_folds is None:
        train_folds = [f for f in folds if f not in val_folds]
        
    if val_folds is None:
        train_folds = [f for f in folds if f not in train_folds]
        
    val = pd.concat([pd.read_csv(folds_folder / f"fold_{fi}.csv") for fi in val_folds])
    train = pd.concat([pd.read_csv(folds_folder / f"fold_{fi}.csv") for fi in train_folds])
    val = val.reset_index()
    train = train.reset_index()
    
    if only_train:
        return train
    if only_val:
        return val

    return train, val


EYE_MEAN = [0.2971]
EYE_STD = [0.1582]


def denormalize(img_tensor, mean=EYE_MEAN, std=EYE_STD):
    img_tensor = img_tensor.clone()
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor


def image_to_std_tensor(image, **params):
    image = torchvision.transforms.functional.to_tensor(image)
    image = torchvision.transforms.functional.normalize(image, EYE_MEAN, EYE_STD)
    return image


def mask_to_tensor(mask, **params):
    return torch.tensor(mask).long()


custom_to_std_tensor = A.Lambda(image=image_to_std_tensor, mask=mask_to_tensor)


class MaskInfo:
    def __init__(self, mask, r=0.0, b=0.0, g=0.0):
        self.value = mask
        self.r = r
        self.g = g
        self.b = b
        
def blend(origin, *masks, alpha=0.5):
    img = torchvision.transforms.functional.to_pil_image(origin)
    
    colors = None
    for mask in masks:
        if mask is not None and mask.value is not None and mask.value.sum() != 0.0:
            mask = np.array(torchvision.transforms.functional.to_pil_image(torch.cat([
                torch.stack([mask.value.float()]) * mask.r,
                torch.stack([mask.value.float()]) * mask.g,
                torch.stack([mask.value.float()]) * mask.b,
            ])))
            if colors is None:
                colors = mask
            else:
                colors += mask
    colors = Image.fromarray(colors)
    img = Image.blend(img, colors, alpha)
    return np.array(img)


def eye_blend(img, mask):
    return blend(
        denormalize(img) * torch.ones(3, *img.shape[-2:]),
        MaskInfo(mask == 1, r=1.0),
        MaskInfo(mask == 2, g=1.0),
        MaskInfo(mask == 3, b=1.0),
        alpha=0.2
    )
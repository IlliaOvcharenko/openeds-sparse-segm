import cv2
import torch
import torchvision

import pandas as pd
import numpy as np
import albumentations as A


class EyeDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode, transform=None, return_pos=False):
        self.df = df
        self.mode = mode
        self.transform = transform
        self.return_pos = return_pos
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        img = cv2.imread(item.img_filename, cv2.IMREAD_GRAYSCALE)
        mask = None        
        if self.mode in ["train", "val"]:
            mask = np.load(item.mask_filename)
            
        if self.transform is not None:
            transformed = self.transform(
                image=img,
                mask=mask
            )
            img = transformed["image"]
            mask = transformed["mask"]

        output = [img, ]
        
        if mask is not None:
            output.append(mask)
        
        if self.return_pos:
            output.append((item.seq, item.order))
        
        return  output

    
    def __len__(self):
        return len(self.df)
    
    
class EyeDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self, 
        df,
        mode,
        transform=None,
        return_pos=False,
        train_df=None,
    ):
        self.df = df
        self.mode = mode
        self.transform = transform
        self.return_pos = return_pos
        self.train_df = train_df
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        img = cv2.imread(item.img_filename, cv2.IMREAD_GRAYSCALE)
        
        if self.train_df is not None:
            train_masks_df = self.train_df[
                (self.train_df.seq == item.seq) &
                (self.train_df.order != item.order) & 
                (self.train_df.mask_filename.notna())
            ]
            train_masks = []
            for train_mask_filename in train_masks_df.mask_filename:
                train_masks.append(np.load(train_mask_filename))
            train_masks_channel = np.mean(train_masks, axis=0)
            train_masks_channel = (train_masks_channel * 50).astype(np.uint8)
            img = np.transpose(np.stack((img, train_masks_channel)), (1, 2, 0))
        
        mask = None        
        if self.mode in ["train", "val"]:
            mask = np.load(item.mask_filename)
            
        if self.transform is not None:
            transformed = self.transform(
                image=img,
                mask=mask
            )
            img = transformed["image"]
            mask = transformed["mask"]

        output = [img, ]
        
        if mask is not None:
            output.append(mask)
        
        if self.return_pos:
            output.append((item.seq, item.order))
        
        return  output

    
    def __len__(self):
        return len(self.df)


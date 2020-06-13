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

import torch 
import torchvision

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp


class ModelEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or np.ones(len(models))
    
    def __call__(self, x):
        res = []
        for model, weight in zip(self.models, self.weights):
            res.append(model(x)*weight)
        res = torch.stack(res)
        return torch.sum(res, dim=0) /sum(self.weights)
    
    def eval(self):
        for model in self.models:
            model.eval()


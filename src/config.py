# TODO 
# - move to AttrDict inside parse 
# - problem with access before declare 
import yaml
import torch
import torchvision
import pytorch_toolbelt

import pandas as pd
import numpy as np
import src.models as models
import src.metrics as metrics
import src.utils as utils
import src.data as data
import albumentations as A

from attrdict import AttrDict
from pprint import pprint
from pathlib import Path
from types import ModuleType
from pytorch_toolbelt import losses as L
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter


class ConfigParser():
    def __init__(self, cfg_fn, cfg_print=False, **kwargs):
        self.config_fn = Path(cfg_fn)
        self.config = yaml.load(open(self.config_fn), Loader=yaml.Loader)
        self.update_config(kwargs)
        self.print_config(cfg_print)
        self.parse()

    def __call__(self):
        return AttrDict(self.config)

    def update_config(self, params):
        # TODO cretate new config file in case of some changes
        for k, v in params.items():
            *path, key = k.split(".")
            conf = self.config
            for p in path:
                if p not in conf or conf[p] is None:
                    conf[p] = {}
                    print(f"Overwriting non-existing attribute {k} = {v}")
                conf = conf[p]
                
            if key not in conf:
                print(f"Overwriting non-existing attribute {k} = {v}")
            else:
                print(f"Overwriting {k} = {v} (was {conf.get(key)})")
                
            conf[key] = v

    def print_config(self, cfg_print):
        if cfg_print:
            print(self.config)

    @classmethod
    def get_module_attr(cls, module_from, path):
        if isinstance(module_from, dict):
            loaded_attr = module_from[path[0]]
        else:
            loaded_attr = getattr(module_from, path[0])
        if len(path) > 1:
            return cls.get_module_attr(loaded_attr, path[1:])
        return loaded_attr

    @classmethod
    def config_to_object(cls, config, *args, deep=True, **kwargs):
        if isinstance(config, dict) and "name" in config:
            if not "params" in config:
                params = {}
            else:
                params = config["params"]
            name = config["name"]
            name = name.split(".")
            
            if deep:
                for k in params:
                    params[k] = cls.config_to_object(params[k])
                
            obj_class = cls.get_module_attr(globals(), name)
            params.update(kwargs)
            obj = obj_class(*args, **params)
        elif isinstance(config, dict) and "replace" in config:
            replace = config["replace"]
            replace = replace.split(".")
            obj = cls.get_module_attr(globals(), replace)
        elif isinstance(config, dict) and "path" in config:
            obj = Path(config["path"])
        elif isinstance(config, list):
            obj = [cls.config_to_object(item) for item in config]
        else:
            obj = config
        return obj

    def parse(self):
        for attr_name in self.config:
            method_name = "parse_" + attr_name
            if hasattr(self, method_name):
                method_to_call = getattr(self, method_name)
                self.config[attr_name] = method_to_call(self.config[attr_name])

    def parse_model(self, model_config):
        model = self.config_to_object(model_config)
        model.train()
        return model

    def parse_device(self, device_config):
        use_gpu = device_config == "gpu"
        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.config["model"].to(device)
        return device

    def parse_criterion(self, criterion_config):
        criterion = self.config_to_object(criterion_config)
        return criterion

    def parse_optimizer(self, optimizer_config):
        optimizer = self.config_to_object(optimizer_config, self.config["model"].parameters())
        return optimizer

    def parse_scheduler(self, scheduler_config):
        if "params" in scheduler_config and "after_scheduler" in scheduler_config["params"]:
            after_scheduler  = self.config_to_object(
                scheduler_config["params"]["after_scheduler"],
                self.config["optimizer"]
            )
            scheduler = self.config_to_object(
                scheduler_config,
                self.config["optimizer"],
                after_scheduler=after_scheduler,
                deep=False
            )
        else:
            scheduler = self.config_to_object(scheduler_config, self.config["optimizer"])
        return scheduler

    def parse_dataloaders(self, dataloaders_config):
        for k in dataloaders_config:
            dataloaders_config[k] = self.config_to_object(dataloaders_config[k])
        return dataloaders_config

    def parse_metrics(self, metrics_config):
        for k in metrics_config:
            metrics_config[k] = self.config_to_object(metrics_config[k])
        return metrics_config

    def parse_model_folder(self, model_folder_config):
        return self.config_to_object(model_folder_config)

    def parse_model_name(self, model_name_config):
        # TODO
        # here to implement complex logic about creating model name automaticaly
        # base on model, time and so on 
        model_name = self.config_fn.stem if model_name_config is None else model_name_config
        return model_name

    def parse_log_folder(self, log_folder_config):
        return self.config_to_object(log_folder_config)

    def parse_writer(self, writer_config):

        if writer_config and "log_folder" in self.config:
            writer_folder = (self.config["log_folder"] / self.config["model_name"])
            if not writer_folder.exists():
                writer_folder.mkdir()
            writer = SummaryWriter(writer_folder)
            return writer 
        else:
            return None

    def parse_checkpoint(self, checkpoint_config):
        if checkpoint_config is not None:
            checkpoint_config["filename"] = self.config_to_object(checkpoint_config["filename"])
            checkpoint = torch.load(checkpoint_config["filename"], map_location=torch.device("cpu"))

            load_model = "model" in checkpoint_config and checkpoint_config["model"] 
            load_optimizer = "optimizer" in checkpoint_config and checkpoint_config["optimizer"]
            load_scheduler = "scheduler" in checkpoint_config and checkpoint_config["scheduler"]
            load_epoch = "initial_epoch" in checkpoint_config and checkpoint_config["initial_epoch"]
            
            if load_model and "model_state_dict" in checkpoint:
                self.config["model"].load_state_dict(checkpoint["model_state_dict"])
                print("Load model from state dict")
            elif load_model:
                self.config["model"].load_state_dict(checkpoint)
                print("Load model from checkpoint")

            if load_optimizer and "optimizer_state_dict" in checkpoint: 
                self.config["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
                print("Load optimizer")

            if load_scheduler and "scheduler_state_dict" in checkpoint:    
                self.config["scheduler"].load_state_dict(checkpoint["scheduler_state_dict"])
                print("Load scheduler")

            if load_epoch and "epoch" in checkpoint:
                self.config["initial_epoch"] = checkpoint["epoch"]
                print("Load initial epoch")
        # TODO return checkpoint itself, or it take extra memory?
        return checkpoint_config

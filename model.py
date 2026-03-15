import os
import torch
import torch.nn.functional as F
from torch import nn, cat, optim
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
import utility
import random

class ai23(pl.LightningModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config.GlobalConfig
        self.gpu_device = device
        self.automatic_optimization = True

    def forward(self, rgbs, pointcloud_xs, pointcloud_zs, rp1, rp2, velocity):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
        # optima = optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.1, patience=3, min_lr=1e-6)
        # return [
        #     {"optimizer": optima, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss"}},
        # ]
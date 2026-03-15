import os
from collections import deque
import sys
import numpy as np
import tqdm
import torch
from torch import nn, cat
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import model
import dataloader
import config

pl.seed_everything(42, workers=True)
epoch = 50

early_stop = EarlyStopping(
        monitor='val_wp',
        mode='min',
        patience=7,
        min_delta=0.001,
        verbose=True,
        check_finite=True
    )

model = model.ai23(config=config, device="cuda")
dataloader = dataloader.KARR_DataModule(sequences_length=config.GlobalConfig.seq_len, config=config)
trainer = pl.Trainer(accelerator="cuda", devices=1, precision="32-true", max_epochs=epoch, enable_model_summary=True, callbacks=[early_stop])

trainer.fit(model, dataloader)
trainer.validate(model, dataloader)
trainer.test(model, dataloader)

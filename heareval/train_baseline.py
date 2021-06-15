"""
Train the baseline model on a single multiclass task.
"""

import os.path
from typing import List

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from csvdataset import CSVDataset
from torch.utils.data import DataLoader, random_split

import heareval.baseline
from heareval.baseline import RandomProjectionMelEmbedding

# from importlib import import_module

TASK = "coughvid-v2.0.0"

# For this task
# TODO: Don't hardcode
FRAME_RATE = 8
SAMPLE_LENGTH_SECONDS = 8.0
N_LABELS = 3

# TODO: Support for multiple GPUs?
device = "gpu" if torch.cuda.is_available() else "cpu"

# +
VALIDATION = 0.2

AUDIO_BATCH_SIZE = 64
# FRAME_BATCH_SIZE = 64
EMBEDDING_SIZE = 4096

# There should be a way to use this code both for fine-tuning and non-fined-prediction
FINE_TUNE = True

# +
# TODO: Use validation if it exists
# Not sure why I have to add .. to the path
full_train_dataset = CSVDataset(
    os.path.join("../tasks", TASK, "train.csv"), labels_as_ints=True
)
n_validation = int(round(len(full_train_dataset) * VALIDATION))
train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [len(full_train_dataset) - n_validation, n_validation],
    generator=torch.Generator().manual_seed(42),
)
test_dataset = CSVDataset(
    os.path.join("../tasks", TASK, "test.csv"), labels_as_ints=True
)

train_dataloader = DataLoader(
    train_dataset, batch_size=AUDIO_BATCH_SIZE, shuffle=True, drop_last=False
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=AUDIO_BATCH_SIZE, shuffle=False, drop_last=False
)
test_dataloader = DataLoader(
    test_dataset, batch_size=AUDIO_BATCH_SIZE, shuffle=False, drop_last=False
)

# -


class MulticlassEmbedding(pl.LightningModule):
    SR = heareval.baseline.input_sample_rate()

    def __init__(self):
        super().__init__()
        self.embedding = RandomProjectionMelEmbedding()
        # Add one because the start and end are framed
        self.embedding_dim = int(
            (FRAME_RATE * SAMPLE_LENGTH_SECONDS + 1) * EMBEDDING_SIZE
        )
        self.fc = torch.nn.Linear(self.embedding_dim, N_LABELS)
        self.loss = torch.nn.CrossEntropyLoss()
        # self.squash = torch.nn.Tanh()

    def embeddings_from_filenames(self, filenames, split):
        # Might want to make this an option in CSVDataset
        audio = []
        for f in filenames:
            x, sr = sf.read(
                os.path.join(os.path.join("../tasks", TASK, str(self.SR), split, f)),
                dtype=np.float32,
            )
            assert sr == self.SR
            audio.append(x)
        audio = torch.tensor(np.vstack(audio), device=device)

        embeddings = heareval.baseline.get_audio_embedding(
            audio,
            self.embedding,
            frame_rate=FRAME_RATE,
            disable_gradients=not FINE_TUNE,
        )[0][EMBEDDING_SIZE]
        return embeddings.view(len(filenames), -1)

    def forward(self, filenames, split):
        embeddings = self.embeddings_from_filenames(filenames, split)
        # embeddings = self.squash(embeddings)
        y_hat = self.fc(embeddings)
        return y_hat

    def training_step(self, batch, batch_idx):
        filenames, y = batch

        y_hat = self.forward(filenames, split="train")
        loss = self.loss(y_hat, y.flatten())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        filenames, y = batch

        # If we did an 80/20 train/validation split,
        # the audio files are still in the "train" directory
        y_hat = self.forward(filenames, split="train")
        loss = self.loss(y_hat, y.flatten())
        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %load_ext tensorboard
# %tensorboard --logdir lightning_logs

# +
# init model
model = MulticlassEmbedding()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer()
trainer.fit(model, train_dataloader, validation_dataloader)
# -

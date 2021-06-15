"""
Train the baseline model on a single multiclass task.

TODO: Some of the boilerplate code is shared with task_embeddings.py
and can be cleaned up.
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
FRAME_BATCH_SIZE = 64
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

        # TODO: Lame that we copy all this code from get_audio_embedding
        # I'm concerned that if we have module functions and
        # not class functions in the model that we won't be able
        # to backprop thru it

        # Split the input audio signals into frames and then flatten to create a tensor
        # of audio frames that can be batch processed. We will unflatten back out to
        # (audio_baches, num_frames, embedding_size) after creating embeddings.
        frames, timestamps = heareval.baseline.frame_audio(
            audio,
            frame_size=self.embedding.n_fft,
            frame_rate=FRAME_RATE,
            sample_rate=self.SR,
        )
        audio_batches, num_frames, frame_size = frames.shape
        frames = frames.flatten(end_dim=1)

        # We're using a DataLoader to help with batching of frames
        dataset = torch.utils.data.TensorDataset(frames)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=FRAME_BATCH_SIZE, shuffle=False, drop_last=False
        )

        # Put the model into eval mode, and don't compute any gradients.
        model.eval()
        with torch.no_grad():
            # Iterate over all batches and accumulate the embeddings
            list_embeddings: List[Tensor] = []
            for batch in loader:
                result = self.embedding(batch[0])
                list_embeddings.append(result[EMBEDDING_SIZE])

        embeddings = torch.cat(list_embeddings, dim=0)
        embeddings = embeddings.unflatten(0, (audio_batches, num_frames))
        embeddings = embeddings.view(len(filenames), -1)
        return embeddings

    def forward(self, filenames, split):
        embeddings = self.embeddings_from_filenames(filenames, split)
        y_hat = self.fc(embeddings)
        return y_hat

    def training_step(self, batch, batch_idx):
        filenames, y = batch

        y_hat = self.forward(filenames, split="train")
        loss = self.loss(y_hat, y.flatten())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
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

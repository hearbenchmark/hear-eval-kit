"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

import math
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

from nnAudio import Spectrogram
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class RandomProjectionMelEmbedding(torch.nn.Module):
    n_fft = 4096
    n_mels = 256
    sample_rate = 44100
    seed = 0
    epsilon = 1e-4

    def __init__(self):
        super().__init__()
        torch.random.manual_seed(self.seed)

        # Create a Hann window buffer to apply to frames prior to FFT.
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Create a mel filter buffer.
        mel_scale: Tensor = torch.tensor(
            librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        )
        self.register_buffer("mel_scale", mel_scale)

        # Projection matrices.
        normalization = math.sqrt(self.n_mels)
        self.emb4096 = torch.nn.Parameter(torch.rand(self.n_mels, 4096) / normalization)
        self.emb2048 = torch.nn.Parameter(torch.rand(self.n_mels, 2048) / normalization)
        self.emb512 = torch.nn.Parameter(torch.rand(self.n_mels, 512) / normalization)
        self.emb128 = torch.nn.Parameter(torch.rand(self.n_mels, 128) / normalization)
        self.emb20 = torch.nn.Parameter(torch.rand(self.n_mels, 20) / normalization)

        # An activation to squash the 20D embedding to a [0, 1] range.
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: Tensor):
        # Compute the real-valued Fourier transform on windowed input signal.
        x = torch.fft.rfft(x * self.window)

        # Convert to a power spectrum.
        x = torch.abs(x) ** 2.0

        # Apply the mel-scale filter to the power spectrum.
        x = torch.matmul(x, self.mel_scale.transpose(0, 1))

        # Convert to a log mel spectrum.
        x = torch.log(x + self.epsilon)

        # Apply projections to get all required embeddings
        x4096 = x.matmul(self.emb4096)
        x2048 = x.matmul(self.emb2048)
        x512 = x.matmul(self.emb512)
        x128 = x.matmul(self.emb128)
        x20 = x.matmul(self.emb20)

        # The 20-dimensional embedding is specified to be int8. To cast to int8 we'll
        # apply an activation to ensure the embedding is in a 0 to 1 range first.
        x20 = self.activation(x20)

        # Scale to int8 value range and cast to int
        int8_max = torch.iinfo(torch.int8).max
        int8_min = torch.iinfo(torch.int8).min
        x20 = x20 * (int8_max - int8_min) + int8_min
        x20 = x20.type(torch.int8)

        return {4096: x4096, 2048: x2048, 512: x512, 128: x128, 20: x20}


class CBRBlock(nn.Module):
    def __init__(self, inputlayer, outputlayer, maxpool=False):
        super(CBRBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(inputlayer, outputlayer, 3)
        self.batchnorm_1 = nn.BatchNorm2d(outputlayer)

        self.conv2d_2 = nn.Conv2d(outputlayer, outputlayer, 3)
        self.batchnorm_2 = nn.BatchNorm2d(outputlayer)

        self.maxpool = maxpool
        if maxpool:
            self.maxpool2d = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = F.relu(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = F.relu(x)
        if self.maxpool:
            x = self.maxpool2d(x)
        return x


class Openl3Convnet(nn.Module):
    n_fft = 4096
    n_mels = 256
    sample_rate = 44100

    def __init__(self):
        super(Openl3Convnet, self).__init__()

        self.mellayer = Spectrogram.MelSpectrogram(
            sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=242,
        )
        self.batch_normalization_input = nn.BatchNorm2d(1)

        self.block1 = CBRBlock(1, 64, True)
        self.block2 = CBRBlock(64, 128, True)
        self.block3 = CBRBlock(128, 256, True)

        self.block4 = CBRBlock(256, 512)

        self.pool = nn.MaxPool2d((8, 2))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.mellayer(x).unsqueeze(1)
        x = self.batch_normalization_input(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return {2048: x}


def input_sample_rate(model_name: Optional[str] = "Openl3") -> int:
    """
    Returns:
        One of the following values: [16000, 22050, 44100, 48000].
            To avoid resampling on-the-fly, we will query your model
            to find out what sample rate audio to provide it.
    """
    if "openl3" in model_name:
        return Openl3Convnet.sample_rate
    else:
        return RandomProjectionMelEmbedding.sample_rate


def load_model(
    model_file_path: str, device: str = "cpu", model_name: Optional[str] = "Openl3"
) -> Any:
    """
    In this baseline, we don't load anything from disk.

    Args:
        model_file_path: Load model checkpoint from this file path.
            device: For inference on machines with multiple GPUs,
            this instructs the participant which device to use. If
            “cpu”, the CPU should be used (Multi-GPU support is not
            required).
    Returns:
        Model
    """
    if "openl3" in model_name:
        return Openl3Convnet().to(device)
    else:
        return RandomProjectionMelEmbedding().to(device)


def frame_audio(
    audio: Tensor, frame_size: int, frame_rate: float, sample_rate: int
) -> Tuple[Tensor, Tensor]:
    """
    Slices input audio into frames that are centered and occur every
    sample_rate / frame_rate samples. If sample_rate is not divisible
    by frame_rate, we round to the nearest sample.

    Args:
            audio: input audio, expects a 2d Tensor of shape:
            (batch_size, num_samples)
        frame_size: the number of samples each resulting frame should be
        frame_rate: number of frames per second of audio
        sample_rate: sampling rate of the input audio

    Returns:
        - A Tensor of shape (batch_size, num_frames, frame_size)
        - A 1d Tensor of timestamps corresponding to the frame
        centers.
    """
    audio = F.pad(audio, (frame_size // 2, frame_size - frame_size // 2))
    num_padded_samples = audio.shape[1]

    frame_number = 0
    frames = []
    timestamps = []
    frame_start = 0
    frame_end = frame_size
    while True:
        frames.append(audio[:, frame_start:frame_end])
        timestamps.append(frame_number / frame_rate)

        # Increment the frame_number and break the loop if the next frame end
        # will extend past the end of the padded audio samples
        frame_number += 1
        frame_start = int(round(sample_rate * frame_number / frame_rate))
        frame_end = frame_start + frame_size

        if not frame_end <= num_padded_samples:
            break

    return torch.stack(frames, dim=1), torch.tensor(timestamps)


def get_audio_embedding(
    audio: Tensor,
    model: Union[Openl3Convnet, RandomProjectionMelEmbedding],
    frame_rate: float,
    batch_size: Optional[int] = 512,
) -> Tuple[Dict[int, Tensor], Tensor]:

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (batch_size, num_samples)"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, Openl3Convnet) or not isinstance(
        model, RandomProjectionMelEmbedding
    ):
        raise ValueError(
            f"Model must be an instance of {Openl3Convnet.__name__} or {RandomProjectionMelEmbedding.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed. We will unflatten back out to
    # (audio_baches, num_frames, embedding_size) after creating embeddings.
    frames, timestamps = frame_audio(
        audio,
        frame_size=model.n_fft,
        frame_rate=frame_rate,
        sample_rate=Openl3Convnet.sample_rate,
    )
    audio_batches, num_frames, frame_size = frames.shape
    frames = frames.flatten(end_dim=1)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and don't compute any gradients.
    model.eval()

    with torch.no_grad():
        # Iterate over all batches and accumulate the embeddings
        list_embeddings: DefaultDict[int, List[Tensor]] = defaultdict(list)
        for batch in loader:
            result = model(batch[0])
            for size, embedding in result.items():
                list_embeddings[size].append(embedding)

    # Concatenate mini-batches back together and unflatten the frames back
    # to audio batches
    embeddings: Dict[int, Tensor] = {}
    for size, embedding in list_embeddings.items():
        embeddings[size] = torch.cat(embedding, dim=0)
        embeddings[size] = embeddings[size].unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def pairwise_distance(emb1: Tensor, emb2: Tensor) -> Tensor:
    """
    Note that if you are calling this with the 20-dim int8,
    you should cast them to .float() first.

    Args:
        emb1: Tensor of shape (n_samples1, n_frames, emb_dimension)
        emb2: Tensor of shape (n_samples2, n_frames, emb_dimension)

    Returns:
        Pairwise distance tensor (n_samples1, n_samples2).
        Unnormalized l1.
    """
    assert emb1.ndim == 3
    assert emb1.shape == emb2.shape
    # Flatten each embedding across frames
    emb1 = emb1.view(emb1.shape[0], -1)
    emb2 = emb2.view(emb2.shape[0], -1)
    # Compute the pairwise 1-norm distance
    d = torch.cdist(emb1, emb2, p=1.0)
    assert d.shape == (emb1.shape[0], emb2.shape[0])
    return d


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model = load_model("", device=device)
    # White noise
    audio = torch.rand(1024, 20000, device=device) * 2 - 1
    embs, timestamps = get_audio_embedding(
        audio=audio, model=model, frame_rate=Openl3Convnet.sample_rate / 1000,
    )


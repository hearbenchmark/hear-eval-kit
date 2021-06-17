"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

import math
from typing import Optional, Tuple

import librosa
import torch
import torch.nn.functional as F
from torch import Tensor


class RandomProjectionMelEmbedding(torch.nn.Module):
    # sample rate, embedding size, and version number are
    # required model attributes for the HEAR API
    # TODO how do we want to handle version?
    sample_rate = 44100
    embedding_size = 4096
    version = "0.0.1"

    # These attributes are specific to this baseline model
    n_fft = 4096
    n_mels = 256
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
        self.projection = torch.nn.Parameter(
            torch.rand(self.n_mels, self.embedding_size) / normalization
        )

    def forward(self, x: Tensor):
        # Compute the real-valued Fourier transform on windowed input signal.
        x = torch.fft.rfft(x * self.window)

        # Convert to a power spectrum.
        x = torch.abs(x) ** 2.0

        # Apply the mel-scale filter to the power spectrum.
        x = torch.matmul(x, self.mel_scale.transpose(0, 1))

        # Convert to a log mel spectrum.
        x = torch.log(x + self.epsilon)

        # Apply projection to get a 4096 dimension embedding
        embedding = x.matmul(self.projection)

        return embedding


def load_model(model_file_path: str = "", device: str = "cpu") -> torch.nn.Module:
    """
    In this baseline, we don't load anything from disk.

    Args:
        model_file_path: Load model checkpoint from this file path. For this baseline,
            if no path is provided then the default random init weights for the
            linear projection layer will be used.
        device: For inference on machines with multiple GPUs,
            this instructs the participant which device to use. If
            “cpu”, the CPU should be used (Multi-GPU support is not
            required).
    Returns:
        Model: torch.nn.Module loaded on the specified device.
    """

    # We would expect that model_file_path is the location of a saved model containing
    # model weights that we would reload. For the baseline model we have a non-learned
    # 'basic' version, so that can be specified. If a filename is passed in then the
    # the learned version of the baseline will be loaded.
    if model_file_path == "":
        model = RandomProjectionMelEmbedding().to(device)
    else:
        # TODO: implement loading weights from disk
        raise NotImplementedError("Loading model weights not implemented yet")

    return model


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
    model: torch.nn.Module,
    frame_rate: float,
    batch_size: Optional[int] = 512,
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        audio: n_sounds x n_samples of mono audio in the range
            [-1, 1]. We are making the simplifying assumption that
            for every task, all sounds will be padded/trimmed to
            the same length. This doesn’t preclude people from
            using the API for corpora of variable-length sounds;
            merely we don’t implement that as a core feature. It
            could be a wrapper function added later.
        model: Loaded model, in PyTorch or Tensorflow 2.x. This
            should be moved to the device the audio tensor is on.
        frame_rate: Number of embeddings that the model should return
            per second. Embeddings and the corresponding timestamps should
            start at 0s and increment by 1/frame_rate seconds. For example,
            if the audio is 1.1s and the frame_rate is 4.0, then we should
            return embeddings centered at 0.0s, 0.25s, 0.5s, 0.75s and 1.0s.
        batch_size: The participants are responsible for estimating
            the batch_size that will achieve high-throughput while
            maintaining appropriate memory constraints. However,
            batch_size is a useful feature for end-users to be able to
            toggle.

    Returns:
        - Tensor: Embeddings, `(n_sounds, n_frames, embedding_size)`.
        - Tensor: Frame-center timestamps, 1d.
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (batch_size, num_samples)"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, RandomProjectionMelEmbedding):
        raise ValueError(
            f"Model must be an instance of {RandomProjectionMelEmbedding.__name__}"
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
        sample_rate=RandomProjectionMelEmbedding.sample_rate,
    )
    audio_batches, num_frames, frame_size = frames.shape
    frames = frames.flatten(end_dim=1)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings_list = [model(batch[0]) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

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

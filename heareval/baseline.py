"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

import math
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import librosa


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
        mel_scale = Tensor(
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


def input_sample_rate() -> int:
    """
    Returns:
        One of the following values: [16000, 22050, 44100, 48000].
            To avoid resampling on-the-fly, we will query your model
            to find out what sample rate audio to provide it.
    """
    return RandomProjectionMelEmbedding.sample_rate


def load_model(model_file_path: str, device: str = "cpu") -> Any:
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
    return RandomProjectionMelEmbedding().to(device)


def frame_audio(audio: Tensor, frame_size: int, frame_rate: float, sample_rate: int):
    """
    Slice audio into equal length frames. Each adjacent frame is a hop_size number of
    samples apart.

    Args:
        audio: input audio, expects a 2d Tensor of shape: (batch_size, num)samples)
        frame_size: the length each resulting frame should be
        hop_size: number of samples between frames.
        is_centered: Pad audio by frame_size // 2 to center audio in frame

    Returns:
        # TODO also return the timestamps
        A Tensor of shape (batch_size, num_frames, frame_size)
    """
    audio = F.pad(audio, (frame_size // 2, frame_size - frame_size // 2))
    num_padded_samples = audio.shape[1]

    frame_number = 0
    frames = []
    frame_start = 0
    frame_end = frame_size
    while frame_end < num_padded_samples:
        frame_start = int(round(sample_rate * frame_number / frame_rate))
        frame_end = frame_start + frame_size
        frames.append(audio[:, frame_start:frame_end])
        frame_number += 1

    return torch.stack(frames, dim=1)


def get_timestamps_for_embedding(
    sample_rate: int,
    num_frames: int,
    hop_size: int,
    frame_size: Optional[int] = None,
    center: Optional[bool] = True,
) -> Tensor:
    """
    Returns a tensor of timestamps in seconds that correspond to the time
    locations of each embedding.

    Args:
        sample_rate: audio sampling rate of input audio
        num_frames: number of frames to compute timestamps for
        hop_size: distance between adjacent frames used to calculate
        frame_size: used to calculate time offset when center is False
        center: whether of not the frames were padded to center audio, if false
            an offset will be applied equal to frame_size // 2.

    Returns:
        Timestamps in seconds.
    """

    offset = 0
    if not center:
        if frame_size is None:
            raise ValueError(
                "When computing timestamps for non-centered frames, "
                "the frame_size must be provided."
            )
        offset = int(frame_size // 2)

    timestamps = (torch.arange(0, num_frames) * hop_size + offset) / sample_rate
    return timestamps


def get_audio_embedding(
    audio: Tensor,
    model: RandomProjectionMelEmbedding,
    frame_rate: float,
    batch_size: Optional[int] = 512,
) -> Tuple[Dict[int, Tensor], Tensor]:
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
            ({embedding_size: Tensor}, list(frame timestamps)) where
            embedding_size can be any of [4096, 2048, 512, 128,
            20].  Tensor is float32 (or signed int for 20-dim),
            n_sounds x n_frames x dim.

    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            f"audio input tensor must be 2D with shape (batch_size, num_samples)"
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
    frames = frame_audio(audio, model.n_fft, frame_rate)
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
        embeddings = defaultdict(list)
        for batch in loader:
            result = model(batch[0])
            for size, embedding in result.items():
                embeddings[size].append(embedding)

    # Concatenate mini-batches back together and unflatten the frames back
    # to audio batches
    embeddings = dict(embeddings)
    for size, embedding in embeddings.items():
        embeddings[size] = torch.cat(embedding, dim=0)
        embeddings[size] = embeddings[size].unflatten(0, (audio_batches, num_frames))

    # Get timestamps in seconds.
    timestamps = get_timestamps_for_embedding(
        model.sample_rate, num_frames, hop_size, frame_size, center
    )

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
    embs, timestamps = get_audio_embedding(audio=audio, model=model, hop_size=1000)

    pairwise_distance(embs[20].float(), embs[20].float())

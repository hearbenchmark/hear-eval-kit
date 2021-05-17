"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram


class RandomProjectionMelEmbedding(torch.nn.Module):
    n_fft = 4096
    n_mels = 256
    sample_rate = 44100
    seed = 0

    def __init__(self):
        super().__init__()
        torch.random.manual_seed(self.seed)
        self.mel = MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels
        )
        self.emb4096 = torch.nn.Parameter(
            torch.rand(self.n_mels, 4096) / math.sqrt(self.n_mels)
        )
        self.emb2048 = torch.nn.Parameter(torch.rand(4096, 2048) / math.sqrt(4096))
        self.emb512 = torch.nn.Parameter(torch.rand(2048, 512) / math.sqrt(2048))
        self.emb128 = torch.nn.Parameter(torch.rand(512, 128) / math.sqrt(512))
        self.emb20 = torch.nn.Parameter(torch.rand(128, 20) / math.sqrt(128))
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: Tensor, hop_size_samples: int, center: bool):
        self.mel.hop_length = hop_size_samples
        self.mel.center = center
        self.mel.spectrogram.hop_length = hop_size_samples
        self.mel.spectrogram.center = center
        x = torch.log(self.mel(x) + 1e-4)
        x = x.swapaxes(1, 2)
        x4096 = x.matmul(self.emb4096)
        x2048 = x4096.matmul(self.emb2048)
        x512 = x2048.matmul(self.emb512)
        x128 = x512.matmul(self.emb128)
        x20 = x128.matmul(self.emb20)
        # [0, 1] range
        x20 = self.activation(x20)
        # Convert x20 to int8.
        newmax = torch.iinfo(torch.int8).max
        newmin = torch.iinfo(torch.int8).min
        x20 = torch.tensor(
            (x20 * (newmax - newmin) + newmin), dtype=torch.int8, device=x20.device
        )
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


def get_audio_embedding(
    audio: Tensor,
    model: Any,
    hop_size_samples: int,
    batch_size: Optional[int] = None,
    center: bool = True,
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
        hop_size: Number of audio samples between adjacent frames
            batch_size: The participants are responsible for estimating
        the batch_size that will achieve high-throughput while
        maintaining appropriate memory constraints. However,
        batch_size is a useful feature for end-users to be able to
        toggle.
            center: If True, the timestamps correspond to the center
            of each analysis window. center=True will be used for
            all evaluation tasks.
    Returns:
            ({embedding_size: Tensor}, list(frame timestamps)) where
            embedding_size can be any of [4096, 2048, 512, 128,
            20].  Tensor is float32 (or signed int for 20-dim),
            n_sounds x n_frames x dim.

    """
    assert audio.ndim == 2
    model = model.to(audio.device)

    # Implement batching of the audio
    if batch_size is None:
        # Here we just pick a sensible default batch size
        batch_size = 512
    dataset = torch.utils.data.TensorDataset(audio)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    if center:
        timestamps = (
            torch.range(0, audio.shape[1], hop_size_samples) / input_sample_rate()
        )
    else:
        # We will only use center=True in HEAR 2021
        raise ValueError("center = False not supported")

    # Put the model into eval mode, and don't compute any gradients.
    model.eval()
    with torch.no_grad():
        # Iterate over all batches and accumulate the embeddings
        # into allembs
        allembs = defaultdict(list)
        for batch in loader:
            # The dataset only has one element, which is the audio
            # batch tensor
            embs = model(batch[0], hop_size_samples=hop_size_samples, center=center)
            for e in embs:
                allembs[e].append(embs[e])
    # Concatenate the minibatches before returning
    # TODO: Check that returns are the right type?
    for e in allembs:
        allembs[e] = torch.cat(allembs[e], dim=0)
        assert allembs[e].shape[0] == audio.shape[0]
        assert len(timestamps) == allembs[e].shape[1]
    return allembs, timestamps


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
        audio=audio, model=model, hop_size_samples=1000
    )

    pairwise_distance(embs[20].float(), embs[20].float())

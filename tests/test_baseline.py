"""
Tests for the baseline model
"""

import numpy as np
import torch

from heareval.model.baseline import (
    load_model,
    get_audio_embedding,
    frame_audio,
)

torch.backends.cudnn.deterministic = True


class TestEmbeddingsTimestamps:
    def setup(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model(device=self.device)
        self.sample_rate = self.model.sample_rate
        self.audio = torch.rand(64, 96000, device=self.device) * 2 - 1
        self.embeddings_ct, self.ts_ct = get_audio_embedding(
            audio=self.audio,
            model=self.model,
            frame_rate=self.sample_rate / 256,
            batch_size=512,
        )

    def teardown(self):
        del self.model
        del self.audio
        del self.embeddings_ct
        del self.ts_ct

    def test_embeddings_replicability(self):
        # Test if all the embeddings are replicable
        embeddings_ct, _ = get_audio_embedding(
            audio=self.audio,
            model=self.model,
            frame_rate=self.sample_rate / 256,
            batch_size=512,
        )

        assert torch.allclose(self.embeddings_ct, embeddings_ct)

    def test_embeddings_batched(self):
        # methodA - Pass two audios individually and get embeddings. methodB -
        # Pass the two audio in a batch and get the embeddings. All
        # corresponding embeddings by method A and method B should be similar.
        audioa = self.audio[0].unsqueeze(0)
        audiob = self.audio[1].unsqueeze(0)
        audioab = self.audio[:2]
        assert torch.all(torch.cat([audioa, audiob]) == audioab)

        embeddingsa, _ = get_audio_embedding(
            audio=audioa,
            model=self.model,
            frame_rate=self.sample_rate / 256,
            batch_size=512,
        )
        embeddingsb, _ = get_audio_embedding(
            audio=audiob,
            model=self.model,
            frame_rate=self.sample_rate / 256,
            batch_size=512,
        )
        embeddingsab, _ = get_audio_embedding(
            audio=audioab,
            model=self.model,
            frame_rate=self.sample_rate / 256,
            batch_size=512,
        )

        assert torch.allclose(torch.cat([embeddingsa, embeddingsb]), embeddingsab)

    def test_embeddings_sliced(self):
        # Slice the audio to select every even audio in the batch. Produce the
        # embedding for this sliced audio batch. The embeddings for
        # corresponding audios should match the embeddings when the full batch
        # was passed.
        audio_sliced = self.audio[::2]

        # Ensure framing is identical [.???] -> Yes ensuring that.
        audio_sliced_framed, _ = frame_audio(
            audio_sliced,
            frame_size=4096,
            frame_rate=self.sample_rate / 256,
            sample_rate=self.sample_rate,
        )
        audio_framed, _ = frame_audio(
            self.audio,
            frame_size=4096,
            frame_rate=self.sample_rate / 256,
            sample_rate=self.sample_rate,
        )
        assert torch.all(audio_sliced_framed == audio_framed[::2])

        # Test for centered
        embeddings_sliced, _ = get_audio_embedding(
            audio=audio_sliced,
            model=self.model,
            frame_rate=self.sample_rate / 256,
            batch_size=512,
        )

        assert torch.allclose(embeddings_sliced, self.embeddings_ct[::2])

    def test_embeddings_shape(self):
        # Test the embeddings shape.
        # The shape returned is (batch_size, num_frames, embedding_size). We expect
        # num_frames to be equal to the number of full audio frames that can fit into
        # the audio sample. The centered example is padded with frame_size (4096) number
        # of samples, so we don't need to subtract that in that test.
        assert self.embeddings_ct.shape == (64, 96000 // 256 + 1, int(4096))

    def test_embeddings_nan(self):
        # Test for null values in the embeddings.
        assert not torch.any(torch.isnan(self.embeddings_ct))

    def test_embeddings_type(self):
        # Test the data type of the embeddings.
        assert self.embeddings_ct.dtype == torch.float32

    def test_timestamps_begin(self):
        # Test the beginning of the time stamp
        assert self.ts_ct[0] == 0

    def test_timestamps_spacing(self):
        # Test the spacing between the time stamp
        assert torch.all(torch.abs(torch.diff(self.ts_ct) - self.ts_ct[1]) < 1e-5)

    def test_timestamps_end(self):
        # Test the end of the timestamp.
        duration = self.audio.shape[1] / self.sample_rate

        # For a centered frame the difference between the end and the duration should
        # be zero (an equal number of frames fit into the padded signal, so the center
        # of the last frame should be right at the end of the input). This is just for
        # this particular input signal.
        centered_diff = duration - self.ts_ct[-1]
        assert np.isclose(centered_diff.detach().cpu().numpy(), 0.0)


class TestModel:
    def setup(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model(device=device)
        self.frames = torch.rand(512, self.model.n_fft, device=device) * 2 - 1

    def teardown(self):
        del self.model
        del self.frames

    def test_model_attributes(self):
        # Each model should have sample_rate and embedding size information
        assert hasattr(self.model, "sample_rate")
        assert hasattr(self.model, "embedding_size")
        assert hasattr(self.model, "version")

    def test_model_sliced(self):
        frames_sliced = self.frames[::2]
        assert torch.allclose(frames_sliced[0], self.frames[0])
        assert torch.allclose(frames_sliced[1], self.frames[2])
        assert torch.allclose(frames_sliced, self.frames[::2])

        outputs = self.model(self.frames)
        outputs_sliced = self.model(frames_sliced)

        assert torch.allclose(outputs_sliced[0], outputs[0])
        assert torch.allclose(outputs_sliced[1], outputs[2])
        assert torch.allclose(outputs_sliced, outputs[::2])


class TestFraming:
    def test_frame_audio(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sr = 44100
        num_audio = 16
        duration = 1.1
        frame_rate = 4.0
        frame_size = 4096

        audio = torch.rand((num_audio, int(sr * duration)), device=device)
        frames, timestamps = frame_audio(
            audio, frame_size=frame_size, frame_rate=frame_rate, sample_rate=sr
        )

        expected_frames_shape = (num_audio, 5, frame_size)
        expected_timestamps = np.arange(0.0, duration, 0.25)

        assert expected_frames_shape == frames.shape
        assert np.all(expected_timestamps == timestamps.detach().cpu().numpy())

import torch
import os
from baseline import load_model, get_audio_embedding, input_sample_rate

torch.backends.cudnn.deterministic = True


class TestEmbeddingsTimestamps:
    def __init__(self):
        self.model = load_model(
            "", device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.audio = torch.rand(64, 96000) * 2 - 1
        self.embeddings_ct, self.ts_ct = get_audio_embedding(
            audio=self.audio,
            model=self.model,
            hop_size=256,
            batch_size=512,
            center=True,
        )

        self.embeddings_not_ct, self.ts_not_ct = get_audio_embedding(
            audio=self.audio,
            model=self.model,
            hop_size=256,
            batch_size=512,
            center=False,
        )

    def test_embeddings_replicability(self):
        embeddings_ct, _ = get_audio_embedding(
            audio=self.audio,
            model=self.model,
            hop_size=256,
            batch_size=512,
            center=True,
        )
        for embeddinga, embeddingb in zip(
            self.embeddings_ct.values(), embeddings_ct.values()
        ):
            assert torch.all(torch.abs(embeddinga - embeddingb) < 1e-5)

        embeddings_not_ct, _ = get_audio_embedding(
            audio=self.audio,
            model=self.model,
            hop_size=256,
            batch_size=512,
            center=False,
        )
        for embeddinga, embeddingb in zip(
            self.embeddings_not_ct.values(), embeddings_not_ct.values()
        ):
            assert torch.all(torch.abs(embeddinga - embeddingb) < 1e-5)

    def test_embeddings_batched(self):

        audioa = self.audio[0, ...].unsqueeze(0)
        audiob = self.audio[1, ...].unsqueeze(0)
        audioab = self.audio[:2, ...]
        assert torch.all(torch.cat([audioa, audiob]) == audioab)

        for center in [True, False]:
            embeddingsa, _ = get_audio_embedding(
                audio=audioa,
                model=self.model,
                hop_size=256,
                batch_size=512,
                center=center,
            )
            embeddingsb, _ = get_audio_embedding(
                audio=audiob,
                model=self.model,
                hop_size=256,
                batch_size=512,
                center=center,
            )
            embeddingsab, _ = get_audio_embedding(
                audio=audioab,
                model=self.model,
                hop_size=256,
                batch_size=512,
                center=center,
            )
            for embeddinga, embeddingb, embeddingab in zip(
                    embeddingsa.values(), embeddingsb.values(), embeddingsab.values()):
                assert torch.all(
                    torch.abs(
                        torch.cat([embeddinga, embeddingb]) - embeddingab) < 1e-5
                )

    def test_embeddings_sliced(self):
        audio_sliced = self.audio[::2, ...]

        embeddings_sliced, _ = get_audio_embedding(
            audio=audio_sliced,
            model=self.model,
            hop_size=256,
            batch_size=512,
            center=True,
        )
        for embedding_sliced, embedding_ct in zip(
            embeddings_sliced.values(), self.embeddings_ct.values()
        ):
            assert torch.all(
                torch.abs(embedding_sliced - embedding_ct[::2, ...]) < 1e-5
            )

        embeddings_sliced, _ = get_audio_embedding(
            audio=audio_sliced,
            model=self.model,
            hop_size=256,
            batch_size=512,
            center=False,
        )
        for embedding_sliced, embedding_not_ct in zip(
            embeddings_sliced.values(), self.embeddings_not_ct.values()
        ):
            assert torch.all(
                torch.abs(embedding_sliced - embedding_not_ct[::2, ...]) < 1e-5
            )

    def test_embeddings_shape(self):
        for size, embedding in self.embeddings_not_ct.items():
            assert embedding.shape == (64, 96000 // 256, int(size))

        for size, embedding in self.embeddings_ct.items():
            assert embedding.shape == (
                64, (4096 // 2 + 96000) // 256, int(size))

    def test_embeddings_nan(self):
        for embeddings in [self.embeddings_ct, self.embeddings_not_ct]:
            for size, embedding in embeddings.items():
                assert not torch.any(torch.isnan(embedding))

    def test_embeddings_type(self):
        for embeddings in [self.embeddings_ct, self.embeddings_not_ct]:
            for size, embedding in embeddings.items():
                if size != 20:
                    assert embedding[size].dtype == torch.float32
                else:
                    assert embedding[size].dtype == torch.int8

    def test_timestamps_begin(self):
        assert self.ts_ct[0] == 0
        assert (
            torch.abs(self.ts_not_ct[0] - int(4096 //
                      2) / input_sample_rate()) < 1e-5
        )

    def test_timestamps_spacing(self):
        assert torch.all(torch.abs(torch.diff(
            self.ts_ct) - self.ts_ct[1]) < 1e-5)
        assert torch.all(
            torch.abs(
                torch.diff(self.ts_not_ct) -
                (self.ts_not_ct[2] - self.ts_not_ct[1])
            )
            < 1e-5
        )

    def test_timestamps_end(self):
        assert torch.abs(self.ts_ct[-1] - 96000 / input_sample_rate()) < 1e-5
        assert torch.abs(self.ts_not_ct[-1] -
                         96000 / input_sample_rate()) < 1e-5


class TestModel:
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model("", device=device)
        self.frames = torch.rand(512, self.model.n_fft) * 2 - 1

    def test_model_sliced(self):
        frames_sliced = self.frames[::2, ...]
        assert torch.all(frames_sliced[0] - self.frames[0] == 0)
        assert torch.all(frames_sliced[1] - self.frames[2] == 0)
        assert torch.all(frames_sliced - self.frames[::2, ...] == 0)

        outputs = self.model(self.frames)
        outputs_sliced = self.model(frames_sliced)

        for output, output_sliced in zip(
                outputs.values(), outputs_sliced.values()):
            assert torch.all(torch.abs(output_sliced[0] - output[0]) < 1e-5)
            assert torch.all(torch.abs(output_sliced[1] - output[2]) < 1e-5)
            assert torch.all(
                torch.abs(output_sliced - output[::2, ...]) < 1e-5)


class TestLayerbyLayer:
    def test_layers_find_error(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = load_model("", device=device)

        frames = torch.rand(512, model.n_fft)
        frames_sliced = frames[::2, ...]
        assert torch.all(torch.abs(frames[2] - frames_sliced[1]) == 0)

        # Layer by the layer the same operations are performed and the best
        # possible cap is set. These cap have been tested. The purpose is to
        # understand why the batched inputs are failing
        x = torch.fft.rfft(frames * model.window)
        y = torch.fft.rfft(frames_sliced * model.window)
        assert torch.all(torch.abs(x[::2, ...] - y) < 1e-25)

        x = torch.abs(x) ** 2.0
        y = torch.abs(y) ** 2.0
        assert torch.all(torch.abs(x[::2, ...] - y) < 1e-25)

        x = torch.matmul(x, model.mel_scale.transpose(0, 1))
        y = torch.matmul(y, model.mel_scale.transpose(0, 1))
        assert torch.all(torch.abs(x[::2, ...] - y) < 1e-5)

        x = torch.log(x + model.epsilon)
        y = torch.log(y + model.epsilon)
        assert torch.all(torch.abs(x[::2, ...] - y) < 1e-6)

        x4096 = x.matmul(model.emb4096)
        y4096 = y.matmul(model.emb4096)
        assert torch.all(torch.abs(x4096[::2, ...] - y4096) < 1e-4)

        x2048 = x4096.matmul(model.emb2048)
        y2048 = y4096.matmul(model.emb2048)
        assert torch.all(torch.abs(x2048[::2, ...] - y2048) < 1e-3)

        x512 = x2048.matmul(model.emb512)
        y512 = y2048.matmul(model.emb512)
        assert torch.all(torch.abs(x512[::2, ...] - y512) < 1e-2)

        x128 = x512.matmul(model.emb128)
        y128 = y512.matmul(model.emb128)
        assert torch.all(torch.abs(x128[::2, ...] - y128) < 1e-1)

        int8_max = torch.iinfo(torch.int8).max
        int8_min = torch.iinfo(torch.int8).min

        x20 = x128.matmul(model.emb20)
        x20 = model.activation(x20)
        x20 = x20 * (int8_max - int8_min) + int8_min
        x20 = x20.type(torch.int8)

        y20 = y128.matmul(model.emb20)
        y20 = model.activation(y20)
        y20 = y20 * (int8_max - int8_min) + int8_min
        y20 = y20.type(torch.int8)
        assert torch.all(torch.abs(x20[::2, ...] - y20) < 1e-1)


if __name__ == "__main__":
    # Embedding testings
    test_embedding_timestamp = TestEmbeddingsTimestamps()
    test_embedding_timestamp.test_embeddings_replicability()
    test_embedding_timestamp.test_embeddings_shape()
    test_embedding_timestamp.test_embeddings_nan()
    test_embedding_timestamp.test_timestamps_begin()
    test_embedding_timestamp.test_timestamps_spacing()

    try:
        test_embedding_timestamp.test_timestamps_end()
    except BaseException:
        print("Test Time Stamps End is not working")
    try:
        test_embedding_timestamp.test_embeddings_batched()
    except BaseException:
        print("Test Embedding Batched is not working")
    try:
        test_embedding_timestamp.test_embeddings_sliced()
    except BaseException:
        print("Test Embedding Sliced is not working")

    # Model testing to see errors at a frame level
    test_model = TestModel()
    try:
        test_model.test_model_sliced()
    except BaseException:
        print("Test Model Sliced is not working")

    # Layer by layer testing for understanding why the above are failing
    test_layerbylayer = TestLayerbyLayer()
    try:
        test_layerbylayer.test_layers_find_error()
    except BaseException:
        print("Test Layer by layer is not working")

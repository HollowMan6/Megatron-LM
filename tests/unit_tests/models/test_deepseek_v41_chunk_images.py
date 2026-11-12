# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed image placement across physical CP ranks and temporal chunks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.models.deepseek_v41.image_processing import ImageInput, image_token_types
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
from megatron.core.models.deepseek_v41.vision import Aligner, ViT
from megatron.core.packed_seq_params import PackedSeqParams


class _Group:
    """Only the immutable rank geometry is needed by the image frontend."""

    def __init__(self, size=1, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _Embedding(nn.Embedding):
    def forward(self, input_ids, position_ids):
        return super().forward(input_ids).transpose(0, 1)


class _RecordDecoder(nn.Module):
    def forward(self, hidden, attention_mask, *, forward_context, **kwargs):
        return hidden, forward_context


class _DropoutLoRALinear(nn.Module):
    """Match Bridge's dropout-before-A/B adapter, including zero-initialized B."""

    def __init__(self, base, probability):
        super().__init__()
        self.base = base
        self.dropout = nn.Dropout(probability)
        self.lora_A = nn.Linear(base.in_features, 2, bias=False)
        self.lora_B = nn.Linear(2, base.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        return self.base(hidden) + self.lora_B(self.lora_A(self.dropout(hidden)))


def _add_dropout(model, tower, probability):
    if tower == "vision":
        model.vision.patch_embed.proj = _DropoutLoRALinear(
            model.vision.patch_embed.proj, probability
        )
    else:
        model.aligner.w1 = _DropoutLoRALinear(model.aligner.w1, probability)


class _ImageFrontend(nn.Module):
    """Run the production image frontend and complete ViT without a language backbone."""

    forward = DeepSeekV41Model.forward_features
    encode_image = DeepSeekV41Model.encode_image
    _insert_image_tokens = DeepSeekV41Model._insert_image_tokens

    def __init__(self):
        super().__init__()
        args = SimpleNamespace(
            dim=8,
            vision_dim=8,
            vision_patch_size=1,
            vision_n_heads=2,
            vision_inter_dim=16,
            vision_n_layers=1,
            vision_rope_theta=10000,
            vision_downsample_ratio=1,
        )
        self.embedding = _Embedding(32, args.dim)
        self.vision = ViT(args)
        self.aligner = Aligner(args)
        self.decoder = _RecordDecoder()
        self.pg_collection = SimpleNamespace(cp=_Group())
        self.engram_hash = object()
        for name in ("image_start", "image_end", "image_newline"):
            self.register_parameter(name, nn.Parameter(torch.randn(args.dim)))

    def _hash_packed_tokens(self, input_ids, token_mask, packed_seq_params):
        # Verify the exact mask handed to the real distributed Engram hasher.
        self.hash_token_mask = token_mask.clone()
        return token_mask.to(torch.int64)


def _packed(cp_size=1, cp_rank=0, chunks=1, index=0):
    logical = torch.tensor([0, 29, 54], dtype=torch.int32)
    physical = torch.tensor([0, 32, 64], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=32,
        max_seqlen_kv=32,
        cp_partition_mode="contiguous",
        cp_group=_Group(cp_size, cp_rank),
        cp_chunk_count=chunks,
        cp_chunk_index=index,
    )


def _images():
    def image(start, height, width):
        return ImageInput(
            start,
            torch.randn(height * width, 3, 1, 1, requires_grad=True),
            height,
            width,
            image_token_types(height, width),
        )

    # The first image straddles rank and chunk cuts; later images use the second
    # sequence's padded physical prefix, rather than its logical prefix of 29.
    return [[image(5, 4, 4)], [image(1, 2, 2), image(12, 2, 3)]]


@pytest.mark.parametrize("cp_size,chunks", [(2, 1), (2, 2), (2, 8), (4, 8)])
def test_chunked_image_frontend_matches_unsplit_outputs_masks_and_gradients(cp_size, chunks):
    """Every image feature, delimiter and trainable vision gradient is preserved."""
    torch.manual_seed(123)
    reference = _ImageFrontend()
    actual = deepcopy(reference)
    images = _images()
    chunk_images = deepcopy(images)
    ids = torch.arange(64).remainder(32).unsqueeze(0)
    positions = torch.arange(64).unsqueeze(0)
    expected, expected_context = reference(
        ids, positions, images=images, packed_seq_params=_packed()
    )
    weights = torch.linspace(0.7, 1.3, expected.numel()).view_as(expected)
    weights[29:32] = 0
    weights[57:] = 0
    (expected * weights).sin().square().sum().backward()

    local_rows = ids.shape[1] // (cp_size * chunks)
    output_parts, mask_parts = [], []
    # An explicit packed CP group must override a different model default.
    actual.pg_collection.cp = _Group(8, 7)
    for chunk_index in range(chunks):
        for rank in range(cp_size):
            start = (chunk_index * cp_size + rank) * local_rows
            stop = start + local_rows
            output, context = actual(
                ids[:, start:stop],
                positions[:, start:stop],
                images=chunk_images,
                packed_seq_params=_packed(cp_size, rank, chunks, chunk_index),
            )
            expected_mask = expected_context.image_mask[:, start:stop]
            torch.testing.assert_close(context.image_mask, expected_mask)
            torch.testing.assert_close(context.token_mask, ~expected_mask)
            torch.testing.assert_close(actual.hash_token_mask, ~expected_mask)
            output_parts.append(output.detach())
            mask_parts.append(context.image_mask)
            (output * weights[start:stop]).sin().square().sum().backward()

    torch.testing.assert_close(torch.cat(output_parts), expected.detach(), atol=0, rtol=0)
    torch.testing.assert_close(torch.cat(mask_parts, dim=1), expected_context.image_mask)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, expected_parameters[name].grad, atol=2e-6, rtol=2e-5, msg=name
        )
    for name in ("vision.patch_embed.proj.weight", "aligner.w1.weight", "image_newline"):
        assert dict(actual.named_parameters())[name].grad.abs().sum() > 0, name
    for sample, expected_sample in zip(chunk_images, images):
        for image, expected_image in zip(sample, expected_sample):
            assert image.patches.grad.abs().sum() > 0
            torch.testing.assert_close(
                image.patches.grad, expected_image.patches.grad, atol=2e-6, rtol=2e-5
            )


@pytest.mark.parametrize("index", [-1, 2])
def test_packed_image_placement_rejects_invalid_chunk_index(index):
    model = _ImageFrontend()
    with pytest.raises(ValueError, match="chunk index outside"):
        model(
            torch.zeros(1, 16, dtype=torch.int64),
            torch.zeros(1, 16, dtype=torch.int64),
            images=_images(),
            packed_seq_params=_packed(2, 0, 2, index),
        )


def test_packed_image_placement_keeps_logical_span_bounds():
    model = _ImageFrontend()
    images = _images()
    images[1][1].start = 20
    with pytest.raises(ValueError, match="fit the logical THD sequence"):
        model(
            torch.zeros(1, 4, dtype=torch.int64),
            torch.zeros(1, 4, dtype=torch.int64),
            images=images,
            packed_seq_params=_packed(2, 0, 8, 7),
        )


def test_native_model_declares_chunked_image_support():
    assert DeepSeekV41Model.supports_cp_chunk_images is True


def test_empty_image_metadata_preserves_text_rows():
    model = _ImageFrontend()
    ids = torch.arange(4).unsqueeze(0)
    positions = torch.arange(4).unsqueeze(0)
    output, context = model(ids, positions, images=[[], []], packed_seq_params=_packed(2, 1, 8, 4))
    torch.testing.assert_close(output, model.embedding(ids, positions), atol=0, rtol=0)
    assert not context.image_mask.any()
    assert context.token_mask.all() and model.hash_token_mask.all()


def test_ordinary_batched_images_match_unchunked_packed_rows():
    model = _ImageFrontend()
    model.engram_hash = None
    images = _images()
    ids = torch.arange(64).remainder(32)
    batched, batched_context = model(ids.view(2, 32), torch.arange(32).expand(2, -1), images=images)
    packed, packed_context = model(
        ids.view(1, 64), torch.arange(64).unsqueeze(0), images=images, packed_seq_params=_packed()
    )
    torch.testing.assert_close(batched.transpose(0, 1).reshape_as(packed), packed, atol=0, rtol=0)
    torch.testing.assert_close(batched_context.image_mask.reshape(1, 64), packed_context.image_mask)


@pytest.mark.parametrize("tower", ["vision", "aligner"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_chunked_images_reject_stochastic_adapter_dropout(tower, grad_enabled):
    model = _ImageFrontend()
    _add_dropout(model, tower, 0.5)
    with torch.set_grad_enabled(grad_enabled):
        with pytest.raises(NotImplementedError, match="stochastic vision/aligner dropout"):
            model(
                torch.zeros(1, 4, dtype=torch.int64),
                torch.zeros(1, 4, dtype=torch.int64),
                images=_images(),
                packed_seq_params=_packed(2, 0, 8, 1),
            )


@pytest.mark.parametrize("tower", ["vision", "aligner"])
@pytest.mark.parametrize("probability,training", [(0.0, True), (1.0, True), (0.5, False)])
def test_chunked_images_allow_deterministic_dropout(tower, probability, training):
    model = _ImageFrontend()
    _add_dropout(model, tower, probability)
    model.train(training)
    output, context = model(
        torch.zeros(1, 4, dtype=torch.int64),
        torch.zeros(1, 4, dtype=torch.int64),
        images=_images(),
        packed_seq_params=_packed(2, 0, 8, 1),
    )
    assert torch.isfinite(output).all() and context.image_mask.all()


@pytest.mark.parametrize("text_only", [False, True])
def test_stochastic_dropout_preserves_unchunked_images_and_chunked_text(text_only):
    model = _ImageFrontend()
    _add_dropout(model, "vision", 0.5)
    _add_dropout(model, "aligner", 0.5)
    rows = 4 if text_only else 64
    output, context = model(
        torch.zeros(1, rows, dtype=torch.int64),
        torch.zeros(1, rows, dtype=torch.int64),
        images=[[], []] if text_only else _images(),
        packed_seq_params=_packed(2, 0, 8, 1) if text_only else _packed(),
    )
    assert torch.isfinite(output).all()
    assert context.image_mask.any().item() == (not text_only)


def test_repeated_image_encoding_changes_zero_b_adapter_gradients_with_dropout():
    """Identical initial features do not imply equivalent gradients with random masks."""
    torch.manual_seed(123)
    reference = _ImageFrontend()
    _add_dropout(reference, "vision", 0.5)
    actual = deepcopy(reference)
    image = _images()[0][0]
    torch.manual_seed(2026)
    expected = reference.encode_image(image)
    expected.square().sum().backward()

    torch.manual_seed(2026)
    parts = []
    for start in (0, 8):
        output = actual.encode_image(image)[start : start + 8]
        parts.append(output.detach())
        output.square().sum().backward()
    torch.testing.assert_close(torch.cat(parts), expected.detach(), atol=0, rtol=0)
    expected_grad = reference.vision.patch_embed.proj.lora_B.weight.grad
    actual_grad = actual.vision.patch_embed.proj.lora_B.weight.grad
    assert expected_grad.abs().sum() > 0 and actual_grad.abs().sum() > 0
    assert not torch.allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Pin cumulative global-coordinate KV behavior across outer chunks.

Each pass's gathered KV uses cumulative coordinates, but only its current-pass groups
have current values. The tape restores earlier groups in prefill and uses an autograd
leaf for that history during replay.
Tests use CPU float64, independent of accelerator dependencies.
"""

import pytest
import torch

from megatron.core.pipeline_parallel.chunk_tape import (
    ChunkTape,
    assert_replay_matches,
    extend_chunk_kv,
)

DTYPE = torch.float64
ROWS = 3
WIDTH = 2


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["prefill", "grad"])
def test_cumulative_kv_keeps_prefix_and_adds_only_new_coordinates(n_chunks, mode):
    tape = ChunkTape(n_chunks)
    published = None
    for index in range(n_chunks):
        tape.begin(index, mode)
        endpoint = ROWS * (index + 1)
        current = torch.arange(endpoint * WIDTH, dtype=DTYPE).reshape(endpoint, WIDTH)
        if mode == "grad":
            current.requires_grad_()
        out = extend_chunk_kv(tape, 0, "global_kv", current)
        assert out.shape == current.shape
        assert torch.equal(out, current)
        published = out

    assert published is not None


def test_grad_cumulative_output_preserves_prefix_leaf_adjoint():
    tape = ChunkTape(2)
    tape.begin(0, "prefill")
    prefix = torch.arange(ROWS * WIDTH, dtype=DTYPE).reshape(ROWS, WIDTH)
    extend_chunk_kv(tape, 0, "global_kv", prefix)

    tape.begin(1, "grad")
    current = torch.arange(2 * ROWS * WIDTH, dtype=DTYPE).reshape(2 * ROWS, WIDTH)
    current = current.clone().requires_grad_()
    out = extend_chunk_kv(tape, 0, "global_kv", current)
    out.sum().backward()

    assert torch.equal(current.grad, torch.cat((torch.zeros_like(prefix), torch.ones_like(prefix))))
    assert torch.equal(tape.leaves[("global_kv", 0, 0)].grad, torch.ones_like(prefix))


@pytest.mark.parametrize("mode", ["prefill", "grad"])
def test_history_replaces_current_rank_aliases_without_hiding_new_groups(mode):
    tape = ChunkTape(2)
    tape.begin(0, "prefill")
    prefix = torch.tensor([[2.0], [3.0], [0.0]], dtype=DTYPE)
    extend_chunk_kv(tape, 0, "global_kv", prefix)
    tape.begin(1, mode)
    current = torch.tensor([[90.0], [91.0], [7.0], [8.0]], dtype=DTYPE, requires_grad=True)
    historical = torch.tensor([True, True, False, False])
    out = extend_chunk_kv(tape, 0, "global_kv", current, historical)
    assert torch.equal(out, torch.tensor([[2.0], [3.0], [7.0], [8.0]], dtype=DTYPE))
    if mode == "grad":
        out.sum().backward()
        assert torch.equal(current.grad, torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=DTYPE))
        assert torch.equal(tape.leaves[("global_kv", 0, 0)].grad, historical[:3, None].to(DTYPE))


def test_prefill_does_not_replace_the_replay_baseline():
    tape = ChunkTape(2)
    tape.begin(0, "prefill")
    prefix = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    extend_chunk_kv(tape, 0, "global_kv", prefix)
    tape.begin(1, "prefill")
    extend_chunk_kv(tape, 0, "global_kv", torch.tensor([[1.0], [2.0], [3.0]], dtype=DTYPE))

    tape.begin(1, "grad")
    replay = torch.tensor([[1.0], [2.0], [4.0]], dtype=DTYPE)
    actual = extend_chunk_kv(tape, 0, "global_kv", replay)
    with pytest.raises(AssertionError, match="replay"):
        assert_replay_matches(tape, ("global_kv", 0, 1), actual)


def test_prefix_mask_corrections_replace_uncommitted_rows_once():
    tape = ChunkTape(3)
    tape.begin(0, "prefill")
    first = torch.tensor([[2.0], [3.0], [0.0]], dtype=DTYPE)
    assert torch.equal(extend_chunk_kv(tape, 0, "global_kv", first), first)

    tape.begin(1, "prefill")
    second = torch.tensor([[90.0], [91.0], [7.0], [8.0]], dtype=DTYPE)
    historical = torch.tensor([True, True, False])
    actual = extend_chunk_kv(tape, 0, "global_kv", second, historical)
    expected = torch.tensor([[2.0], [3.0], [7.0], [8.0]], dtype=DTYPE)
    assert torch.equal(actual, expected)
    correction_bytes = sum(
        tensor.untyped_storage().nbytes()
        for indices, values in tape.value_patches.values()
        for tensor in (indices, values)
    )
    stored_bytes = sum(value.untyped_storage().nbytes() for value in tape.values.values())
    assert stored_bytes + correction_bytes == expected.numel() * DTYPE.itemsize + DTYPE.itemsize + 8
    assert torch.equal(tape.value(("global_kv", 0, 1)), expected)

    tape.begin(2, "prefill")
    third = torch.cat((torch.full((4, 1), 99.0, dtype=DTYPE), torch.tensor([[9.0]], dtype=DTYPE)))
    actual = extend_chunk_kv(tape, 0, "global_kv", third, torch.ones(4, dtype=torch.bool))
    expected = torch.tensor([[2.0], [3.0], [7.0], [8.0], [9.0]], dtype=DTYPE)
    assert torch.equal(actual, expected)
    assert torch.equal(tape.value(("global_kv", 0, 2)), expected)
    correction_bytes = sum(
        tensor.untyped_storage().nbytes()
        for indices, values in tape.value_patches.values()
        for tensor in (indices, values)
    )
    stored_bytes = sum(value.untyped_storage().nbytes() for value in tape.values.values())
    assert stored_bytes + correction_bytes == expected.numel() * DTYPE.itemsize + DTYPE.itemsize + 8


def _extend_cumulative_reference(tape, layer_idx, name, current, prefix_mask=None):
    """The cumulative-storage implementation before the delta conversion."""
    from megatron.core.pipeline_parallel.chunk_tape import InjectChunkAdjoint

    if tape.chunk_count <= 1:
        return current
    prior_key = (name, layer_idx, tape.index - 1)
    this_key = (name, layer_idx, tape.index)
    prior = tape.values.get(prior_key)
    if prior is None:
        out = current
    else:
        prefix = tape.make_leaf(prior_key, prior) if tape.mode == "grad" else prior
        if prefix_mask is not None:
            mask = prefix_mask[: prior.shape[0]].reshape(-1, *([1] * (current.ndim - 1)))
            prefix = torch.where(mask, prefix, current[: prior.shape[0]])
        out = torch.cat((prefix, current[prior.shape[0] :]), dim=0)
    if tape.mode == "grad" and tape.has_adjoint(this_key):
        out = InjectChunkAdjoint.apply(out, tape, this_key)
    return tape.store(this_key, out)


def _parameter_replays(extend, n_chunks, ratio, masked):
    generator = torch.Generator().manual_seed(20261030)
    rows_per_chunk = 6 // ratio
    inputs = torch.randn(n_chunks * rows_per_chunk, WIDTH, generator=generator, dtype=DTYPE)
    weight = torch.nn.Parameter(torch.randn(WIDTH, 3, generator=generator, dtype=DTYPE))
    tape = ChunkTape(n_chunks)
    outputs = []

    def forward(index, mode):
        tape.begin(index, mode)
        current = inputs[: (index + 1) * rows_per_chunk] @ weight
        prefix_mask = None
        if masked and index:
            prefix_mask = torch.ones(index * rows_per_chunk, dtype=torch.bool)
            prefix_mask[-1] = False
        return extend(tape, 0, "global_kv", current, prefix_mask)

    for index in range(n_chunks):
        with torch.no_grad():
            outputs.append(forward(index, "prefill").clone())
    for index in reversed(range(n_chunks)):
        out = forward(index, "grad")
        if extend is extend_chunk_kv and n_chunks > 1:
            assert_replay_matches(tape, ("global_kv", 0, index), out)
        outputs.append(out.detach().clone())
        (out.square().sum() * (index + 1)).backward()
        tape.collect_all_adjoints()
    return outputs, weight.grad, tape


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 4])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("masked", [False, True])
def test_delta_tape_is_bit_identical_to_cumulative_reference(n_chunks, ratio, masked):
    expected, expected_grad, old_tape = _parameter_replays(
        _extend_cumulative_reference, n_chunks, ratio, masked
    )
    actual, actual_grad, tape = _parameter_replays(extend_chunk_kv, n_chunks, ratio, masked)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert torch.equal(actual_grad, expected_grad)
    assert torch.count_nonzero(actual_grad) == actual_grad.numel()
    for key in tape.leaves:
        assert torch.equal(tape.leaves[key], old_tape.leaves[key])
        assert torch.equal(tape.accum[key], old_tape.accum[key])


@pytest.mark.parametrize("n_chunks", [2, 3, 4])
@pytest.mark.parametrize("ratio", [1, 2])
def test_delta_storage_owns_only_one_full_sequence(n_chunks, ratio):
    _, _, tape = _parameter_replays(extend_chunk_kv, n_chunks, ratio, False)
    full_rows = n_chunks * (6 // ratio)
    expected_bytes = full_rows * 3 * torch.tensor([], dtype=DTYPE).element_size()
    assert sum(value.untyped_storage().nbytes() for value in tape.values.values()) == expected_bytes
    assert tape.value_patches == {}
    assert all(value.grad_fn is None for value in tape.values.values())


def test_single_chunk_bypasses_tape_storage():
    tape = ChunkTape(1)
    current = torch.randn(ROWS, WIDTH, dtype=DTYPE)
    assert extend_chunk_kv(tape, 0, "global_kv", current) is current
    assert tape.values == {}


def test_empty_delta_and_layer_keys_reconstruct_independently():
    tape = ChunkTape(2)
    first = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    other = torch.tensor([[8.0], [9.0]], dtype=DTYPE)
    tape.begin(0, "prefill")
    extend_chunk_kv(tape, 0, "global_kv", first)
    extend_chunk_kv(tape, 1, "indexer_k", other)
    tape.begin(1, "prefill")
    assert torch.equal(extend_chunk_kv(tape, 0, "global_kv", torch.zeros_like(first)), first)
    assert torch.equal(extend_chunk_kv(tape, 1, "indexer_k", torch.zeros_like(other)), other)
    assert tape.values[("global_kv", 0, 1)].numel() == 0
    assert torch.equal(tape.value(("global_kv", 0, 1)), first)
    assert torch.equal(tape.value(("indexer_k", 1, 1)), other)


def test_cumulative_publication_rejects_a_smaller_pass():
    tape = ChunkTape(2)
    tape.begin(0, "prefill")
    extend_chunk_kv(tape, 0, "global_kv", torch.ones((2, 1), dtype=DTYPE))
    tape.begin(1, "prefill")
    with pytest.raises(ValueError, match="prior pass published"):
        extend_chunk_kv(tape, 0, "global_kv", torch.ones((1, 1), dtype=DTYPE))


# CPU reference replay pins the adjoint keys and parameter gradients here. The real
# cross-rank consumer and halo transport are covered by tools/check_outer_chunk_numerics.py.

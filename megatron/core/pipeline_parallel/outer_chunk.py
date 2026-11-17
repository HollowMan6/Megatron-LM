# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Execute a logical microbatch as successive context-parallel chunks."""

from collections.abc import Callable

import torch

from megatron.core.pipeline_parallel.chunk_tape import ChunkTape
from megatron.core.pipeline_parallel.schedules import backward_step, forward_step_calc_loss
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.utils import unwrap_model


def validate_outer_chunk_config(
    *,
    context_parallel_size: int,
    pipeline_model_parallel_size: int,
    dynamic_context_parallel: bool = False,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
    overlap_grad_reduce: bool = False,
    router_replay_mode: str = "disabled",
    forward_only: bool = False,
    transformer_config=None,
) -> None:
    """Check topology, input layout, and objective constraints for chunk counts above one."""
    if dynamic_context_parallel:
        raise NotImplementedError(
            "Outer-sequence chunking requires static CP: "
            f"dynamic_context_parallel={dynamic_context_parallel!r}"
        )
    if pipeline_model_parallel_size != 1:
        raise NotImplementedError(
            "Outer-sequence chunking requires pipeline_model_parallel_size == 1: "
            f"pipeline_model_parallel_size={pipeline_model_parallel_size!r}"
        )
    if not use_remove_padding or use_fused_kernels:
        raise NotImplementedError(
            "Outer-sequence chunking requires the non-fused THD forward: "
            f"use_remove_padding={use_remove_padding!r}, use_fused_kernels={use_fused_kernels!r}"
        )
    if overlap_grad_reduce:
        raise NotImplementedError(
            "Outer-sequence chunking requires overlap_grad_reduce=False: "
            f"overlap_grad_reduce={overlap_grad_reduce!r}"
        )
    if router_replay_mode != "disabled":
        raise NotImplementedError(
            "Outer-sequence chunking does not support router replay: "
            f"router_replay.mode={router_replay_mode!r}"
        )
    if context_parallel_size < 2:
        raise NotImplementedError(
            "Outer-sequence chunking requires physical context_parallel_size > 1: "
            f"context_parallel_size={context_parallel_size!r}; "
            "CSA compression's CP1 path cannot address a later chunk's global token indices"
        )
    if transformer_config is not None:
        _validate_model_config(transformer_config, forward_only=forward_only)


def _validate_model_config(config, *, forward_only: bool) -> None:
    attention = getattr(config, "experimental_attention_variant", None)
    if attention != "dsv4_hybrid":
        raise NotImplementedError(
            "Outer-sequence chunking requires DSV4 hybrid attention and its chunk tape: "
            f"experimental_attention_variant={attention!r}"
        )
    balancing_types = getattr(config, "moe_router_load_balancing_type", "aux_loss")
    balancing_coeff = getattr(config, "moe_aux_loss_coeff", 0)
    balancing_losses = (
        [(balancing_types, balancing_coeff)]
        if isinstance(balancing_types, str)
        else zip(balancing_types, balancing_coeff, strict=True)
    )
    if not forward_only and any(
        kind in ("aux_loss", "seq_aux_loss", "global_aux_loss") and coeff is not None and coeff > 0
        for kind, coeff in balancing_losses
    ):
        raise NotImplementedError(
            "Outer-sequence chunking does not support MoE balancing auxiliary losses: "
            "their expert-frequency products require the complete logical sequence; "
            "per-token scaling only supports additive z-loss; "
            f"moe_router_load_balancing_type={balancing_types!r}, moe_aux_loss_coeff={balancing_coeff!r}"
        )
    z_loss = getattr(config, "moe_z_loss_coeff", None)
    if not config.calculate_per_token_loss and z_loss:
        raise NotImplementedError(
            "Outer-sequence chunking with MoE z-loss requires calculate_per_token_loss=True: "
            f"calculate_per_token_loss={config.calculate_per_token_loss!r}, moe_z_loss_coeff={z_loss!r}"
        )


def validate_outer_chunk_models(models, *, forward_only: bool = False):
    """Check actual routers after construction, returning the provided models unchanged."""
    from megatron.core.transformer.moe.router import TopKRouter

    for module in models if isinstance(models, (list, tuple)) else [models]:
        model = unwrap_model(module)
        _validate_model_config(model.config, forward_only=forward_only)
        for router in model.modules():
            if not isinstance(router, TopKRouter):
                continue
            # Hash layers bypass balancing algorithms, and frozen/eval routers
            # do not update the per-microbatch quantile.
            hash_router = getattr(router, "is_hash_layer", False)
            if (
                router.training
                and not hash_router
                and router.routing_type == "sinkhorn"
                and router.topk < router.num_experts
            ):
                raise NotImplementedError(
                    "Outer-sequence chunking does not support training-mode Sinkhorn routing: "
                    "its assignments require the complete token population; "
                    f"training={router.training!r}, is_hash_layer={hash_router!r}, "
                    f"routing_type={router.routing_type!r}, topk={router.topk!r}, "
                    f"num_experts={router.num_experts!r}"
                )
            if (
                router.training
                and not forward_only
                and not hash_router
                and router.routing_type == "quantile_balancing"
                and router.config.moe_router_quantile_balancing_ema != 1
            ):
                raise NotImplementedError(
                    "Outer-sequence chunking does not support quantile-balancing updates: "
                    "averaging chunk quantiles changes the next step's router bias; use frozen EMA=1 or inference; "
                    f"training={router.training!r}, forward_only={forward_only!r}, "
                    f"is_hash_layer={hash_router!r}, "
                    f"moe_router_quantile_balancing_ema={router.config.moe_router_quantile_balancing_ema!r}"
                )
            capacity_factor = router.config.moe_expert_capacity_factor
            # For smaller positive factors, each chunk can drop different
            # tokens even when the full sequence's router accepts them.
            if (
                capacity_factor is not None
                and 0 < capacity_factor * router.topk < router.num_experts
            ):
                raise NotImplementedError(
                    "Outer-sequence chunking does not support competitive MoE token dropping: "
                    "disable capacity or set capacity_factor * router_topk >= num_experts; "
                    f"capacity_factor={capacity_factor!r}, topk={router.topk!r}, "
                    f"num_experts={router.num_experts!r}"
                )
    return models


def _capture_rng():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        get_cuda_rng_tracker().get_states(),
    )


def _restore_rng(state):
    cpu_state, cuda_state, tracker_states = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    get_cuda_rng_tracker().set_states(tracker_states)


def _add_output(model_output: dict[str, torch.Tensor], output: dict[str, torch.Tensor]) -> None:
    for key, value in output.items():
        if value.requires_grad:
            value = value.detach()
        if key not in model_output:
            model_output[key] = value
        elif value.is_nested:
            previous = model_output[key]
            model_output[key] = torch.nested.nested_tensor_from_jagged(
                previous.values() + value.values(),
                offsets=previous.offsets(),
                lengths=previous.lengths(),
            )
        else:
            model_output[key] = model_output[key] + value


def _differentiate_output(model_output, loss_func):
    leaves = {}
    differentiable_output = dict(model_output)
    with torch.enable_grad():
        for key, value in model_output.items():
            if not value.is_floating_point():
                continue
            values = value.values() if value.is_nested else value
            leaf = values.detach().requires_grad_(True)
            leaves[key] = leaf
            differentiable_output[key] = (
                torch.nested.nested_tensor_from_jagged(
                    leaf, offsets=value.offsets(), lengths=value.lengths()
                )
                if value.is_nested
                else leaf
            )
        if not leaves:
            raise ValueError("Outer-sequence chunk training requires floating-point model outputs")
        result = loss_func(differentiable_output)
        loss = result[0]
        gradients = (
            torch.autograd.grad(loss, tuple(leaves.values()), allow_unused=True)
            if loss.requires_grad
            else (None,) * len(leaves)
        )
    return result, dict(zip(leaves, gradients, strict=True))


def _output_surrogate(output, adjoints):
    surrogate = None
    for key, adjoint in adjoints.items():
        if adjoint is not None:
            value = output[key]
            values = value.values() if value.is_nested else value
            contribution = (values * adjoint).sum()
            surrogate = contribution if surrogate is None else surrogate + contribution
    if surrogate is None:
        # A constant objective can still have model auxiliary losses.
        value = next(value for value in output.values() if value.is_floating_point())
        values = value.values() if value.is_nested else value
        surrogate = values.sum() * 0.0
    return surrogate


def forward_backward_outer_chunks(
    *,
    model: torch.nn.Module,
    forward_pass: Callable,
    chunk_count: int,
    num_microbatches: int,
    cp_group_size: int,
    forward_only: bool = False,
    profile_phase: Callable[[str], None] | None = None,
):
    """Return a forward-step output and loss function for one logical microbatch.

    ``forward_pass(index, tape)`` returns tensor outputs and a Megatron loss
    function. Each tensor is laid out in logical-sequence coordinates with zero
    outside this pass's contribution. The loss function from the last prefill
    pass sees the sum of these tensors outside the pass's model context.

    During training, prefill builds the tape and complete output without
    retaining model graphs. Differentiating the loss once produces the output
    adjoints. Reverse replay combines those with the tape's state adjoints,
    freeing each model graph before the next pass. The enclosing schedule still
    finalizes parameter gradients with the logical microbatch's token count.
    """
    if chunk_count < 2:
        raise ValueError("chunk_count < 2 must use the ordinary path")
    unwrapped = unwrap_model(model)
    config = unwrapped.config
    tape = ChunkTape(chunk_count)
    model_output = {}
    replay_rng = []
    with torch.no_grad():
        for index in range(chunk_count):
            if not forward_only:
                replay_rng.append(_capture_rng())
            tape.begin(index, "prefill")
            output, loss_func = forward_pass(index, tape)
            _add_output(model_output, output)
            if profile_phase is not None:
                profile_phase(f"chunk_{index}_prefill_end")

    if forward_only:
        with torch.no_grad():
            full_result = loss_func(model_output)
    else:
        full_result, output_adjoints = _differentiate_output(model_output, loss_func)
        del model_output, output, loss_func
        # Keep loss-function RNG consumption while replaying the model with
        # each pass's original snapshot.
        final_rng = _capture_rng()
        try:
            for index in reversed(range(chunk_count)):
                tape.begin(index, "grad")
                _restore_rng(replay_rng[index])
                with torch.enable_grad():
                    output, _ = forward_pass(index, tape)
                    surrogate = _output_surrogate(output, output_adjoints)

                    def surrogate_loss(_output):
                        return (surrogate, *full_result[1:])

                    # Apply Core's normal CP/microbatch or per-token scaling
                    # and its auxiliary-loss scaling to each pass.
                    loss, _ = forward_step_calc_loss(
                        model=model,
                        output_tensor=output,
                        loss_func=surrogate_loss,
                        config=config,
                        vp_stage=getattr(unwrapped, "vp_stage", None),
                        collect_non_loss_data=False,
                        num_microbatches=num_microbatches,
                        forward_data_store=[],
                        cp_group_size=cp_group_size,
                        is_last_stage=True,
                    )
                if profile_phase is not None:
                    profile_phase(f"chunk_{index}_forward_end")
                    profile_phase(f"chunk_{index}_backward_start")
                backward_step(None, loss, None, config)
                if profile_phase is not None:
                    profile_phase(f"chunk_{index}_backward_end")
                tape.collect_all_adjoints()
                del output, surrogate, loss
        finally:
            _restore_rng(final_rng)

    seed = torch.zeros_like(full_result[0], requires_grad=not forward_only) * 1.0
    logged = full_result[-1]
    if config.calculate_per_token_loss:
        logical_num_tokens = (
            full_result[1]
            if len(full_result) == 3
            else torch.zeros((), dtype=torch.int, device=seed.device)
        )
        return seed, lambda _output: (seed, logical_num_tokens, logged)
    return seed, lambda _output: (seed, logged)

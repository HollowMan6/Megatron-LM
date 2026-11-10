# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validation of the logical CSA2 layer schedule, independent of stack layout."""

import math

import torch


def _validate_tilelang_backend_dependencies() -> None:
    """Fail fast when the CSA2 TileLang sparse-MLA kernels cannot run here.

    Importing the runtime is not the same as having the kernels: the adapter's own
    guard tolerates a missing shared object or a failed import, in which case it
    reports the backend as unavailable and every forward silently takes the
    reference path. Checking the adapter's availability flag makes the configured
    backend a promise instead of a hope.
    """
    try:
        from .ops.tilelang_csa_sparse_mla import _HAVE_TILELANG_SPARSE_MLA
    except (ImportError, OSError) as exc:  # missing runtime, missing kernel .so
        raise ImportError(
            "dsa_kernel_backend='tilelang' requires the CSA2 sparse-MLA adapter, which "
            f"could not be imported: {exc}. Install TileLang with its TVM FFI runtime, or "
            "select dsa_kernel_backend='cudnn' or 'none'."
        ) from exc
    if not _HAVE_TILELANG_SPARSE_MLA:
        raise ImportError(
            "dsa_kernel_backend='tilelang' was requested but the TileLang sparse-MLA "
            "forward/backward kernels are unavailable in this environment (the adapter "
            "imported, its kernels did not). Without them every forward would silently "
            "run the PyTorch reference; install a working TileLang build, or select "
            "dsa_kernel_backend='cudnn' or 'none'."
        )


def validate_csa2_config(config) -> None:
    """Reject inconsistent owners, compression ratios and hierarchical candidate pools."""
    for name in ("csa2_fp4_compressed_kv", "csa2_fp4_indexer_qk", "csa2_fp8_output_act"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if not config.multi_latent_attention or not config.qk_layernorm:
        raise ValueError("CSA2 requires MLA and learned query/KV RMS normalization")
    if config.normalization != "RMSNorm" or config.layernorm_zero_centered_gamma:
        raise ValueError("CSA2 requires ordinary RMSNorm weights")
    if config.tensor_model_parallel_size != 1:
        raise NotImplementedError("CSA2 currently requires TP=1")
    if config.fp16 or config.params_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("CSA2 supports FP32 or BF16 parameters")
    # 'tilelang' is accepted only together with the adapter it needs: the check below
    # imports the kernel runtime, so a config that validates here cannot die with a
    # ModuleNotFoundError on the first forward.
    if config.dsa_kernel_backend not in ("none", "cudnn", "tilelang"):
        raise ValueError(
            "CSA2 supports the native, cuDNN and TileLang sparse attention backends"
        )
    if config.dsa_kernel_backend == "tilelang":
        _validate_tilelang_backend_dependencies()
    if config.dsa_kernel_backend != "none" and config.params_dtype != torch.bfloat16:
        raise ValueError("Fused CSA2 attention requires BF16 parameters")
    if config.attention_dropout or config.hidden_dropout:
        raise ValueError("CSA2 currently requires zero attention and hidden dropout")
    if config.csa_dense_mode or config.qk_clip:
        raise ValueError("CSA2 does not use dense CSA or QK clipping")
    if config.add_bias_linear or config.qk_l2_norm or config.dsa_indexer_rotate_activation:
        raise ValueError(
            "CSA2 uses bias-free projections, RMSNorm and unrotated indexer activations"
        )
    # CSA2's attention GEMMs (MLA q/kv, the compressor and the indexer) are exempted from
    # FP8 by get_fp8_disabled_context(self.config) wrapped around
    # CompressedSparseAttention2.forward, so a global fp8=... run is allowed here: the rest
    # of the model is quantized, these layers stay in BF16/FP32.
    if config.fp4 or config.quant_recipe is not None:
        raise NotImplementedError(
            "CSA2 does not support FP4 or weight QAT: its attention GEMMs are kept in "
            "BF16/FP32 by the disabled quantization context, so there is no CSA2-side "
            "hook that could quantize them or apply a QAT recipe. Relax this guard "
            "together with such a hook."
        )
    for name in ("layernorm_epsilon", "csa_compress_rotary_base"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"CSA2 requires positive finite {name}")
    if config.moe_shortcut_connection:
        raise NotImplementedError("CSA2 does not yet support shortcut MoE")
    if config.pipeline_model_parallel_size != 1 or config.virtual_pipeline_model_parallel_size:
        raise NotImplementedError("CSA2 currently requires PP=1 and no VPP")
    if config.cuda_graph_impl != "none":
        raise NotImplementedError("CSA2 cross-layer state currently requires eager execution")
    _CSA2_ATTN_RECOMPUTE_MODULES = {
        "core_attn",
        "gdn_norm_out",
        "gdp_qkv",
        "gdp_in_proj",
        "shortcut_pre_mlp_layernorm",
    }
    if config.recompute_granularity is not None:
        if config.recompute_granularity == "full":
            if (config.dsa_indexer_loss_coeff or 0) > 0:
                raise NotImplementedError(
                    "CSA2 full recompute does not yet support indexer-loss tracking"
                )
            if config.recompute_method != "uniform" or config.recompute_num_layers != 1:
                raise NotImplementedError(
                    "CSA2 full recompute currently checkpoints one layer per region"
                )
        elif config.recompute_granularity == "selective":
            _attn_in_recompute = set(config.recompute_modules or []) & _CSA2_ATTN_RECOMPUTE_MODULES
            if _attn_in_recompute:
                raise NotImplementedError(
                    f"CSA2 cross-layer state requires eager attention; cannot "
                    f"recompute attention modules {sorted(_attn_in_recompute)}"
                )
        else:
            raise NotImplementedError("CSA2 cross-layer state currently requires eager execution")
    if config.overlap_moe_expert_parallel_comm or config.delay_wgrad_compute:
        raise NotImplementedError("CSA2 does not yet support layer overlap or delayed wgrad")
    # Fine-grained activation offloading is still refused, and now the reason is measured
    # rather than assumed. The guard was lifted on 2026-11-12 (rev bf4870d01): CSA2's
    # forward already calls the offload manager for core_attn/attn_proj and the experts
    # module does the same for expert_fc1/moe_act, so allowing the flag looked
    # numerics-neutral and safe. On the 190K lane it changed nothing -- a 175K run with
    # offload_modules=[moe_act, expert_fc1] peaked at 127.96 GiB and stepped in 41.5 s,
    # identical to the same run without offload -- so the hooks do not engage for this
    # model's MoE path and the flag would advertise memory it does not free. Re-lifted
    # only when a run can show a lower peak.
    if config.fine_grained_activation_offloading:
        # Not a wiring gap and not a configuration mistake: the offload manager keeps
        # every group's tensors resident on the GPU during warmup and only starts moving
        # bytes on the next iteration, so turning it on raises the peak to "all layers'
        # groups at once" before it ever lowers it. Measured 2026-11-12 on this model:
        # 131072 tokens alone peak at 109.74 GiB of a 139.81 GiB card, and enabling
        # offload -- with the schedule lifecycle implemented (5fec05dc9) -- OOMed during
        # the warmup of that very step. The feature needs ~30 GiB of headroom this model
        # does not have at any length we run (nine probes; see the invariance report §6).
        raise NotImplementedError(
            "CSA2 activation offloading is structurally unusable on this model: its "
            "warmup needs every layer's activation resident at once, which exceeds the "
            "card before the offload saves anything (measured 2026-11-12)"
        )
    if config.offload_modules:
        raise ValueError("CSA2 offload_modules has no effect; leave it empty")
    ratios = config.csa_compress_ratios
    if not ratios or any(type(r) is not int or r not in (0, 1, 2) for r in ratios):
        raise ValueError("CSA2 requires one ratio (0, 1, or 2) per stack layer")
    if config.num_layers != len(ratios):
        raise ValueError("CSA2 ratios must contain exactly num_layers entries")
    for name in ("csa2_kv_source_layers", "csa2_index_source_layers"):
        sources = getattr(config, name)
        if sources is None or any(type(i) is not int or not 0 <= i < len(ratios) for i in sources):
            raise ValueError(f"{name} must contain zero-based stack layer IDs")
        if sources != sorted(set(sources)) or any(ratios[i] == 0 for i in sources):
            raise ValueError(f"{name} must be strictly increasing and exclude SWA layers")
    if not set(config.csa2_kv_source_layers).issubset(config.csa2_index_source_layers):
        raise ValueError("Every global KV owner must also produce sparse indices")
    owner_ratio = None
    for i, ratio in enumerate(ratios):
        if not ratio:
            continue
        if i in config.csa2_kv_source_layers:
            owner_ratio = ratio
        elif owner_ratio is None or owner_ratio != ratio:
            raise ValueError(f"CSA2 layer {i} needs a preceding Full owner with the same ratio")
    source = config.csa2_candidate_source_layer
    blocks, width = config.csa2_candidate_topk_blocks, config.csa2_candidate_block_size
    if source is None:
        if blocks or width:
            raise ValueError("Disabled hierarchical indexing requires zero candidate dimensions")
    else:
        if source not in config.csa2_kv_source_layers or any(
            i > source for i in config.csa2_kv_source_layers
        ):
            raise ValueError("The candidate source must be the final Full owner")
        if type(blocks) is not int or type(width) is not int or min(blocks, width) <= 0:
            raise ValueError("Candidate dimensions must be positive integers")
        if blocks * width < config.dsa_indexer_topk:
            raise ValueError("The hierarchical candidate pool must cover the attention Top-K")
    for name in (
        "csa_window_size",
        "dsa_indexer_topk",
        "dsa_indexer_n_heads",
        "dsa_indexer_head_dim",
    ):
        if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
            raise ValueError(f"CSA2 requires positive {name}")
    if not 0 < config.qk_pos_emb_head_dim <= min(config.v_head_dim, config.dsa_indexer_head_dim):
        raise ValueError("CSA2 rotary dimensions must fit both attention and indexer heads")
    if config.qk_pos_emb_head_dim % 2:
        raise ValueError("CSA2 rotary dimensions must be even")

"""Intrinsic metrics computed from residual-stream activations.

The vLLM plugin captures decoder-layer residual streams with shape
``(n_layers, seq_len, hidden_dim)`` where ``seq_len`` is
``prompt_tokens + generated_tokens - 1``.  Metrics are scored only over
generated response tokens; token ``pos`` is scored from the activation at
``pos - 1`` because causal LM logits at position ``t`` predict token
``t + 1``.
"""

from __future__ import annotations

import math
import json
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def jsd_from_logits(
    logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Jensen-Shannon divergence between two vocab distributions."""
    log_p = F.log_softmax(logits_a.float(), dim=-1)
    log_q = F.log_softmax(logits_b.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)

    log_m = (m + eps).log()
    kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
    kl_qm = torch.sum(q * (log_q - log_m), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def kl_uniform_to_probs(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(U || P) where U is uniform over the vocabulary."""
    del eps
    log_p = F.log_softmax(logits.float(), dim=-1)
    vocab_size = logits.shape[-1]
    u = 1.0 / vocab_size
    log_u = math.log(u)
    return torch.sum(torch.full_like(log_p, u) * (log_u - log_p), dim=-1)


def normalized_confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalized KL(P || U), equivalent to 1 - H(P) / log(|V|)."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    vocab_size = logits.shape[-1]
    if vocab_size <= 1:
        return torch.zeros(logits.shape[:-1], dtype=log_p.dtype, device=logits.device)
    entropy = -torch.sum(p * log_p, dim=-1)
    return 1.0 - entropy / math.log(vocab_size)


def normalize_metric_options(options: Any | None) -> dict[str, Any]:
    """Normalize ``extra_args["output_intrinsic_metrics"]``."""
    if isinstance(options, (list, tuple)) and len(options) == 1:
        options = options[0]
    if options == 1:
        options = True
    elif options == 0:
        options = None
    if isinstance(options, str):
        lowered = options.strip().lower()
        if lowered in {"true", "1"}:
            options = True
        elif lowered in {"false", "0", "none", "null"}:
            options = None
        else:
            try:
                options = json.loads(options)
            except json.JSONDecodeError as exc:
                raise TypeError(
                    "output_intrinsic_metrics must be True or a dict of metric options"
                ) from exc
    if options is True or options is None:
        options = {}
    if not isinstance(options, dict):
        raise TypeError(
            "output_intrinsic_metrics must be True or a dict of metric options; "
            f"got {type(options).__name__}: {options!r}"
        )

    logits_batch_size = int(options.get("logits_batch_size", 128))
    if logits_batch_size <= 0:
        raise ValueError("logits_batch_size must be positive")

    lsc_tolerance = float(options.get("lsc_tolerance", 1e-3))
    if lsc_tolerance < 0:
        raise ValueError("lsc_tolerance must be non-negative")

    metrics_storage = str(options.get("metrics_storage", "cpu")).lower()
    if metrics_storage not in {"cpu", "gpu"}:
        raise ValueError('metrics_storage must be either "cpu" or "gpu"')

    return {
        "jsd_threshold": float(options.get("jsd_threshold", 0.5)),
        "deep_layer_fraction": float(options.get("deep_layer_fraction", 0.85)),
        "lsc_tolerance": lsc_tolerance,
        "logits_batch_size": logits_batch_size,
        "metrics_storage": metrics_storage,
    }


def _response_hidden_states(
    residual_stream: torch.Tensor,
    full_token_ids: list[int],
    prompt_len: int,
    *,
    response_positions_only: bool,
) -> torch.Tensor:
    num_response_tokens = len(full_token_ids) - prompt_len
    if num_response_tokens <= 0:
        return residual_stream[:, :0, :]

    if response_positions_only:
        if residual_stream.shape[1] < num_response_tokens:
            raise ValueError(
                "residual_stream is shorter than the response token sequence: "
                f"{residual_stream.shape[1]} < {num_response_tokens}"
            )
        return residual_stream[:, :num_response_tokens, :]

    expected_activation_len = max(len(full_token_ids) - 1, 0)
    activation_seq_len = residual_stream.shape[1]
    if activation_seq_len < expected_activation_len:
        raise ValueError(
            "residual_stream is shorter than the token sequence: "
            f"{activation_seq_len} < {expected_activation_len}"
        )
    if activation_seq_len > expected_activation_len:
        residual_stream = residual_stream[:, :expected_activation_len, :]

    prev_positions = torch.arange(prompt_len - 1, len(full_token_ids) - 1)
    return residual_stream.index_select(1, prev_positions.to(residual_stream.device))


@torch.no_grad()
def compute_intrinsic_metrics_from_activations(
    residual_stream: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    full_token_ids: list[int],
    prompt_len: int,
    *,
    jsd_threshold: float = 0.5,
    deep_layer_fraction: float = 0.85,
    lsc_tolerance: float = 1e-3,
    logits_device: torch.device | None = None,
    final_logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    logits_batch_size: int = 128,
    response_positions_only: bool = False,
    metrics_storage: str | None = None,
) -> dict[str, float]:
    """Compute DTR, self-certainty, LSC metrics, and average log probability.

    Args:
        residual_stream: Tensor with shape ``(n_layers, seq_len, hidden_dim)``.
        logits_fn: Callable that maps intermediate hidden states to vocabulary logits.
        full_token_ids: Prompt token IDs followed by generated token IDs.
        prompt_len: Number of prompt tokens in ``full_token_ids``.
    """
    if residual_stream.dim() != 3:
        raise ValueError(
            "residual_stream must have shape (n_layers, seq_len, hidden_dim)"
        )
    if prompt_len <= 0:
        raise ValueError("prompt_len must be at least 1")
    if prompt_len > len(full_token_ids):
        raise ValueError("prompt_len cannot exceed len(full_token_ids)")

    if logits_batch_size <= 0:
        raise ValueError("logits_batch_size must be positive")
    if lsc_tolerance < 0:
        raise ValueError("lsc_tolerance must be non-negative")
    del metrics_storage

    num_layers, _, _ = residual_stream.shape
    response_hidden = _response_hidden_states(
        residual_stream,
        full_token_ids,
        prompt_len,
        response_positions_only=response_positions_only,
    )
    num_response_tokens = response_hidden.shape[1]

    deep_start_layer = math.ceil(num_layers * deep_layer_fraction)
    deep_start_layer = max(1, min(deep_start_layer, num_layers))

    if num_response_tokens == 0:
        return {
            "deep_thinking_ratio": 0.0,
            "settled_deep_thinking_ratio": 0.0,
            "average_deep_thinking_settling_depth": 0.0,
            "self_certainty": 0.0,
            "layerwise_self_certainty_settling_depth": 0.0,
            "layerwise_self_certainty_deep_thinking_ratio": 0.0,
            "normalized_confidence": 0.0,
            "average_log_probability": 0.0,
            "num_response_tokens": 0.0,
            "deep_layer_start": float(deep_start_layer),
            "num_layers": float(num_layers),
            "jsd_threshold": float(jsd_threshold),
            "lsc_tolerance": float(lsc_tolerance),
        }

    device = logits_device or residual_stream.device
    response_token_ids = torch.tensor(
        full_token_ids[prompt_len:], dtype=torch.long, device=device
    )

    final_logits_fn_resolved = final_logits_fn or logits_fn
    final_hidden = response_hidden[-1].to(device=device)
    final_logits = final_logits_fn_resolved(final_hidden)
    if final_logits.dim() == 1:
        final_logits = final_logits.unsqueeze(0)
    vocab_size = final_logits.shape[-1]
    too_large = response_token_ids >= vocab_size
    if bool(torch.any(too_large).item()):
        token_id = int(response_token_ids[too_large][0].item())
        raise ValueError(
            f"token_id {token_id} is outside output vocab size {vocab_size}"
        )

    final_log_probs = F.log_softmax(final_logits.float(), dim=-1)
    token_logprobs_t = final_log_probs.gather(1, response_token_ids[:, None]).squeeze(1)
    token_self_certainties_t = kl_uniform_to_probs(final_logits)
    token_normalized_confidences_t = normalized_confidence_from_logits(final_logits)

    jsd_per_layer = torch.empty(
        (num_layers, num_response_tokens), dtype=torch.float32, device=device
    )
    self_certainty_per_layer = torch.empty(
        (num_layers, num_response_tokens), dtype=torch.float32, device=device
    )
    for layer_idx in range(num_layers):
        layer_hidden_all = response_hidden[layer_idx]
        layer_logits_fn = (
            final_logits_fn_resolved if layer_idx == num_layers - 1 else logits_fn
        )
        for start in range(0, num_response_tokens, logits_batch_size):
            end = min(start + logits_batch_size, num_response_tokens)
            hidden = layer_hidden_all[start:end].to(device=device)
            layer_logits = layer_logits_fn(hidden)
            if layer_logits.dim() == 1:
                layer_logits = layer_logits.unsqueeze(0)
            expanded_final = final_logits[start:end]
            jsd_per_layer[layer_idx, start:end] = jsd_from_logits(
                layer_logits, expanded_final
            )
            self_certainty_per_layer[layer_idx, start:end] = kl_uniform_to_probs(
                layer_logits
            )

    running_min_jsd = torch.cummin(jsd_per_layer, dim=0).values
    below_threshold = running_min_jsd <= jsd_threshold
    first_below = below_threshold.float().argmax(dim=0) + 1
    has_below = below_threshold.any(dim=0)
    settling_depth = torch.where(
        has_below,
        first_below,
        torch.full_like(first_below, num_layers),
    )
    deep_thinking_flags_t = (settling_depth >= deep_start_layer).float()
    deep_thinking_depths = settling_depth[deep_thinking_flags_t.bool()].float()
    average_deep_thinking_settling_depth = (
        float(deep_thinking_depths.mean().item())
        if deep_thinking_depths.numel()
        else 0.0
    )

    settled_below_threshold = jsd_per_layer <= jsd_threshold
    settled_from_layer = (
        torch.cumprod(settled_below_threshold.flip(0).int(), dim=0).flip(0).bool()
    )
    first_settled = settled_from_layer.float().argmax(dim=0) + 1
    has_settled = settled_from_layer.any(dim=0)
    settled_depth = torch.where(
        has_settled,
        first_settled,
        torch.full_like(first_settled, num_layers),
    )
    settled_deep_thinking_flags_t = (settled_depth >= deep_start_layer).float()

    final_self_certainty = self_certainty_per_layer[-1]
    lsc_within_tolerance = (
        torch.abs(self_certainty_per_layer - final_self_certainty.unsqueeze(0))
        <= lsc_tolerance
    )
    lsc_stable_from_layer = (
        torch.cumprod(lsc_within_tolerance.flip(0).int(), dim=0).flip(0).bool()
    )
    lsc_settling_depth = lsc_stable_from_layer.float().argmax(dim=0) + 1
    if num_layers > 1:
        lsc_normalized_settling_depth = (lsc_settling_depth.float() - 1.0) / (
            num_layers - 1
        )
    else:
        lsc_normalized_settling_depth = torch.zeros_like(
            lsc_settling_depth, dtype=torch.float32
        )
    lsc_deep_thinking_flags_t = (lsc_settling_depth >= deep_start_layer).float()

    return {
        "deep_thinking_ratio": float(deep_thinking_flags_t.mean().item()),
        "settled_deep_thinking_ratio": float(
            settled_deep_thinking_flags_t.mean().item()
        ),
        "average_deep_thinking_settling_depth": average_deep_thinking_settling_depth,
        "self_certainty": float(token_self_certainties_t.mean().item()),
        "layerwise_self_certainty_settling_depth": float(
            lsc_normalized_settling_depth.mean().item()
        ),
        "layerwise_self_certainty_deep_thinking_ratio": float(
            lsc_deep_thinking_flags_t.mean().item()
        ),
        "normalized_confidence": float(token_normalized_confidences_t.mean().item()),
        "average_log_probability": float(token_logprobs_t.mean().item()),
        "num_response_tokens": float(num_response_tokens),
        "deep_layer_start": float(deep_start_layer),
        "num_layers": float(num_layers),
        "jsd_threshold": float(jsd_threshold),
        "lsc_tolerance": float(lsc_tolerance),
    }

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
    kl_pm = torch.sum(p * (log_p - log_m))
    kl_qm = torch.sum(q * (log_q - log_m))
    return 0.5 * (kl_pm + kl_qm)


def kl_uniform_to_probs(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(U || P) where U is uniform over the vocabulary."""
    del eps
    log_p = F.log_softmax(logits.float(), dim=-1)
    vocab_size = logits.shape[-1]
    u = 1.0 / vocab_size
    log_u = math.log(u)
    return torch.sum(torch.full_like(log_p, u) * (log_u - log_p))


def normalize_metric_options(options: Any | None) -> dict[str, float]:
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

    return {
        "jsd_threshold": float(options.get("jsd_threshold", 0.02)),
        "deep_layer_fraction": float(options.get("deep_layer_fraction", 0.25)),
    }


@torch.no_grad()
def compute_intrinsic_metrics_from_activations(
    residual_stream: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    full_token_ids: list[int],
    prompt_len: int,
    *,
    jsd_threshold: float = 0.02,
    deep_layer_fraction: float = 0.25,
    logits_device: torch.device | None = None,
    final_logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compute DTR, self-certainty, and average log probability.

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

    num_layers, activation_seq_len, _ = residual_stream.shape
    expected_activation_len = max(len(full_token_ids) - 1, 0)
    if activation_seq_len < expected_activation_len:
        raise ValueError(
            "residual_stream is shorter than the token sequence: "
            f"{activation_seq_len} < {expected_activation_len}"
        )
    if activation_seq_len > expected_activation_len:
        residual_stream = residual_stream[:, :expected_activation_len, :]

    deep_start_layer = math.floor(num_layers * (1.0 - deep_layer_fraction)) + 1
    deep_start_layer = max(1, min(deep_start_layer, num_layers))

    token_logprobs: list[float] = []
    token_self_certainties: list[float] = []
    deep_thinking_flags: list[int] = []

    device = logits_device or residual_stream.device

    for pos in range(prompt_len, len(full_token_ids)):
        prev_pos = pos - 1
        token_id = int(full_token_ids[pos])

        final_hidden = residual_stream[-1, prev_pos].to(device=device)
        final_logits = (final_logits_fn or logits_fn)(final_hidden.unsqueeze(0))[0]
        final_log_probs = F.log_softmax(final_logits.float(), dim=-1)
        if token_id >= final_logits.shape[-1]:
            raise ValueError(
                f"token_id {token_id} is outside output vocab size "
                f"{final_logits.shape[-1]}"
            )
        token_logprobs.append(float(final_log_probs[token_id].item()))
        token_self_certainties.append(float(kl_uniform_to_probs(final_logits).item()))

        jsd_per_layer: list[float] = []
        for layer_idx in range(num_layers):
            hidden = residual_stream[layer_idx, prev_pos].to(device=device)
            layer_logits = logits_fn(hidden.unsqueeze(0))[0]
            jsd_per_layer.append(
                float(jsd_from_logits(layer_logits, final_logits).item())
            )

        settling_depth = num_layers
        for layer_idx in range(1, num_layers + 1):
            tail = jsd_per_layer[layer_idx - 1 :]
            if tail and tail[0] <= jsd_threshold and all(
                v <= jsd_threshold for v in tail
            ):
                settling_depth = layer_idx
                break

        deep_thinking_flags.append(int(settling_depth >= deep_start_layer))

    denom = max(len(token_logprobs), 1)
    return {
        "deep_thinking_ratio": float(sum(deep_thinking_flags) / denom),
        "self_certainty": float(sum(token_self_certainties) / denom),
        "average_log_probability": float(sum(token_logprobs) / denom),
        "num_response_tokens": float(len(token_logprobs)),
        "deep_layer_start": float(deep_start_layer),
        "num_layers": float(num_layers),
        "jsd_threshold": float(jsd_threshold),
    }

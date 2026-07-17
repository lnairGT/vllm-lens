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


FINAL_METRICS = {
    "average_log_probability",
    "negative_perplexity",
    "normalized_confidence",
    "self_certainty",
}
DEEP_THINKING_METRICS = {
    "average_deep_thinking_settling_depth",
    "deep_thinking_ratio",
    "settled_deep_thinking_ratio",
}
REVISION_METRICS = {"ema_final_score", "ema_mean_score", "peak_change_score"}
ALL_METRICS = FINAL_METRICS | DEEP_THINKING_METRICS | REVISION_METRICS


def normalize_metric_names(metrics: Any | None) -> list[str]:
    if not metrics:
        return ["self_certainty"]
    if not isinstance(metrics, (list, tuple)) or not all(
        isinstance(metric, str) for metric in metrics
    ):
        raise TypeError("metrics must be a list of metric names")
    if "all" in metrics:
        if len(metrics) != 1:
            raise ValueError('"all" cannot be combined with other metrics')
        return sorted(ALL_METRICS)
    unknown = set(metrics) - ALL_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    return list(dict.fromkeys(metrics))


def required_metric_layers(
    total_layers: int,
    metrics: list[str],
    revision_middle_layer: int | None,
    *,
    return_token_self_certainties: bool,
) -> list[int]:
    selected = set(metrics)
    layers = {total_layers - 1}
    if selected & DEEP_THINKING_METRICS:
        layers.update(range(total_layers))
    if selected & REVISION_METRICS:
        middle = revision_middle_layer
        if middle is None:
            middle = min(total_layers // 2, total_layers - 2)
        if not (0 <= middle < total_layers - 1):
            raise ValueError(
                "revision_middle_layer must satisfy "
                f"0 <= revision_middle_layer < num_layers-1; got {middle}"
            )
        layers.update(range(middle, total_layers))
    if not selected and not return_token_self_certainties:
        return []
    return sorted(layers)


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


def normalized_jsd_from_logits(
    logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Jensen-Shannon divergence normalized to [0, 1]."""
    return jsd_from_logits(logits_a, logits_b, eps=eps) / math.log(2.0)


def total_variation_from_logits(
    logits_a: torch.Tensor, logits_b: torch.Tensor
) -> torch.Tensor:
    """Total variation distance between two vocab distributions."""
    probs_a = F.softmax(logits_a.float(), dim=-1)
    probs_b = F.softmax(logits_b.float(), dim=-1)
    return 0.5 * torch.sum(torch.abs(probs_a - probs_b), dim=-1)


def kl_uniform_to_probs(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(U || P) where U is uniform over the vocabulary."""
    del eps
    log_p = F.log_softmax(logits.float(), dim=-1)
    vocab_size = logits.shape[-1]
    return -math.log(vocab_size) - log_p.mean(dim=-1)


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

    metrics_storage = str(options.get("metrics_storage", "cpu")).lower()
    if metrics_storage not in {"cpu", "gpu"}:
        raise ValueError('metrics_storage must be either "cpu" or "gpu"')

    revision_alpha = float(options.get("revision_alpha", 0.5))
    if not (0.0 < revision_alpha <= 1.0):
        raise ValueError("revision_alpha must be in (0, 1]")

    revision_distance = str(options.get("revision_distance", "jsd")).lower()
    if revision_distance not in {"jsd", "tv"}:
        raise ValueError('revision_distance must be either "jsd" or "tv"')

    revision_middle_layer = options.get("revision_middle_layer")
    if revision_middle_layer is not None:
        revision_middle_layer = int(revision_middle_layer)

    metrics = normalize_metric_names(options.get("metrics"))

    return {
        "metrics": metrics,
        "jsd_threshold": float(options.get("jsd_threshold", 0.5)),
        "deep_layer_fraction": float(options.get("deep_layer_fraction", 0.85)),
        "logits_batch_size": logits_batch_size,
        "metrics_storage": metrics_storage,
        "revision_alpha": revision_alpha,
        "revision_middle_layer": revision_middle_layer,
        "revision_distance": revision_distance,
        "return_token_self_certainties": bool(
            options.get("return_token_self_certainties", False)
        ),
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


def _revision_distance_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    distance: str,
) -> torch.Tensor:
    if distance == "jsd":
        return normalized_jsd_from_logits(logits_a, logits_b)
    if distance == "tv":
        return total_variation_from_logits(logits_a, logits_b)
    raise ValueError(f"Unsupported revision_distance: {distance}")


def _compute_revision_scores_from_hidden_states(
    response_hidden: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    final_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    middle_layer: int,
    alpha: float,
    distance: str,
    logits_device: torch.device,
    logits_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_response_tokens, _ = response_hidden.shape
    if not (0 <= middle_layer < num_layers - 1):
        raise ValueError(
            "revision_middle_layer must satisfy "
            f"0 <= revision_middle_layer < num_layers-1; got {middle_layer}"
        )

    num_steps = num_layers - middle_layer - 1
    deltas = torch.empty(
        (num_steps, num_response_tokens), dtype=torch.float32, device=logits_device
    )
    for step, layer_idx in enumerate(range(middle_layer, num_layers - 1)):
        next_layer_idx = layer_idx + 1
        current_logits_fn = logits_fn
        next_logits_fn = (
            final_logits_fn if next_layer_idx == num_layers - 1 else logits_fn
        )
        for start in range(0, num_response_tokens, logits_batch_size):
            end = min(start + logits_batch_size, num_response_tokens)
            current_hidden = response_hidden[layer_idx, start:end].to(
                device=logits_device
            )
            next_hidden = response_hidden[next_layer_idx, start:end].to(
                device=logits_device
            )
            current_logits = current_logits_fn(current_hidden)
            next_logits = next_logits_fn(next_hidden)
            if current_logits.dim() == 1:
                current_logits = current_logits.unsqueeze(0)
            if next_logits.dim() == 1:
                next_logits = next_logits.unsqueeze(0)
            deltas[step, start:end] = _revision_distance_from_logits(
                current_logits, next_logits, distance=distance
            )

    token_peak_change = torch.max(deltas, dim=0).values
    ema = torch.empty_like(deltas)
    ema[0] = deltas[0]
    for step in range(1, num_steps):
        ema[step] = alpha * deltas[step] + (1.0 - alpha) * ema[step - 1]

    return token_peak_change, ema[-1], ema.mean(dim=0)


@torch.no_grad()
def compute_intrinsic_metrics_from_activations(
    residual_stream: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    full_token_ids: list[int],
    prompt_len: int,
    *,
    jsd_threshold: float = 0.5,
    deep_layer_fraction: float = 0.85,
    logits_device: torch.device | None = None,
    final_logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    logits_batch_size: int = 128,
    response_positions_only: bool = False,
    metrics_storage: str | None = None,
    revision_alpha: float = 0.5,
    revision_middle_layer: int | None = None,
    revision_distance: str = "jsd",
    return_token_self_certainties: bool = False,
    metrics: list[str] | None = None,
    layer_indices: list[int] | None = None,
    num_model_layers: int | None = None,
) -> dict[str, float | list[float]]:
    """Compute selected intrinsic metrics from the required residual layers.

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
    if not (0.0 < revision_alpha <= 1.0):
        raise ValueError("revision_alpha must be in (0, 1]")
    revision_distance = revision_distance.lower()
    if revision_distance not in {"jsd", "tv"}:
        raise ValueError('revision_distance must be either "jsd" or "tv"')
    del metrics_storage

    captured_layers, _, _ = residual_stream.shape
    if layer_indices is None:
        layer_indices = list(range(captured_layers))
    if len(layer_indices) != captured_layers:
        raise ValueError("layer_indices must match residual_stream's first dimension")
    num_layers = num_model_layers or (max(layer_indices) + 1)
    selected_metrics = set(normalize_metric_names(metrics))
    required_layers = required_metric_layers(
        num_layers,
        list(selected_metrics),
        revision_middle_layer,
        return_token_self_certainties=return_token_self_certainties,
    )
    missing_layers = set(required_layers) - set(layer_indices)
    if missing_layers:
        raise ValueError(
            f"Selected metrics require decoder layers {required_layers}; got {layer_indices}"
        )
    if revision_middle_layer is None:
        revision_middle_layer = min(num_layers // 2, num_layers - 2)

    response_hidden = _response_hidden_states(
        residual_stream,
        full_token_ids,
        prompt_len,
        response_positions_only=response_positions_only,
    )
    num_response_tokens = response_hidden.shape[1]

    deep_start_layer = math.ceil(num_layers * deep_layer_fraction)
    deep_start_layer = max(1, min(deep_start_layer, num_layers))

    device = logits_device or residual_stream.device
    response_token_ids = torch.tensor(
        full_token_ids[prompt_len:], dtype=torch.long, device=device
    )
    metadata = {
        "num_response_tokens": float(num_response_tokens),
        "deep_layer_start": float(deep_start_layer),
        "num_layers": float(num_layers),
        "jsd_threshold": float(jsd_threshold),
        "revision_middle_layer": float(revision_middle_layer),
        "revision_alpha": float(revision_alpha),
        "revision_distance_jsd": 1.0 if revision_distance == "jsd" else 0.0,
    }
    if num_response_tokens == 0:
        empty = {
            metric: 1.0 if metric == "negative_perplexity" else 0.0
            for metric in selected_metrics
        }
        if return_token_self_certainties:
            empty["token_self_certainties"] = []
        return empty | metadata

    wants_deep_thinking = bool(selected_metrics & DEEP_THINKING_METRICS)
    wants_revision = bool(selected_metrics & REVISION_METRICS)
    wants_logprobs = bool(
        selected_metrics & {"average_log_probability", "negative_perplexity"}
    )
    wants_self_certainty = (
        "self_certainty" in selected_metrics or return_token_self_certainties
    )
    wants_normalized_confidence = "normalized_confidence" in selected_metrics
    row_for_layer = {layer: row for row, layer in enumerate(layer_indices)}
    final_row = row_for_layer[num_layers - 1]
    final_logits_fn_resolved = final_logits_fn or logits_fn
    token_logprobs_t = (
        torch.empty(num_response_tokens, dtype=torch.float32, device=device)
        if wants_logprobs
        else None
    )
    token_self_certainties_t = (
        torch.empty(num_response_tokens, dtype=torch.float32, device=device)
        if wants_self_certainty
        else None
    )
    token_normalized_confidences_t = (
        torch.empty(num_response_tokens, dtype=torch.float32, device=device)
        if wants_normalized_confidence
        else None
    )
    jsd_per_layer = (
        torch.empty(
            (num_layers, num_response_tokens), dtype=torch.float32, device=device
        )
        if wants_deep_thinking
        else None
    )
    for start in range(0, num_response_tokens, logits_batch_size):
        end = min(start + logits_batch_size, num_response_tokens)
        final_hidden = response_hidden[final_row, start:end].to(device=device)
        final_logits = final_logits_fn_resolved(final_hidden)
        if final_logits.dim() == 1:
            final_logits = final_logits.unsqueeze(0)
        vocab_size = final_logits.shape[-1]
        batch_token_ids = response_token_ids[start:end]
        if wants_logprobs or wants_self_certainty or wants_normalized_confidence:
            final_log_probs = F.log_softmax(final_logits.float(), dim=-1)
            if wants_logprobs:
                too_large = batch_token_ids >= vocab_size
                if bool(torch.any(too_large).item()):
                    token_id = int(batch_token_ids[too_large][0].item())
                    raise ValueError(
                        f"token_id {token_id} is outside output vocab size {vocab_size}"
                    )
                assert token_logprobs_t is not None
                token_logprobs_t[start:end] = final_log_probs.gather(
                    1, batch_token_ids[:, None]
                ).squeeze(1)
            if wants_self_certainty:
                assert token_self_certainties_t is not None
                token_self_certainties_t[start:end] = -math.log(
                    vocab_size
                ) - final_log_probs.mean(dim=-1)
            if wants_normalized_confidence:
                assert token_normalized_confidences_t is not None
                final_probs = final_log_probs.exp()
                token_normalized_confidences_t[start:end] = 1.0 - (
                    -(final_probs * final_log_probs).sum(dim=-1) / math.log(vocab_size)
                )

        if jsd_per_layer is not None:
            for layer_idx in range(num_layers - 1):
                hidden = response_hidden[row_for_layer[layer_idx], start:end].to(
                    device=device
                )
                layer_logits = logits_fn(hidden)
                if layer_logits.dim() == 1:
                    layer_logits = layer_logits.unsqueeze(0)
                jsd_per_layer[layer_idx, start:end] = jsd_from_logits(
                    layer_logits, final_logits
                )
            jsd_per_layer[-1, start:end] = 0.0

    metric_values: dict[str, float | list[float]] = {}
    if token_logprobs_t is not None:
        average_log_probability = token_logprobs_t.mean()
        metric_values["average_log_probability"] = float(average_log_probability.item())
        metric_values["negative_perplexity"] = float(
            torch.exp(-average_log_probability).item()
        )
    if token_self_certainties_t is not None:
        metric_values["self_certainty"] = float(token_self_certainties_t.mean().item())
    if token_normalized_confidences_t is not None:
        metric_values["normalized_confidence"] = float(
            token_normalized_confidences_t.mean().item()
        )

    if wants_revision:
        revision_rows = [
            row_for_layer[layer] for layer in range(revision_middle_layer, num_layers)
        ]
        revision_hidden = response_hidden[revision_rows]
        (
            token_peak_change_t,
            token_ema_final_t,
            token_ema_mean_t,
        ) = _compute_revision_scores_from_hidden_states(
            revision_hidden,
            logits_fn,
            final_logits_fn_resolved,
            middle_layer=0,
            alpha=revision_alpha,
            distance=revision_distance,
            logits_device=device,
            logits_batch_size=logits_batch_size,
        )
        metric_values.update(
            peak_change_score=float(token_peak_change_t.mean().item()),
            ema_final_score=float(token_ema_final_t.mean().item()),
            ema_mean_score=float(token_ema_mean_t.mean().item()),
        )

    if jsd_per_layer is not None:
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
        average_depth = (
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
        deep_mask = deep_thinking_flags_t.bool()
        settled_count = ((settled_depth <= settling_depth) & deep_mask).float().sum()
        deep_count = deep_thinking_flags_t.sum()
        settled_ratio = (
            float((settled_count / deep_count).item())
            if float(deep_count.item()) > 0.0
            else 0.0
        )
        metric_values.update(
            deep_thinking_ratio=float(deep_thinking_flags_t.mean().item()),
            settled_deep_thinking_ratio=settled_ratio,
            average_deep_thinking_settling_depth=average_depth,
        )

    result = {metric: metric_values[metric] for metric in selected_metrics}
    if return_token_self_certainties:
        assert token_self_certainties_t is not None
        result["token_self_certainties"] = token_self_certainties_t.tolist()
    return result | metadata

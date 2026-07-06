from __future__ import annotations

import math

import pytest
import torch

from vllm_lens.metrics import (
    jsd_from_logits,
    kl_uniform_to_probs,
    compute_intrinsic_metrics_from_activations,
    normalize_metric_options,
)


def test_normalize_metric_options_defaults() -> None:
    assert normalize_metric_options(True) == {
        "jsd_threshold": 0.5,
        "deep_layer_fraction": 0.85,
        "logits_batch_size": 128,
        "metrics_storage": "cpu",
    }


def test_normalize_metric_options_custom() -> None:
    assert normalize_metric_options(
        {"jsd_threshold": 0.1, "deep_layer_fraction": 0.5}
    ) == {
        "jsd_threshold": 0.1,
        "deep_layer_fraction": 0.5,
        "logits_batch_size": 128,
        "metrics_storage": "cpu",
    }
    assert normalize_metric_options(
        {
            "jsd_threshold": 0.1,
            "deep_layer_fraction": 0.5,
            "logits_batch_size": 4,
            "metrics_storage": "gpu",
        }
    ) == {
        "jsd_threshold": 0.1,
        "deep_layer_fraction": 0.5,
        "logits_batch_size": 4,
        "metrics_storage": "gpu",
    }


def test_normalize_metric_options_from_openai_xargs_string() -> None:
    assert normalize_metric_options("true") == {
        "jsd_threshold": 0.5,
        "deep_layer_fraction": 0.85,
        "logits_batch_size": 128,
        "metrics_storage": "cpu",
    }
    assert normalize_metric_options(["true"]) == {
        "jsd_threshold": 0.5,
        "deep_layer_fraction": 0.85,
        "logits_batch_size": 128,
        "metrics_storage": "cpu",
    }
    assert normalize_metric_options(1) == {
        "jsd_threshold": 0.5,
        "deep_layer_fraction": 0.85,
        "logits_batch_size": 128,
        "metrics_storage": "cpu",
    }
    assert normalize_metric_options(
        '{"jsd_threshold": 0.1, "deep_layer_fraction": 0.5}'
    ) == {
        "jsd_threshold": 0.1,
        "deep_layer_fraction": 0.5,
        "logits_batch_size": 128,
        "metrics_storage": "cpu",
    }


def test_normalize_metric_options_rejects_invalid() -> None:
    with pytest.raises(TypeError):
        normalize_metric_options("yes")


def test_compute_intrinsic_metrics_from_activations() -> None:
    lm_head = torch.nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        )

    residual_stream = torch.tensor(
        [
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
            [
                [0.9, 0.0, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.0, 0.9],
            ],
        ]
    )
    full_token_ids = [0, 1, 2, 3]

    metrics = compute_intrinsic_metrics_from_activations(
        residual_stream,
        lm_head,
        full_token_ids,
        prompt_len=2,
        jsd_threshold=1.0,
        deep_layer_fraction=0.5,
    )

    assert set(metrics) == {
        "deep_thinking_ratio",
        "average_deep_thinking_settling_depth",
        "self_certainty",
        "average_log_probability",
        "num_response_tokens",
        "deep_layer_start",
        "num_layers",
        "jsd_threshold",
    }
    assert metrics["num_response_tokens"] == 2.0
    assert metrics["num_layers"] == 2.0
    assert metrics["deep_layer_start"] == 1.0
    assert metrics["jsd_threshold"] == 1.0
    assert all(math.isfinite(v) for v in metrics.values())


def test_compute_intrinsic_metrics_uses_final_logits_fn_for_targets() -> None:
    lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(2))

    residual_stream = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    full_token_ids = [0, 1]

    metrics = compute_intrinsic_metrics_from_activations(
        residual_stream,
        lm_head,
        full_token_ids,
        prompt_len=1,
        final_logits_fn=lambda hidden: lm_head(-hidden),
    )

    assert metrics["average_log_probability"] == pytest.approx(
        torch.log_softmax(torch.tensor([-1.0, 0.0]), dim=-1)[1].item()
    )


def test_final_layer_jsd_uses_final_logits_fn_when_available() -> None:
    lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(2))

    residual_stream = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    full_token_ids = [0, 1]

    metrics = compute_intrinsic_metrics_from_activations(
        residual_stream,
        lm_head,
        full_token_ids,
        prompt_len=1,
        jsd_threshold=1e-7,
        deep_layer_fraction=0.75,
        final_logits_fn=lambda hidden: lm_head(-hidden),
    )

    assert metrics["deep_thinking_ratio"] == 1.0


def test_dtr_uses_running_min_settling_depth() -> None:
    def logits_fn(hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden, -hidden], dim=-1)

    residual_stream = torch.tensor(
        [
            [[5.0]],
            [[0.0]],
            [[5.0]],
            [[0.0]],
        ]
    )
    full_token_ids = [0, 1]

    metrics = compute_intrinsic_metrics_from_activations(
        residual_stream,
        logits_fn,
        full_token_ids,
        prompt_len=1,
        jsd_threshold=0.01,
        deep_layer_fraction=0.75,
    )

    assert metrics["deep_layer_start"] == 3.0
    assert metrics["deep_thinking_ratio"] == 0.0
    assert metrics["average_deep_thinking_settling_depth"] == 0.0


def test_average_deep_thinking_settling_depth() -> None:
    final_logits = torch.tensor([10.0, -10.0])
    per_layer_logits = torch.tensor(
        [
            [[-10.0, 10.0], [-10.0, 10.0]],
            [[10.0, -10.0], [-10.0, 10.0]],
            [[10.0, -10.0], [10.0, -10.0]],
        ]
    )

    def logits_fn(hidden: torch.Tensor) -> torch.Tensor:
        return per_layer_logits.view(-1, 2).index_select(0, hidden.long().view(-1))

    residual_stream = torch.tensor(
        [
            [[0], [1]],
            [[2], [3]],
            [[4], [5]],
        ]
    )

    metrics = compute_intrinsic_metrics_from_activations(
        residual_stream,
        logits_fn,
        full_token_ids=[0, 0, 0],
        prompt_len=1,
        jsd_threshold=1e-6,
        deep_layer_fraction=0.5,
        final_logits_fn=lambda hidden: final_logits.expand(hidden.shape[0], -1),
    )

    assert metrics["deep_layer_start"] == 2.0
    assert metrics["deep_thinking_ratio"] == 1.0
    assert metrics["average_deep_thinking_settling_depth"] == 2.5


def _scalar_intrinsic_metrics(
    residual_stream: torch.Tensor,
    logits_fn,
    full_token_ids: list[int],
    prompt_len: int,
    *,
    jsd_threshold: float,
    deep_layer_fraction: float,
    final_logits_fn=None,
) -> dict[str, float]:
    num_layers = residual_stream.shape[0]
    deep_start_layer = math.ceil(num_layers * deep_layer_fraction)
    deep_start_layer = max(1, min(deep_start_layer, num_layers))

    token_logprobs: list[float] = []
    token_self_certainties: list[float] = []
    deep_thinking_flags: list[int] = []
    deep_thinking_settling_depths: list[int] = []

    for pos in range(prompt_len, len(full_token_ids)):
        prev_pos = pos - 1
        token_id = full_token_ids[pos]
        final_logits = (final_logits_fn or logits_fn)(
            residual_stream[-1, prev_pos].unsqueeze(0)
        )[0]
        final_log_probs = torch.log_softmax(final_logits.float(), dim=-1)
        token_logprobs.append(float(final_log_probs[token_id].item()))
        token_self_certainties.append(float(kl_uniform_to_probs(final_logits).item()))

        jsd_per_layer: list[float] = []
        for layer_idx in range(num_layers):
            layer_logits = logits_fn(residual_stream[layer_idx, prev_pos].unsqueeze(0))[
                0
            ]
            jsd_per_layer.append(
                float(jsd_from_logits(layer_logits, final_logits).detach())
            )

        settling_depth = num_layers
        running_min_jsd = math.inf
        for layer_idx in range(1, num_layers + 1):
            running_min_jsd = min(running_min_jsd, jsd_per_layer[layer_idx - 1])
            if running_min_jsd <= jsd_threshold:
                settling_depth = layer_idx
                break
        is_deep_thinking = int(settling_depth >= deep_start_layer)
        deep_thinking_flags.append(is_deep_thinking)
        if is_deep_thinking:
            deep_thinking_settling_depths.append(settling_depth)

    denom = max(len(token_logprobs), 1)
    depth_denom = max(len(deep_thinking_settling_depths), 1)
    return {
        "deep_thinking_ratio": float(sum(deep_thinking_flags) / denom),
        "average_deep_thinking_settling_depth": float(
            sum(deep_thinking_settling_depths) / depth_denom
        ),
        "self_certainty": float(sum(token_self_certainties) / denom),
        "average_log_probability": float(sum(token_logprobs) / denom),
        "num_response_tokens": float(len(token_logprobs)),
        "deep_layer_start": float(deep_start_layer),
        "num_layers": float(num_layers),
        "jsd_threshold": float(jsd_threshold),
    }


def test_vectorized_metrics_match_scalar_reference() -> None:
    torch.manual_seed(0)
    lm_head = torch.nn.Linear(7, 11, bias=False)
    residual_stream = torch.randn(4, 8, 7)
    full_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    expected = _scalar_intrinsic_metrics(
        residual_stream,
        lm_head,
        full_token_ids,
        prompt_len=3,
        jsd_threshold=0.03,
        deep_layer_fraction=0.5,
    )
    actual = compute_intrinsic_metrics_from_activations(
        residual_stream,
        lm_head,
        full_token_ids,
        prompt_len=3,
        jsd_threshold=0.03,
        deep_layer_fraction=0.5,
        logits_batch_size=3,
    )

    assert actual == pytest.approx(expected, abs=1e-6)


def test_response_positions_only_metrics_match_full_stream() -> None:
    torch.manual_seed(1)
    lm_head = torch.nn.Linear(5, 13, bias=False)
    residual_stream = torch.randn(3, 9, 5)
    full_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    prompt_len = 4
    response_positions = residual_stream[:, prompt_len - 1 : len(full_token_ids) - 1]

    full_metrics = compute_intrinsic_metrics_from_activations(
        residual_stream,
        lm_head,
        full_token_ids,
        prompt_len=prompt_len,
        logits_batch_size=2,
    )
    response_only_metrics = compute_intrinsic_metrics_from_activations(
        response_positions,
        lm_head,
        full_token_ids,
        prompt_len=prompt_len,
        logits_batch_size=2,
        response_positions_only=True,
    )

    assert response_only_metrics == pytest.approx(full_metrics, abs=1e-6)

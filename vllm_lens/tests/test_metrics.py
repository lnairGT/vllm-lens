from __future__ import annotations

import math

import pytest
import torch

from vllm_lens.metrics import (
    compute_intrinsic_metrics_from_activations,
    normalize_metric_options,
)


def test_normalize_metric_options_defaults() -> None:
    assert normalize_metric_options(True) == {
        "jsd_threshold": 0.02,
        "deep_layer_fraction": 0.25,
    }


def test_normalize_metric_options_custom() -> None:
    assert normalize_metric_options(
        {"jsd_threshold": 0.1, "deep_layer_fraction": 0.5}
    ) == {
        "jsd_threshold": 0.1,
        "deep_layer_fraction": 0.5,
    }


def test_normalize_metric_options_from_openai_xargs_string() -> None:
    assert normalize_metric_options("true") == {
        "jsd_threshold": 0.02,
        "deep_layer_fraction": 0.25,
    }
    assert normalize_metric_options(["true"]) == {
        "jsd_threshold": 0.02,
        "deep_layer_fraction": 0.25,
    }
    assert normalize_metric_options(1) == {
        "jsd_threshold": 0.02,
        "deep_layer_fraction": 0.25,
    }
    assert normalize_metric_options(
        '{"jsd_threshold": 0.1, "deep_layer_fraction": 0.5}'
    ) == {
        "jsd_threshold": 0.1,
        "deep_layer_fraction": 0.5,
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
        "self_certainty",
        "average_log_probability",
        "num_response_tokens",
        "deep_layer_start",
        "num_layers",
        "jsd_threshold",
    }
    assert metrics["num_response_tokens"] == 2.0
    assert metrics["num_layers"] == 2.0
    assert metrics["deep_layer_start"] == 2.0
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

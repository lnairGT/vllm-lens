from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_lens._worker_ext import (
    _metric_relative_indices,
    _select_attention_metadata,
)


def test_metric_relative_indices_skip_early_prompt_tokens() -> None:
    indices = _metric_relative_indices(
        abs_start=0,
        n_query=10,
        prompt_len=8,
        device=torch.device("cpu"),
    )

    assert indices.tolist() == [7, 8, 9]


def test_metric_relative_indices_capture_chunked_prompt_boundary() -> None:
    indices = _metric_relative_indices(
        abs_start=64,
        n_query=10,
        prompt_len=70,
        device=torch.device("cpu"),
    )

    assert indices.tolist() == [5, 6, 7, 8, 9]


def test_metric_relative_indices_capture_all_decode_positions() -> None:
    indices = _metric_relative_indices(
        abs_start=70,
        n_query=3,
        prompt_len=70,
        device=torch.device("cpu"),
    )

    assert indices.tolist() == [0, 1, 2]


def test_select_attention_metadata_uses_owner_of_query_start_loc() -> None:
    owner = SimpleNamespace(query_start_loc=torch.tensor([0, 1]))
    other = SimpleNamespace()

    selected = _select_attention_metadata({"gdn": other, "attention": owner})

    assert selected is owner

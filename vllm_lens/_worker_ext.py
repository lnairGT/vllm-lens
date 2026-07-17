"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens.metrics import (
    compute_intrinsic_metrics_from_activations,
    normalize_metric_options,
    required_metric_layers,
)
from vllm_lens._helpers.types import SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the transformer decoder layers regardless of model architecture."""
    # Module.__getattr__ returns Tensor | Module, so pyright can't narrow
    # through chained attribute access.  Use Any for duck-typed traversal.
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if (
        hasattr(m, "model")
        and hasattr(m.model, "decoder")
        and hasattr(m.model.decoder, "layers")
    ):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.language_model.model.layers, "
        "model.model.decoder.layers, or model.model.layers"
    )


def _get_logits_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Find the model's hidden-state-to-logits function."""
    compute_logits = getattr(model, "compute_logits", None)
    if callable(compute_logits):

        def _compute(hidden_states: torch.Tensor) -> torch.Tensor:
            logits = compute_logits(hidden_states)
            if logits is None:
                raise ValueError("Model compute_logits returned None")
            return cast(torch.Tensor, logits)

        return _compute

    raise AttributeError(f"Cannot find compute_logits on {type(model).__name__}")


def _get_final_logits_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Find a logits function that applies the model's final norm first."""
    logits_fn = _get_logits_fn(model)
    m: Any = model
    final_norm = None
    if hasattr(m, "model") and hasattr(m.model, "norm"):
        final_norm = m.model.norm
    elif hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        inner = m.language_model.model
        if hasattr(inner, "norm"):
            final_norm = inner.norm

    if final_norm is None:
        return logits_fn

    def _compute(hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = final_norm(hidden_states)
        if isinstance(normalized, tuple):
            normalized = normalized[0]
        return logits_fn(cast(torch.Tensor, normalized))

    return _compute


def _iter_attention_metadata(attn_metadata: Any) -> list[Any]:
    """Return concrete attention metadata objects from vLLM's wrappers."""
    if attn_metadata is None:
        return []
    if isinstance(attn_metadata, list):
        entries: list[Any] = []
        for item in attn_metadata:
            entries.extend(_iter_attention_metadata(item))
        return entries
    if isinstance(attn_metadata, dict):
        return list(attn_metadata.values())
    return [attn_metadata]


def _select_attention_metadata(attn_metadata: Any) -> Any | None:
    """Find the metadata object that owns query_start_loc."""
    for meta in _iter_attention_metadata(attn_metadata):
        if hasattr(meta, "query_start_loc"):
            return meta
    return None


def _request_prompt_len(runner: Any, req_state: Any, batch_idx: int) -> int | None:
    """Best-effort prompt length lookup across vLLM versions."""
    for attr in ("num_prompt_tokens", "prompt_len"):
        value = getattr(req_state, attr, None)
        if value is not None:
            return int(value.item() if isinstance(value, torch.Tensor) else value)

    prompt_token_ids = getattr(req_state, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)

    inputs = getattr(req_state, "inputs", None)
    prompt_token_ids = getattr(inputs, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)

    input_batch = getattr(runner, "input_batch", None)
    num_prompt_tokens = getattr(input_batch, "num_prompt_tokens", None)
    if num_prompt_tokens is None:
        return None
    if isinstance(num_prompt_tokens, torch.Tensor):
        if num_prompt_tokens.dim() == 0:
            return int(num_prompt_tokens.item())
        return int(num_prompt_tokens[batch_idx].item())
    if isinstance(num_prompt_tokens, (list, tuple)):
        return int(num_prompt_tokens[batch_idx])
    return int(num_prompt_tokens)


def _sequence_start(meta: Any, req_state: Any, batch_idx: int, n_query: int) -> int:
    """Absolute position for the first token in a request's current slice."""
    seq_lens: Any = getattr(meta, "seq_lens", None)
    if seq_lens is not None:
        sl = seq_lens[batch_idx]
        sl_val = sl.item() if isinstance(sl, torch.Tensor) else int(sl)
        return int(sl_val - n_query)

    for attr in ("num_computed_tokens", "num_tokens"):
        value = getattr(req_state, attr, None)
        if value is not None:
            val = value.item() if isinstance(value, torch.Tensor) else int(value)
            return int(val - n_query)

    return 0


def _metric_relative_indices(
    abs_start: int,
    n_query: int,
    prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Token indices in the current slice needed for intrinsic metrics.

    The final prompt position scores the first generated token. Decode
    positions score later generated tokens. Earlier prompt positions are
    irrelevant for response-only metrics.
    """
    first_metric_pos = max(prompt_len - 1, 0)
    rel_start = max(first_metric_pos - abs_start, 0)
    if rel_start >= n_query:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.arange(rel_start, n_query, dtype=torch.long, device=device)


def _find_steering_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[SteeringVector]:
    """Find all steering configs that apply to an internal request ID.

    Matches by ``"{external_id}-"`` prefix (async path: vLLM appends
    ``"-{random_suffix}"`` to external IDs) and by ``_steering_id``
    sentinel in ``extra_args`` (offline path).
    """
    results: list[SteeringVector] = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    # Offline path stores a lightweight string key in extra_args
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in extension._steering_data:
            results.extend(extension._steering_data[steering_id])
    return results


def norm_match(
    residual: torch.Tensor,
    steering: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a steering vector to match the L2 norm of the residual stream.

    Norm matching approach from the Activation Oracles paper
    (arXiv:2512.15674):

        h'_i = h_i + ‖h_i‖ · v_i / ‖v_i‖

    This rescales the steering vector so its magnitude matches the
    residual before addition, ensuring activations of varying provenance
    are automatically scaled to a consistent magnitude.
    """
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.
    """
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)  # (hidden,) or (n_pos, hidden)

        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(target[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            # 3D: position-specific
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else list(range(vec.shape[0]))
            )
            abs_end = abs_start + n_tokens
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic, separated so _make_hook can wrap it in try/except."""
    if not is_forward_context_available():
        return None

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None

    req_ids = runner.input_batch.req_ids

    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    selected_attn_metadata = _select_attention_metadata(attn_metadata)
    if selected_attn_metadata is None:
        logger.warning(
            "No attention metadata with query_start_loc found "
            "(metadata type: %s). Skipping hook for this step.",
            type(attn_metadata).__name__,
        )
        return None
    query_start_loc: Int[torch.Tensor, "num_reqs_plus1"] = getattr(  # type: ignore[reportUndefinedVariable]
        selected_attn_metadata, "query_start_loc"
    )

    # --- Phase 1: detect steering requests --------------------------
    per_req_steering: list[list[SteeringVector]] = []
    needs_steering = False
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        configs = _find_steering_configs(extension, req_id, extra)
        per_req_steering.append(configs)
        if configs:
            needs_steering = True

    # --- Phase 2: apply steering ------------------------------------
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if needs_steering:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
        else:
            modified_output = output.clone()
            target = modified_output

        for i in range(num_reqs):
            if not per_req_steering[i]:
                continue
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            n_query = end - start
            req_state = runner.requests.get(req_ids[i])
            abs_start = _sequence_start(
                selected_attn_metadata, req_state, i, n_query
            )
            _apply_steering(
                per_req_steering[i], layer_idx, target, start, end, abs_start
            )

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if getattr(extension, "_should_capture", True):
        capture_src = modified_output if modified_output is not None else output
        hidden_states: Float[torch.Tensor, "total_tokens hidden_dim"]  # type: ignore[reportUndefinedVariable]
        if isinstance(capture_src, tuple):
            if capture_src[1] is not None:
                hidden_states = capture_src[0] + capture_src[1]
            else:
                hidden_states = capture_src[0]
        else:
            hidden_states = capture_src

        for i in range(num_reqs):
            req_id = req_ids[i]
            req_state = runner.requests.get(req_id)
            if req_state is None or req_state.sampling_params is None:
                continue
            extra = req_state.sampling_params.extra_args
            if not extra:
                continue

            output_residual_stream = extra.get("output_residual_stream")
            output_intrinsic_metrics = extra.get("output_intrinsic_metrics")
            if output_residual_stream is None and output_intrinsic_metrics is None:
                continue

            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            n_query = end - start

            wants_activation_layer = output_residual_stream is not None and not (
                isinstance(output_residual_stream, list)
                and layer_idx not in output_residual_stream
            )
            if wants_activation_layer:
                # Blocking .cpu() benchmarked faster than non_blocking + event sync
                activation: Float[torch.Tensor, "seq_len hidden_dim"] = hidden_states[  # type: ignore[reportUndefinedVariable]
                    start:end
                ].cpu()

                if req_id not in extension._captured_states:
                    extension._captured_states[req_id] = {}
                layer_states = extension._captured_states[req_id]
                if layer_idx not in layer_states:
                    layer_states[layer_idx] = []
                layer_states[layer_idx].append(activation)

            if output_intrinsic_metrics is not None:
                prompt_len = _request_prompt_len(runner, req_state, i)
                if prompt_len is None:
                    logger.warning(
                        "Cannot determine prompt length for %s; skipping metrics "
                        "capture on this step.",
                        req_id,
                    )
                    continue
                abs_start = _sequence_start(
                    selected_attn_metadata, req_state, i, n_query
                )
                rel_indices = _metric_relative_indices(
                    abs_start, n_query, prompt_len, hidden_states.device
                )
                if rel_indices.numel() == 0:
                    continue
                metric_options = normalize_metric_options(output_intrinsic_metrics)
                metric_layers = required_metric_layers(
                    len(_get_layers(runner.model)),
                    metric_options["metrics"],
                    metric_options["revision_middle_layer"],
                    return_token_self_certainties=metric_options[
                        "return_token_self_certainties"
                    ],
                )
                if layer_idx not in metric_layers:
                    continue
                metric_activation = hidden_states[start:end].index_select(
                    0, rel_indices
                )
                if metric_options["metrics_storage"] == "gpu":
                    metric_activation = metric_activation.detach().clone()
                else:
                    metric_activation = metric_activation.detach().cpu()

                if req_id not in extension._metric_states:
                    extension._metric_states[req_id] = {}
                metric_layer_states = extension._metric_states[req_id]
                if layer_idx not in metric_layer_states:
                    metric_layer_states[layer_idx] = []
                metric_layer_states[layer_idx].append(metric_activation)

    return modified_output


def _make_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook(
        _module: torch.nn.Module,
        _input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Forward hook: apply steering vectors then capture activations.

        Returns the modified output if any steering was applied, ``None``
        otherwise (so PyTorch leaves the original output untouched).
        """
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime.

    Configured via the ``worker_extension_cls`` engine arg. vLLM dynamically
    adds this class as a base of Worker
    (``Worker.__bases__ += (HiddenStatesExtension,)``), so ``self`` is the
    Worker instance and its methods are callable via
    ``collective_rpc("method_name")``.

    It doesn't extend Worker directly — vLLM handles that injection.
    """

    if TYPE_CHECKING:
        model_runner: Any  # Provided by Worker at runtime
        rank: int
        parallel_config: ParallelConfig

    # Per-request captured activations:
    # internal_req_id → { layer_idx → [tensor, ...] }
    _captured_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _metric_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _hooks_installed: bool = False

    # Per-request steering configs:
    # key (external_req_id or _steering_id) → list of SteeringVector
    _steering_data: dict[str, list[SteeringVector]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Requires ``enforce_eager=True`` in engine args — otherwise
        ``@support_torch_compile`` would compile the forward graph and
        hooks won't fire.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}
        self._metric_states = {}
        self._steering_data = {}

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layers = _get_layers(self.model_runner.model)
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            layer.register_forward_hook(_make_hook(self, layer_idx))

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store steering vectors for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``SteeringVector`` instances, validates layer indices
        against the model, moves activation tensors to GPU in the model's
        dtype, and stores them keyed by *key* (an external request ID or a
        synthetic ``_steering_id``).
        """
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)

        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype

        num_layers = len(_get_layers(self.model_runner.model))
        vectors: list[SteeringVector] = []

        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )

            vectors.append(
                sv.model_copy(
                    update={
                        "activations": sv.activations.to(device=device, dtype=dtype)
                    }
                )
            )

        self._steering_data[key] = vectors

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._steering_data.pop(key, None)

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
                logger.debug("Cleared leaked activations for %s", req_id)
        for req_id in list(self._metric_states):
            if req_id.startswith(prefix):
                del self._metric_states[req_id]
                logger.debug("Cleared leaked metric activations for %s", req_id)

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle for safe ZMQ
        transport.

        Returns a dict when deserialized::

            {
                "activations": {
                    "residual_stream": Tensor,  # (n_layers, total_pos, d_model)
                }
            }

        Layers are stacked in ascending order along dim 0.
        Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                layer_dict = self._captured_states.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
                    torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
                ]
                stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
                    torch.stack(per_layer, dim=0)
                )
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps(
                        {
                            "activations": {"residual_stream": stacked},
                        }
                    )
                )
        return None

    def get_intrinsic_metrics(
        self,
        external_req_id: str,
        full_token_ids: list[int],
        prompt_len: int,
        options: dict[str, Any] | bool | None = None,
    ) -> bytes | None:
        """Compute intrinsic metrics for a captured request.

        Metrics require a complete residual stream and the model LM head,
        so this is computed on the worker before activations are serialized
        back to the driver.
        """
        prefix = f"{external_req_id}-"
        for req_id, layer_dict in self._metric_states.items():
            if not req_id.startswith(prefix):
                continue

            sorted_indices = sorted(layer_dict.keys())
            if not sorted_indices:
                return None

            total_layers = len(_get_layers(self.model_runner.model))
            metric_options = normalize_metric_options(options)
            expected_indices = required_metric_layers(
                total_layers,
                metric_options["metrics"],
                metric_options["revision_middle_layer"],
                return_token_self_certainties=metric_options[
                    "return_token_self_certainties"
                ],
            )
            if sorted_indices != expected_indices:
                logger.warning(
                    "Intrinsic metrics require decoder layers %s; got %s",
                    expected_indices,
                    sorted_indices,
                )
                return None

            per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
                torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
            ]
            residual_stream: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
                torch.stack(per_layer, dim=0)
            )

            metric_options["response_positions_only"] = True
            metrics = compute_intrinsic_metrics_from_activations(
                residual_stream,
                _get_logits_fn(self.model_runner.model),
                full_token_ids,
                prompt_len,
                logits_device=next(self.model_runner.model.parameters()).device,
                final_logits_fn=_get_final_logits_fn(self.model_runner.model),
                layer_indices=sorted_indices,
                num_model_layers=total_layers,
                **metric_options,
            )
            self._metric_states.pop(req_id, None)
            return _ZSTD_COMPRESSOR.compress(pickle.dumps(metrics))

        return None

    def _debug_captured_states_count(self) -> int:
        """Return the number of captured-state entries (for testing)."""
        return len(self._captured_states) + len(self._metric_states)

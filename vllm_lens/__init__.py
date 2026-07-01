from importlib.metadata import PackageNotFoundError, version

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    serialize_activations,
    serialize_tensor,
)
from vllm_lens._helpers.types import SteeringVector
from vllm_lens.metrics import (
    compute_intrinsic_metrics_from_activations,
    normalize_metric_options,
)

try:
    __version__ = version("vllm-lens")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "decode_activations",
    "deserialize_tensor",
    "serialize_activations",
    "serialize_tensor",
    "SteeringVector",
    "compute_intrinsic_metrics_from_activations",
    "normalize_metric_options",
    "__version__",
]

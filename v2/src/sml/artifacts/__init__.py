"""Artifact ownership package."""

from sml.artifacts.arrays import load_safetensors_payload
from sml.artifacts.verify import VerificationResult, verify_artifact

__all__ = [
    "VerificationResult",
    "load_safetensors_payload",
    "verify_artifact",
]

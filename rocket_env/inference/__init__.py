"""
Inference Package

Lightweight deployment utilities for running trained models on
embedded systems like Raspberry Pi 4.

Dependencies:
- numpy
- onnxruntime (no PyTorch required!)
- scipy (for normalization stats)
"""

from .onnx_runner import ONNXRunner
from .controller import RocketController

__all__ = [
    "ONNXRunner",
    "RocketController",
]

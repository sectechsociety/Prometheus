"""
Model inference module for Prometheus.
Handles loading and running the fine-tuned prompt enhancement model.
"""

from .inference import PrometheusLightModel as PrometheusModel
from .inference import get_model

__all__ = ["get_model", "PrometheusModel"]

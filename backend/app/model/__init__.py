"""
Model inference module for Prometheus.
Handles loading and running the fine-tuned prompt enhancement model.
"""

from .inference import get_model, PrometheusLightModel as PrometheusModel

__all__ = ["get_model", "PrometheusModel"]

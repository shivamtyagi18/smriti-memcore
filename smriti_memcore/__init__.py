"""
SMRITI v2: Neuro-Inspired EXperience-Unified System
A novel memory architecture for AI agents inspired by human cognition.
"""

from smriti_memcore.core import SMRITI
from smriti_memcore.models import Memory, SalienceScore, MemorySource, Modality, SmritiConfig
from smriti_memcore.metrics import SmritiMetrics

__version__ = "0.1.3"
__all__ = ["SMRITI", "SmritiConfig", "SmritiMetrics", "Memory", "SalienceScore", "MemorySource", "Modality"]


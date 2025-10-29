"""
Simple Seizure Predictor

A streamlined deep learning framework for subject-dependent seizure prediction.
Based on Koutsouvelis et al. 2024 - CNN-Transformer architecture.
"""

__version__ = "1.0.0"

from .data_loader import EEGDataLoader, SubjectDataset
from .model import CNNTransformerSeizurePredictor, create_model
from .train import SeizurePredictor
from .subject_selector import SubjectSelector

__all__ = [
    'EEGDataLoader',
    'SubjectDataset',
    'CNNTransformerSeizurePredictor',
    'create_model',
    'SeizurePredictor',
    'SubjectSelector',
]

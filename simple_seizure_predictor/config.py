"""
Configuration for Simple Seizure Predictor
Based on Koutsouvelis et al. 2024 - CNN-Transformer approach
"""

# Data Configuration
DATA_CONFIG = {
    # Path to your EDF files
    'data_path': '/path/to/your/edf/files',

    # Preprocessing
    'target_freq': 256,  # Resample to 256 Hz
    'lowcut': 0.5,       # High-pass filter
    'highcut': 45,       # Low-pass filter
    'epoch_length': 5,   # 5-second windows

    # Subject selection
    'min_seizures': 2,   # Minimum seizures per subject for training

    # Time windows
    'preictal_minutes': 60,  # How far before seizure to consider preictal
    'interictal_hours_before': 4,  # Exclude 4 hours before seizure for interictal
    'postictal_hours': 1,   # Exclude 1 hour after seizure
}

# Model Configuration (from the paper)
MODEL_CONFIG = {
    # CNN layers
    'cnn_filters': [32, 64, 128],
    'kernel_size': (3, 3),
    'pool_size': (2, 2),
    'dropout_cnn': 0.1,

    # Transformer
    'num_heads': 8,
    'key_dim': 64,
    'num_transformer_blocks': 2,
    'ff_dim': 64,
    'dropout_transformer': 0.3,
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 20,
    'learning_rate': 0.001,
    'validation_split': 0.1,

    # Data augmentation
    'use_augmentation': True,
    'preictal_overlap': 0.66,  # 66% overlap for oversampling
    'augmentation_factor': 3,   # 3x oversampling
}

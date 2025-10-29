"""
Training script for subject-dependent seizure prediction
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split

from data_loader import EEGDataLoader, SubjectDataset
from model import create_model, get_callbacks
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeizurePredictor:
    """Main class for training subject-dependent seizure prediction models"""

    def __init__(self, config=None):
        self.data_config = DATA_CONFIG if config is None else config.get('data', DATA_CONFIG)
        self.model_config = MODEL_CONFIG if config is None else config.get('model', MODEL_CONFIG)
        self.train_config = TRAIN_CONFIG if config is None else config.get('train', TRAIN_CONFIG)

        self.data_loader = EEGDataLoader(
            target_freq=self.data_config['target_freq'],
            lowcut=self.data_config['lowcut'],
            highcut=self.data_config['highcut'],
            epoch_length=self.data_config['epoch_length']
        )

    def prepare_subject_data(self, file_list, seizure_annotations):
        """
        Prepare data for a single subject

        Args:
            file_list: List of EDF file paths for this subject
            seizure_annotations: Dict mapping file_path -> list of (start_sec, end_sec) tuples

        Returns:
            X_train, X_val, y_train, y_val
        """
        dataset = SubjectDataset('subject', self.data_loader)

        for file_path in file_list:
            seizure_times = seizure_annotations.get(file_path, None)
            dataset.add_file(
                file_path,
                seizure_times,
                preictal_minutes=self.data_config['preictal_minutes'],
                interictal_hours_before=self.data_config['interictal_hours_before'],
                postictal_hours=self.data_config['postictal_hours']
            )

        # Get balanced data
        X, y = dataset.get_data(
            balance_classes=True,
            preictal_overlap=self.train_config['preictal_overlap'],
            augmentation_factor=self.train_config['augmentation_factor']
        )

        if X is None:
            return None, None, None, None

        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.train_config['validation_split'],
            stratify=y,
            random_state=42
        )

        logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

        return X_train, X_val, y_train, y_val

    def train_subject_model(self, subject_id, X_train, X_val, y_train, y_val,
                           save_dir='models'):
        """
        Train a model for a single subject

        Args:
            subject_id: Subject identifier
            X_train, X_val, y_train, y_val: Training and validation data
            save_dir: Directory to save model

        Returns:
            Trained model and history
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for subject: {subject_id}")
        logger.info(f"{'='*50}\n")

        # Create model
        input_shape = X_train.shape[1:]  # (time_steps, n_channels)
        model = create_model(input_shape, self.model_config)

        # Build to see architecture
        model.build((None,) + input_shape)
        model.summary()

        # Setup callbacks
        save_path = Path(save_dir) / subject_id
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(save_path / 'best_model.h5')

        callbacks = get_callbacks(
            patience=self.train_config['early_stopping_patience'],
            checkpoint_path=checkpoint_path
        )

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.train_config['batch_size'],
            epochs=self.train_config['max_epochs'],
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        model.save(str(save_path / 'final_model.h5'))

        # Save training history
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(save_path / 'history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"\nModel saved to {save_path}")

        return model, history

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model on test data"""
        results = model.evaluate(X_test, y_test, verbose=0)

        metrics = {}
        for name, value in zip(model.metrics_names, results):
            metrics[name] = float(value)
            logger.info(f"{name}: {value:.4f}")

        return metrics

    def predict(self, model, X):
        """Make predictions"""
        predictions = model.predict(X, verbose=0)
        return predictions


def main():
    """Example usage"""
    predictor = SeizurePredictor()

    # Example: You would load your data here
    # For demonstration, using random data
    subject_id = "test_subject"

    # Simulate data (replace with actual file loading)
    logger.info("Loading subject data...")

    # Example file list and annotations
    file_list = [
        "/path/to/subject/file1.edf",
        "/path/to/subject/file2.edf",
    ]

    seizure_annotations = {
        "/path/to/subject/file1.edf": [(100, 150), (500, 550)],  # Two seizures
        "/path/to/subject/file2.edf": [(200, 250)],  # One seizure
    }

    # Prepare data
    X_train, X_val, y_train, y_val = predictor.prepare_subject_data(
        file_list, seizure_annotations
    )

    if X_train is not None:
        # Train model
        model, history = predictor.train_subject_model(
            subject_id, X_train, X_val, y_train, y_val
        )

        # Evaluate
        metrics = predictor.evaluate_model(model, X_val, y_val)

        logger.info(f"\nFinal validation metrics: {metrics}")
    else:
        logger.error("Failed to load data")


if __name__ == "__main__":
    main()

"""
Utility for making predictions with trained models
"""

import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import json

from data_loader import EEGDataLoader
from config import DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeizurePredictor:
    """Make predictions using a trained model"""

    def __init__(self, model_path, data_config=None):
        """
        Args:
            model_path: Path to saved model (.h5 file)
            data_config: Data configuration (uses default if None)
        """
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")

        self.data_config = DATA_CONFIG if data_config is None else data_config

        self.data_loader = EEGDataLoader(
            target_freq=self.data_config['target_freq'],
            lowcut=self.data_config['lowcut'],
            highcut=self.data_config['highcut'],
            epoch_length=self.data_config['epoch_length']
        )

    def predict_from_file(self, edf_path, threshold=0.5):
        """
        Make predictions on an EDF file

        Args:
            edf_path: Path to EDF file
            threshold: Probability threshold for positive prediction

        Returns:
            Dictionary with predictions and statistics
        """
        logger.info(f"Processing {edf_path}")

        # Load and preprocess
        raw = self.data_loader.load_edf(edf_path)
        if raw is None:
            return None

        data = self.data_loader.preprocess(raw)
        epochs = self.data_loader.create_epochs(data)

        logger.info(f"Created {len(epochs)} epochs")

        # Predict
        predictions = self.model.predict(epochs, verbose=0)

        # Analyze predictions
        probabilities = predictions[:, 0]
        binary_predictions = (probabilities > threshold).astype(int)

        n_positive = np.sum(binary_predictions)
        n_total = len(binary_predictions)

        # Find high-risk periods (consecutive positive predictions)
        high_risk_periods = self._find_high_risk_periods(
            binary_predictions,
            min_duration=3  # At least 3 consecutive epochs
        )

        results = {
            'file': edf_path,
            'n_epochs': n_total,
            'n_positive': int(n_positive),
            'positive_rate': float(n_positive / n_total),
            'mean_probability': float(probabilities.mean()),
            'max_probability': float(probabilities.max()),
            'predictions': probabilities.tolist(),
            'high_risk_periods': high_risk_periods,
            'threshold': threshold
        }

        logger.info(f"Positive epochs: {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)")
        logger.info(f"Mean probability: {results['mean_probability']:.4f}")
        logger.info(f"Found {len(high_risk_periods)} high-risk periods")

        return results

    def _find_high_risk_periods(self, binary_predictions, min_duration=3):
        """
        Find periods of consecutive positive predictions

        Args:
            binary_predictions: Array of 0/1 predictions
            min_duration: Minimum consecutive positive epochs

        Returns:
            List of (start_epoch, end_epoch, duration) tuples
        """
        periods = []
        start = None

        for i, pred in enumerate(binary_predictions):
            if pred == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    duration = i - start
                    if duration >= min_duration:
                        periods.append({
                            'start_epoch': int(start),
                            'end_epoch': int(i),
                            'duration_epochs': int(duration),
                            'duration_seconds': int(duration * self.data_config['epoch_length'])
                        })
                    start = None

        # Handle case where recording ends during high-risk period
        if start is not None:
            duration = len(binary_predictions) - start
            if duration >= min_duration:
                periods.append({
                    'start_epoch': int(start),
                    'end_epoch': int(len(binary_predictions)),
                    'duration_epochs': int(duration),
                    'duration_seconds': int(duration * self.data_config['epoch_length'])
                })

        return periods

    def predict_batch(self, edf_files, threshold=0.5, save_results=None):
        """
        Make predictions on multiple files

        Args:
            edf_files: List of EDF file paths
            threshold: Probability threshold
            save_results: Path to save results (JSON)

        Returns:
            List of prediction results
        """
        all_results = []

        for edf_file in edf_files:
            try:
                result = self.predict_from_file(edf_file, threshold)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {edf_file}: {e}")
                continue

        if save_results:
            with open(save_results, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {save_results}")

        return all_results

    def generate_alert_summary(self, results):
        """
        Generate a human-readable summary of predictions

        Args:
            results: Prediction results from predict_from_file

        Returns:
            String summary
        """
        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"Seizure Prediction Alert Summary")
        summary.append(f"{'='*60}")
        summary.append(f"File: {results['file']}")
        summary.append(f"Total epochs analyzed: {results['n_epochs']}")
        summary.append(f"Positive predictions: {results['n_positive']} ({100*results['positive_rate']:.1f}%)")
        summary.append(f"Average risk: {100*results['mean_probability']:.1f}%")
        summary.append(f"\nHigh-risk periods detected: {len(results['high_risk_periods'])}")

        if results['high_risk_periods']:
            summary.append(f"\nDetailed high-risk periods:")
            for i, period in enumerate(results['high_risk_periods'], 1):
                summary.append(
                    f"  {i}. Epochs {period['start_epoch']}-{period['end_epoch']} "
                    f"({period['duration_seconds']}s)"
                )

            # Calculate time to potential seizure (assuming prediction is accurate)
            preictal_minutes = self.data_config['preictal_minutes']
            first_period = results['high_risk_periods'][0]
            warning_time = preictal_minutes - (first_period['start_epoch'] * self.data_config['epoch_length'] / 60)
            if warning_time > 0:
                summary.append(f"\n⚠️  ALERT: Potential seizure in ~{warning_time:.0f} minutes")
            else:
                summary.append(f"\n⚠️  ALERT: Patient may be in preictal state")
        else:
            summary.append(f"\n✓ No sustained high-risk periods detected")

        summary.append(f"{'='*60}\n")

        return '\n'.join(summary)


def main():
    """Example usage"""
    # Path to trained model
    model_path = "trained_models/subject_001/best_model.h5"

    # Path to new EDF file to predict
    edf_file = "/path/to/new/recording.edf"

    # Create predictor
    predictor = SeizurePredictor(model_path)

    # Make prediction
    results = predictor.predict_from_file(edf_file, threshold=0.5)

    if results:
        # Print summary
        summary = predictor.generate_alert_summary(results)
        print(summary)

        # Save results
        output_file = "prediction_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()

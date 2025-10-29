"""
Data loading and preprocessing for seizure prediction
Handles EDF files with MNE
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataLoader:
    """Load and preprocess EEG data from EDF files"""

    def __init__(self, target_freq=256, lowcut=0.5, highcut=45, epoch_length=5):
        self.target_freq = target_freq
        self.lowcut = lowcut
        self.highcut = highcut
        self.epoch_length = epoch_length

        # Suppress MNE logging
        mne.set_log_level('WARNING')

    def load_edf(self, file_path: str) -> mne.io.Raw:
        """Load EDF file with MNE"""
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            return raw
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def preprocess(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Minimal preprocessing as in the paper:
        1. Re-reference to common average
        2. Bandpass filter 0.5-45 Hz
        3. Resample to target frequency
        """
        # Re-reference to common average
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # Bandpass filter
        raw.filter(self.lowcut, self.highcut, fir_design='firwin',
                   verbose=False, n_jobs=1)

        # Resample if needed
        if raw.info['sfreq'] != self.target_freq:
            raw.resample(self.target_freq, verbose=False)

        # Get data as numpy array (channels x time)
        data = raw.get_data()

        return data

    def create_epochs(self, data: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
        """
        Split continuous data into epochs

        Args:
            data: (channels, time) array
            labels: Optional labels for each timepoint

        Returns:
            epochs: (n_epochs, time_per_epoch, channels)
            epoch_labels: (n_epochs,) if labels provided
        """
        n_channels, n_samples = data.shape
        samples_per_epoch = int(self.epoch_length * self.target_freq)

        # Calculate number of complete epochs
        n_epochs = n_samples // samples_per_epoch

        # Trim to complete epochs
        data = data[:, :n_epochs * samples_per_epoch]

        # Reshape to epochs: (channels, n_epochs, samples_per_epoch)
        data = data.reshape(n_channels, n_epochs, samples_per_epoch)

        # Transpose to: (n_epochs, samples_per_epoch, channels)
        epochs = np.transpose(data, (1, 2, 0))

        if labels is not None:
            # Take label from middle of each epoch
            labels = labels[:n_epochs * samples_per_epoch]
            labels = labels.reshape(n_epochs, samples_per_epoch)
            epoch_labels = labels[:, samples_per_epoch // 2]
            return epochs, epoch_labels

        return epochs

    def load_and_preprocess_file(self, file_path: str,
                                 seizure_times: List[Tuple[float, float]] = None,
                                 preictal_minutes: int = 60,
                                 interictal_hours_before: int = 4,
                                 postictal_hours: int = 1) -> Dict:
        """
        Load a single EDF file and create labeled epochs

        Args:
            file_path: Path to EDF file
            seizure_times: List of (start_sec, end_sec) tuples for seizures
            preictal_minutes: Minutes before seizure to label as preictal
            interictal_hours_before: Hours before preictal to exclude from interictal
            postictal_hours: Hours after seizure to exclude

        Returns:
            Dictionary with 'epochs' and 'labels'
        """
        logger.info(f"Loading {file_path}")

        # Load and preprocess
        raw = self.load_edf(file_path)
        if raw is None:
            return None

        data = self.preprocess(raw)
        n_channels, n_samples = data.shape
        duration_sec = n_samples / self.target_freq

        # Create labels for each timepoint
        # 0 = interictal, 1 = preictal, -1 = exclude (ictal/postictal)
        labels = np.zeros(n_samples, dtype=np.int8)

        if seizure_times:
            preictal_sec = preictal_minutes * 60
            interictal_exclude_sec = interictal_hours_before * 3600
            postictal_sec = postictal_hours * 3600

            for start_sec, end_sec in seizure_times:
                # Convert to sample indices
                start_idx = int(start_sec * self.target_freq)
                end_idx = int(end_sec * self.target_freq)

                # Mark ictal period as exclude
                labels[start_idx:end_idx] = -1

                # Mark postictal as exclude
                postictal_end = min(end_idx + int(postictal_sec * self.target_freq), n_samples)
                labels[end_idx:postictal_end] = -1

                # Mark preictal
                preictal_start = max(start_idx - int(preictal_sec * self.target_freq), 0)
                labels[preictal_start:start_idx] = 1

                # Mark period before preictal as exclude (to avoid contamination)
                exclude_start = max(preictal_start - int(interictal_exclude_sec * self.target_freq), 0)
                labels[exclude_start:preictal_start] = -1

        # Create epochs
        epochs, epoch_labels = self.create_epochs(data, labels)

        # Filter out excluded epochs
        valid_mask = epoch_labels != -1
        epochs = epochs[valid_mask]
        epoch_labels = epoch_labels[valid_mask]

        logger.info(f"Created {len(epochs)} epochs: "
                   f"{np.sum(epoch_labels == 0)} interictal, "
                   f"{np.sum(epoch_labels == 1)} preictal")

        return {
            'epochs': epochs,
            'labels': epoch_labels,
            'file_path': file_path,
            'duration_sec': duration_sec
        }


class SubjectDataset:
    """Manage data for a single subject across multiple files"""

    def __init__(self, subject_id: str, data_loader: EEGDataLoader):
        self.subject_id = subject_id
        self.data_loader = data_loader
        self.files = []
        self.all_epochs = []
        self.all_labels = []

    def add_file(self, file_path: str, seizure_times: List[Tuple[float, float]] = None, **kwargs):
        """Add a file to this subject's dataset"""
        result = self.data_loader.load_and_preprocess_file(file_path, seizure_times, **kwargs)

        if result is not None:
            self.files.append(file_path)
            self.all_epochs.append(result['epochs'])
            self.all_labels.append(result['labels'])

    def get_data(self, balance_classes: bool = True,
                 preictal_overlap: float = 0.66,
                 augmentation_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all data for this subject

        Args:
            balance_classes: Whether to balance interictal/preictal
            preictal_overlap: Overlap for preictal oversampling
            augmentation_factor: How many times to augment preictal

        Returns:
            X: (n_samples, time, channels)
            y: (n_samples,) labels
        """
        if not self.all_epochs:
            return None, None

        X = np.concatenate(self.all_epochs, axis=0)
        y = np.concatenate(self.all_labels, axis=0)

        if balance_classes:
            X, y = self._balance_classes(X, y, preictal_overlap, augmentation_factor)

        return X, y

    def _balance_classes(self, X, y, preictal_overlap, augmentation_factor):
        """Balance classes by oversampling preictal and downsampling interictal"""
        preictal_mask = y == 1
        interictal_mask = y == 0

        X_preictal = X[preictal_mask]
        X_interictal = X[interictal_mask]

        # Oversample preictal with overlap (simple approach: repeat with shift)
        if len(X_preictal) > 0 and augmentation_factor > 1:
            X_preictal_aug = [X_preictal]
            for _ in range(augmentation_factor - 1):
                X_preictal_aug.append(X_preictal)  # Could add noise/augmentation here
            X_preictal = np.concatenate(X_preictal_aug, axis=0)

        # Match interictal to preictal count
        n_preictal = len(X_preictal)
        if len(X_interictal) > n_preictal:
            # Randomly sample interictal
            indices = np.random.choice(len(X_interictal), n_preictal, replace=False)
            X_interictal = X_interictal[indices]

        # Combine
        X_balanced = np.concatenate([X_interictal, X_preictal], axis=0)
        y_balanced = np.concatenate([
            np.zeros(len(X_interictal)),
            np.ones(len(X_preictal))
        ])

        # Shuffle
        indices = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]

        logger.info(f"Balanced dataset: {len(X_balanced)} total epochs "
                   f"({np.sum(y_balanced==0)} interictal, {np.sum(y_balanced==1)} preictal)")

        return X_balanced, y_balanced

"""
Utility to select subjects with sufficient seizures from your dataset
"""

import pandas as pd
from pathlib import Path
import mne
import logging
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubjectSelector:
    """Select subjects with sufficient seizures for training"""

    def __init__(self, data_path, min_seizures=2):
        """
        Args:
            data_path: Path to root directory containing subject folders
            min_seizures: Minimum number of seizures required per subject
        """
        self.data_path = Path(data_path)
        self.min_seizures = min_seizures
        self.subjects = {}

    def scan_directory_structure(self, pattern="**/*.edf"):
        """
        Scan directory for EDF files organized by subject

        Expected structure:
        data_path/
            subject_001/
                file1.edf
                file2.edf
                annotations.txt (optional)
            subject_002/
                ...
        """
        logger.info(f"Scanning {self.data_path} for EDF files...")

        edf_files = list(self.data_path.glob(pattern))
        logger.info(f"Found {len(edf_files)} EDF files")

        # Group by subject (assuming one level up is subject folder)
        subjects = {}
        for edf_file in edf_files:
            subject_id = edf_file.parent.name
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(edf_file)

        logger.info(f"Found {len(subjects)} subjects")

        return subjects

    def load_annotations(self, annotation_file: Path) -> Dict:
        """
        Load seizure annotations from file

        Expected format (JSON):
        {
            "file1.edf": [
                {"start": 100, "end": 150},
                {"start": 500, "end": 550}
            ],
            "file2.edf": [
                {"start": 200, "end": 250}
            ]
        }

        Or CSV format:
        file,start_sec,end_sec
        file1.edf,100,150
        file1.edf,500,550
        file2.edf,200,250
        """
        if annotation_file.suffix == '.json':
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            # Convert to standard format
            result = {}
            for filename, seizures in annotations.items():
                result[filename] = [(s['start'], s['end']) for s in seizures]

        elif annotation_file.suffix == '.csv':
            df = pd.read_csv(annotation_file)
            result = {}
            for filename in df['file'].unique():
                file_seizures = df[df['file'] == filename]
                result[filename] = list(zip(file_seizures['start_sec'], file_seizures['end_sec']))

        else:
            raise ValueError(f"Unsupported annotation format: {annotation_file.suffix}")

        return result

    def count_seizures_from_annotations(self, subject_path: Path) -> Tuple[int, Dict]:
        """
        Count seizures for a subject from annotation file

        Returns:
            (num_seizures, annotations_dict)
        """
        # Look for annotation file
        annotation_files = list(subject_path.glob("*.json")) + list(subject_path.glob("*.csv"))

        if not annotation_files:
            logger.warning(f"No annotation file found for {subject_path.name}")
            return 0, {}

        annotation_file = annotation_files[0]
        annotations = self.load_annotations(annotation_file)

        total_seizures = sum(len(seizures) for seizures in annotations.values())

        return total_seizures, annotations

    def filter_subjects_by_seizure_count(self, subjects_dict=None):
        """
        Filter subjects to only those with minimum number of seizures

        Args:
            subjects_dict: Dict of {subject_id: [file_paths]}

        Returns:
            filtered_subjects: Dict of {subject_id: {'files': [...], 'annotations': {...}, 'n_seizures': N}}
        """
        if subjects_dict is None:
            subjects_dict = self.scan_directory_structure()

        filtered = {}

        for subject_id, file_list in subjects_dict.items():
            subject_path = file_list[0].parent

            # Count seizures
            n_seizures, annotations = self.count_seizures_from_annotations(subject_path)

            if n_seizures >= self.min_seizures:
                filtered[subject_id] = {
                    'files': [str(f) for f in file_list],
                    'annotations': annotations,
                    'n_seizures': n_seizures,
                    'path': str(subject_path)
                }
                logger.info(f"✓ {subject_id}: {n_seizures} seizures, {len(file_list)} files")
            else:
                logger.info(f"✗ {subject_id}: Only {n_seizures} seizures (need {self.min_seizures})")

        logger.info(f"\nSelected {len(filtered)} subjects with ≥{self.min_seizures} seizures")

        return filtered

    def save_selected_subjects(self, filtered_subjects, output_file="selected_subjects.json"):
        """Save selected subjects to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(filtered_subjects, f, indent=2)

        logger.info(f"Saved selected subjects to {output_file}")

    def create_summary_report(self, filtered_subjects):
        """Create a summary report of selected subjects"""
        summary = []

        for subject_id, info in filtered_subjects.items():
            summary.append({
                'subject_id': subject_id,
                'n_seizures': info['n_seizures'],
                'n_files': len(info['files']),
            })

        df = pd.DataFrame(summary)
        df = df.sort_values('n_seizures', ascending=False)

        logger.info("\nSummary Report:")
        logger.info(f"\n{df.to_string()}")
        logger.info(f"\nTotal subjects: {len(df)}")
        logger.info(f"Total seizures: {df['n_seizures'].sum()}")
        logger.info(f"Mean seizures per subject: {df['n_seizures'].mean():.1f}")
        logger.info(f"Median seizures per subject: {df['n_seizures'].median():.1f}")

        return df


def main():
    """Example usage"""
    # Path to your EEG data
    data_path = "/path/to/your/eeg/data"

    # Minimum number of seizures required
    min_seizures = 2

    selector = SubjectSelector(data_path, min_seizures)

    # Scan and filter subjects
    subjects = selector.scan_directory_structure()
    filtered_subjects = selector.filter_subjects_by_seizure_count(subjects)

    # Create summary report
    summary_df = selector.create_summary_report(filtered_subjects)

    # Save results
    selector.save_selected_subjects(filtered_subjects)


if __name__ == "__main__":
    main()

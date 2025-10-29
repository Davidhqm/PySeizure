"""
Complete pipeline example: From data selection to training and prediction

This script demonstrates the full workflow:
1. Select subjects with sufficient seizures
2. Train subject-dependent models
3. Evaluate and save results
"""

import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd

from subject_selector import SubjectSelector
from train import SeizurePredictor
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_select_subjects(data_path, min_seizures=2, output_file="selected_subjects.json"):
    """Step 1: Scan your data and select subjects with enough seizures"""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Selecting subjects with sufficient seizures")
    logger.info("="*70 + "\n")

    selector = SubjectSelector(data_path, min_seizures)

    # Scan directory
    subjects = selector.scan_directory_structure()

    # Filter by seizure count
    filtered_subjects = selector.filter_subjects_by_seizure_count(subjects)

    # Create summary
    summary_df = selector.create_summary_report(filtered_subjects)

    # Save
    selector.save_selected_subjects(filtered_subjects, output_file)

    return filtered_subjects, summary_df


def step2_train_models(selected_subjects, max_subjects=None, save_dir="trained_models"):
    """Step 2: Train subject-dependent models"""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Training subject-dependent models")
    logger.info("="*70 + "\n")

    predictor = SeizurePredictor()

    results = []

    # Get subject list
    subject_ids = list(selected_subjects.keys())
    if max_subjects:
        subject_ids = subject_ids[:max_subjects]

    logger.info(f"Training models for {len(subject_ids)} subjects\n")

    for i, subject_id in enumerate(subject_ids, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Subject {i}/{len(subject_ids)}: {subject_id}")
        logger.info(f"{'='*70}\n")

        subject_info = selected_subjects[subject_id]

        try:
            # Prepare data
            X_train, X_val, y_train, y_val = predictor.prepare_subject_data(
                subject_info['files'],
                subject_info['annotations']
            )

            if X_train is None:
                logger.warning(f"Skipping {subject_id}: No valid data")
                continue

            # Train model
            model, history = predictor.train_subject_model(
                subject_id, X_train, X_val, y_train, y_val,
                save_dir=save_dir
            )

            # Evaluate
            metrics = predictor.evaluate_model(model, X_val, y_val)

            # Store results
            results.append({
                'subject_id': subject_id,
                'n_seizures': subject_info['n_seizures'],
                'n_train': len(X_train),
                'n_val': len(X_val),
                **metrics
            })

            logger.info(f"\n✓ Successfully trained model for {subject_id}")

        except Exception as e:
            logger.error(f"Error training {subject_id}: {e}")
            continue

    return results


def step3_evaluate_results(results, save_dir="trained_models"):
    """Step 3: Evaluate and summarize results across all subjects"""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Evaluating results across all subjects")
    logger.info("="*70 + "\n")

    if not results:
        logger.error("No results to evaluate!")
        return

    # Create results DataFrame
    df = pd.DataFrame(results)

    logger.info("\nResults Summary:")
    logger.info(f"\n{df.to_string()}\n")

    # Calculate average metrics
    metric_cols = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    avg_metrics = {}

    logger.info("\nAverage Performance Across All Subjects:")
    logger.info("-" * 50)
    for col in metric_cols:
        if col in df.columns:
            avg_metrics[col] = df[col].mean()
            logger.info(f"{col:12s}: {avg_metrics[col]:.4f} ± {df[col].std():.4f}")

    # Calculate sensitivity and specificity
    if 'recall' in df.columns:
        logger.info(f"{'sensitivity':12s}: {df['recall'].mean():.4f} (same as recall)")

    if 'precision' in df.columns and 'recall' in df.columns:
        # Estimate specificity (this is approximate)
        logger.info("\n(Note: For exact specificity, compute from confusion matrix)")

    # Save results
    output_path = Path(save_dir) / "all_results.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    # Save summary
    summary_path = Path(save_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'n_subjects': len(df),
            'avg_metrics': avg_metrics,
            'config': {
                'data': DATA_CONFIG,
                'model': MODEL_CONFIG,
                'train': TRAIN_CONFIG
            }
        }, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")

    return df, avg_metrics


def main():
    """
    Main pipeline - customize these paths for your data
    """
    # =================================================================
    # CONFIGURATION - UPDATE THESE FOR YOUR DATASET
    # =================================================================

    # Path to your EEG data directory
    DATA_PATH = "/path/to/your/eeg/data"

    # Minimum number of seizures per subject
    MIN_SEIZURES = 2

    # Maximum number of subjects to train (None = all)
    # Start with a small number for testing, then increase
    MAX_SUBJECTS = 5  # Set to None to train all

    # Output directories
    OUTPUT_DIR = "results"
    MODEL_DIR = "trained_models"

    # =================================================================
    # RUN PIPELINE
    # =================================================================

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(MODEL_DIR).mkdir(exist_ok=True)

    # Step 1: Select subjects
    selected_subjects, summary_df = step1_select_subjects(
        DATA_PATH,
        MIN_SEIZURES,
        output_file=f"{OUTPUT_DIR}/selected_subjects.json"
    )

    if not selected_subjects:
        logger.error("No subjects found! Check your data path and annotations.")
        return

    # Step 2: Train models
    results = step2_train_models(
        selected_subjects,
        max_subjects=MAX_SUBJECTS,
        save_dir=MODEL_DIR
    )

    if not results:
        logger.error("No models trained successfully!")
        return

    # Step 3: Evaluate results
    results_df, avg_metrics = step3_evaluate_results(
        results,
        save_dir=OUTPUT_DIR
    )

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nTrained {len(results)} subject-dependent models")
    logger.info(f"Models saved in: {MODEL_DIR}/")
    logger.info(f"Results saved in: {OUTPUT_DIR}/")
    logger.info("\nYou can now use these models for seizure prediction!")


if __name__ == "__main__":
    main()

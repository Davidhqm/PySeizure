# Simple Seizure Predictor

A streamlined, fast-to-implement seizure prediction system using deep learning for **subject-dependent** (patient-specific) models.

Based on the CNN-Transformer architecture from **Koutsouvelis et al. 2024** - achieves state-of-the-art performance with minimal preprocessing.

## 🚀 Quick Start

### 1. Installation

```bash
pip install tensorflow numpy pandas mne scikit-learn
```

### 2. Prepare Your Data

Organize your EEG data like this:

```
your_data/
├── subject_001/
│   ├── recording1.edf
│   ├── recording2.edf
│   └── annotations.json  # Seizure annotations
├── subject_002/
│   ├── recording1.edf
│   └── annotations.json
└── ...
```

**Annotation format** (`annotations.json`):
```json
{
  "recording1.edf": [
    {"start": 100, "end": 150},
    {"start": 500, "end": 550}
  ],
  "recording2.edf": [
    {"start": 200, "end": 250}
  ]
}
```

Or CSV format (`annotations.csv`):
```csv
file,start_sec,end_sec
recording1.edf,100,150
recording1.edf,500,550
recording2.edf,200,250
```

### 3. Run the Complete Pipeline

```python
python example_complete_pipeline.py
```

**That's it!** The pipeline will:
1. ✅ Select subjects with sufficient seizures (≥2 by default)
2. ✅ Train subject-dependent models
3. ✅ Evaluate and save results

## 📊 What You Get

- **Subject-dependent models**: One model per patient (much more accurate!)
- **Minimal preprocessing**: Just filtering and epoching - no complex feature engineering
- **State-of-the-art architecture**: CNN-Transformer model
- **Fast training**: ~5-10 minutes per subject on GPU

## 📁 Project Structure

```
simple_seizure_predictor/
├── config.py                      # Configuration (customize here!)
├── data_loader.py                 # EDF file reading & preprocessing
├── model.py                       # CNN-Transformer architecture
├── train.py                       # Training logic
├── subject_selector.py            # Select subjects with enough data
├── example_complete_pipeline.py   # Full end-to-end example
└── README.md                      # You are here
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
DATA_CONFIG = {
    'target_freq': 256,           # Sampling rate
    'lowcut': 0.5,                # High-pass filter
    'highcut': 45,                # Low-pass filter
    'epoch_length': 5,            # Window size in seconds
    'preictal_minutes': 60,       # How long before seizure = preictal
    'min_seizures': 2,            # Min seizures to include subject
}

MODEL_CONFIG = {
    'cnn_filters': [32, 64, 128],
    'num_heads': 8,               # Transformer attention heads
    'num_transformer_blocks': 2,
    # ... more options
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 20,
    'learning_rate': 0.001,
}
```

## 🔧 Usage Examples

### Example 1: Select Subjects from Your 125 Subjects

```python
from subject_selector import SubjectSelector

# Point to your data directory
selector = SubjectSelector("/path/to/your/data", min_seizures=2)

# Scan and filter
subjects = selector.scan_directory_structure()
filtered = selector.filter_subjects_by_seizure_count(subjects)

# Show summary
summary_df = selector.create_summary_report(filtered)

# Save selected subjects
selector.save_selected_subjects(filtered, "my_selected_subjects.json")

print(f"Selected {len(filtered)} subjects from 125!")
```

### Example 2: Train a Model for One Subject

```python
from train import SeizurePredictor

predictor = SeizurePredictor()

# Your subject's EDF files
file_list = [
    "/data/subject_001/recording1.edf",
    "/data/subject_001/recording2.edf",
]

# Seizure annotations
seizure_annotations = {
    "/data/subject_001/recording1.edf": [(100, 150), (500, 550)],
    "/data/subject_001/recording2.edf": [(200, 250)],
}

# Prepare data
X_train, X_val, y_train, y_val = predictor.prepare_subject_data(
    file_list, seizure_annotations
)

# Train
model, history = predictor.train_subject_model(
    "subject_001", X_train, X_val, y_train, y_val
)

# Evaluate
metrics = predictor.evaluate_model(model, X_val, y_val)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"AUC: {metrics['auc']:.4f}")
```

### Example 3: Make Predictions on New Data

```python
import tensorflow as tf
import numpy as np
from data_loader import EEGDataLoader

# Load trained model
model = tf.keras.models.load_model("trained_models/subject_001/best_model.h5")

# Load new EEG data
data_loader = EEGDataLoader(target_freq=256, lowcut=0.5, highcut=45, epoch_length=5)
raw = data_loader.load_edf("new_recording.edf")
data = data_loader.preprocess(raw)
epochs = data_loader.create_epochs(data)

# Predict
predictions = model.predict(epochs)

# Interpret: probabilities of seizure
print(f"Seizure probabilities: {predictions[:, 0]}")

# Flag high-risk epochs (e.g., > 0.5)
high_risk = np.where(predictions[:, 0] > 0.5)[0]
print(f"High-risk epochs: {high_risk}")
```

## 📈 Expected Performance

Based on the Koutsouvelis 2024 paper (CHB-MIT dataset):

| Metric        | Average |
|---------------|---------|
| Sensitivity   | ~99%    |
| Specificity   | ~95%    |
| Accuracy      | ~97%    |
| AUC           | ~99%    |
| Prediction Time | ~77 min before seizure |

**Note**: Your results will vary based on:
- Data quality
- Number of seizures per subject
- Preprocessing choices
- Model configuration

## 🎯 Why Subject-Dependent?

**Subject-dependent** (patient-specific) models are:
- ✅ **Easier to train**: Less data needed per patient
- ✅ **More accurate**: Personalized to each patient's unique patterns
- ✅ **Faster**: Simpler than cross-patient generalization
- ✅ **Clinically practical**: Real-world systems are personalized anyway

## 🔬 The Architecture

```
Input EEG (5s window, 23 channels)
         ↓
   [CNN Block 1] → 32 filters
         ↓
   [CNN Block 2] → 64 filters
         ↓
   [CNN Block 3] → 128 filters
         ↓
    [Reshape]
         ↓
  [Transformer] → Multi-head attention
         ↓
  [Transformer] → Multi-head attention
         ↓
   [Classifier] → Sigmoid output
         ↓
  Seizure probability (0-1)
```

## 💡 Tips for Your 125 Subjects

1. **Start small**: Train on 5-10 subjects first to test
2. **Filter by seizure count**: Use subjects with ≥3 seizures for better results
3. **Check data quality**: Remove noisy recordings
4. **Balance your dataset**: Aim for similar amounts of preictal/interictal data
5. **Use leave-one-seizure-out**: Test on held-out seizures, not just random epochs

## 🐛 Troubleshooting

### "No subjects found"
- Check your data path
- Verify folder structure matches expected format
- Make sure annotation files exist

### "Model training fails"
- Check for NaN values in data
- Reduce batch size if running out of memory
- Try with fewer subjects first

### "Low accuracy"
- Check seizure annotations are correct
- Try different preictal window (30, 45, 60 min)
- Ensure enough data per subject
- Check for data quality issues

## 📚 References

This implementation is based on:

**Koutsouvelis et al. 2024**
*"Preictal period optimization for deep learning-based epileptic seizure prediction"*
Journal of Neural Engineering, 21(6)

Key concepts:
- Minimal preprocessing (0.5-45 Hz bandpass, common average reference)
- CNN-Transformer hybrid architecture
- Subject-specific models
- 5-second epochs, 60-minute preictal window

## 🤝 Contributing

Want to improve this? Some ideas:
- [ ] Add real-time prediction mode
- [ ] Implement data augmentation (time-reversal, sign-flipping)
- [ ] Add post-processing for alarm generation
- [ ] Support for other EEG formats (BrainVision, EGI, etc.)
- [ ] Cross-validation with leave-one-seizure-out

## 📝 License

This code is provided as-is for research and educational purposes.

## ✉️ Questions?

This is a simplified implementation for quick prototyping. For production use, consider:
- More robust error handling
- Extensive validation
- Clinical-grade annotation tools
- Real-time processing optimizations

---

**Happy predicting! 🧠⚡**

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:54:52 2026

@author: Raji
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import mne

from scipy.signal import welch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(r"E:\EEG\STUD ENGAGE\All_files_4sec_epoched_organized_used_in_pgm")  # CHANGE THIS PATH

OUTPUT_DIR = ROOT_DIR / "LOSO_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Labels: Engage = 1, Disengage = 0
LABELS = {
    "engage": 1,
    "disengage": 0
}

# Frequency bands in Hz
FREQUENCY_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40)
}

# ============================================================
# 2. DISCOVER FILES AND IDENTIFY SUBJECTS
# ============================================================

def discover_files(root_dir):
    """
    Recursively find EEGLAB .set files.

    Expected structure:
        root/SUB_01/engage/*.set
        root/SUB_01/disengage/*.set

    The subject ID can also be present in the filename.
    """

    records = []

    for file_path in root_dir.rglob("*.set"):

        # Determine label from a parent folder
        label_name = None

        for parent in file_path.parents:
            folder_name = parent.name.lower()

            if folder_name in LABELS:
                label_name = folder_name
                break

            if parent == root_dir:
                break

        if label_name is None:
            continue

        # Identify subject from the path or filename
        match = re.search(
            r"SUB[_-]?(\d+)",
            str(file_path),
            flags=re.IGNORECASE
        )

        if match is None:
            print(f"WARNING: Subject ID not found: {file_path}")
            continue

        subject_id = f"SUB_{int(match.group(1)):02d}"

        records.append({
            "subject": subject_id,
            "label_name": label_name,
            "label": LABELS[label_name],
            "file_path": file_path
        })

    return records


# ============================================================
# 3. EXTRACT PSD FEATURES FROM EACH EPOCH
# ============================================================

def extract_psd_features(epochs, expected_channels=None):
    """
    Extract mean band-power features from each 4-second epoch.

    Output:
        X: (number of epochs, number of features)
        channel_names: ordered EEG channel names

    Features:
        5 frequency bands x EEG channels

    Band power is calculated by integrating Welch PSD
    within each frequency band.
    """

    # Keep EEG channels only
    eeg_epochs = epochs.copy().pick("eeg")

    channel_names = eeg_epochs.ch_names

    if len(channel_names) == 0:
        raise ValueError("No EEG channels found.")

    # Standardize channel order across all files
    if expected_channels is not None:

        if set(channel_names) != set(expected_channels):
            raise ValueError(
                "EEG channel names differ from the reference file."
            )

        eeg_epochs.reorder_channels(expected_channels)
        channel_names = eeg_epochs.ch_names

    data = eeg_epochs.get_data()
    # Shape: (epochs, channels, time points)

    sfreq = float(eeg_epochs.info["sfreq"])

    if data.shape[-1] < 2:
        raise ValueError("Epoch contains too few time samples.")

    # Welch PSD
    nperseg = min(
        int(2 * sfreq),
        data.shape[-1]
    )

    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        axis=-1
    )

    # psd shape: (epochs, channels, frequencies)
    feature_blocks = []

    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():

        mask = (
            (freqs >= low_freq) &
            (freqs < high_freq)
        )

        if not np.any(mask):
            raise ValueError(
                f"No PSD frequency bins for {band_name} band."
            )

        # Integrate PSD over the frequency band
       
        band_power = np.trapz(
           psd[..., mask],
           x=freqs[mask],
           axis=-1
           )

        # Shape: (epochs, channels)
        feature_blocks.append(band_power)

    # Shape: (epochs, bands * channels)
    X = np.concatenate(feature_blocks, axis=1)

    # Log-transform power to reduce skewness
    X = np.log10(np.maximum(X, 1e-20))

    return X, channel_names


# ============================================================
# 4. LOAD ALL EPOCHS AND LABELS
# ============================================================

def load_dataset(records):

    X_all = []
    y_all = []
    subject_all = []
    file_all = []

    reference_channels = None
    feature_names = None

    for record in records:

        file_path = record["file_path"]
        subject_id = record["subject"]
        label = record["label"]

        print(f"Loading: {subject_id} | "
              f"{record['label_name']} | {file_path.name}")

        try:
            # Read epoched EEGLAB .set files
            epochs = mne.io.read_epochs_eeglab(
                str(file_path),
                verbose="ERROR"
            )

            X, channel_names = extract_psd_features(
                epochs,
                expected_channels=reference_channels
            )

            if reference_channels is None:
                reference_channels = channel_names

                feature_names = [
                    f"{band}_{channel}"
                    for band in FREQUENCY_BANDS
                    for channel in reference_channels
                ]

            # One label and subject ID per epoch
            n_epochs = X.shape[0]

            X_all.append(X)
            y_all.extend([label] * n_epochs)
            subject_all.extend([subject_id] * n_epochs)
            file_all.extend([str(file_path)] * n_epochs)

            print(f"  Extracted {n_epochs} epochs; "
                  f"{X.shape[1]} features per epoch.")

        except Exception as error:
            print(f"  SKIPPED: {file_path}")
            print(f"  Reason: {error}")

    if not X_all:
        raise RuntimeError(
            "No usable .set files were loaded. "
            "Check ROOT_DIR, folder names and file format."
        )

    X_all = np.vstack(X_all)
    y_all = np.asarray(y_all, dtype=int)
    subject_all = np.asarray(subject_all)
    file_all = np.asarray(file_all)

    metadata = pd.DataFrame({
        "subject": subject_all,
        "label": y_all,
        "source_file": file_all
    })

    return X_all, y_all, subject_all, metadata, feature_names


# ============================================================
# 5. DEFINE CLASSIFIERS
# ============================================================

def get_classifiers():

    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }


# ============================================================
# 6. LOSO CROSS-VALIDATION
# ============================================================

def run_loso(X, y, subjects, metadata):

    unique_subjects = np.unique(subjects)

    if len(unique_subjects) < 2:
        raise ValueError(
            "LOSO requires at least two subjects."
        )

    classifiers = get_classifiers()

    subject_results = []
    prediction_results = []
    summary_results = []

    for model_name, model_template in classifiers.items():

        print("\n" + "=" * 65)
        print(f"MODEL: {model_name}")
        print("=" * 65)

        all_true = []
        all_pred = []

        for test_subject in unique_subjects:

            train_mask = subjects != test_subject
            test_mask = subjects == test_subject

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Both classes must exist in training data
            if len(np.unique(y_train)) < 2:
                print(
                    f"Skipping {test_subject}: "
                    "training data contains only one class."
                )
                continue

            if len(np.unique(y_test)) < 2:
                print(
                    f"WARNING: {test_subject} test data "
                    "contains only one class; interpret F1 carefully."
                )

            # Clone a fresh model for each LOSO fold
            from sklearn.base import clone
            model = clone(model_template)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            f1 = f1_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0
            )

            precision = precision_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0
            )

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            subject_results.append({
                "model": model_name,
                "held_out_subject": test_subject,
                "n_test_epochs": len(y_test),
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1]
            })

            all_true.extend(y_test)
            all_pred.extend(y_pred)

            # Store every out-of-fold prediction
            test_indices = np.where(test_mask)[0]

            for local_i, global_i in enumerate(test_indices):

                prediction_results.append({
                    "model": model_name,
                    "subject": subjects[global_i],
                    "source_file": metadata.iloc[
                        global_i
                    ]["source_file"],
                    "true_label": int(y[global_i]),
                    "predicted_label": int(y_pred[local_i])
                })

            print(
                f"{test_subject}: "
                f"Accuracy={accuracy:.4f}, "
                f"F1={f1:.4f}, "
                f"Test epochs={len(y_test)}"
            )

        # Pooled out-of-fold performance
        pooled_accuracy = accuracy_score(all_true, all_pred)

        pooled_f1 = f1_score(
            all_true,
            all_pred,
            pos_label=1,
            zero_division=0
        )

        # Mean and SD of per-subject scores
        model_subject_rows = [
            row for row in subject_results
            if row["model"] == model_name
        ]

        subject_accuracies = [
            row["accuracy"] for row in model_subject_rows
        ]

        subject_f1s = [
            row["f1_score"] for row in model_subject_rows
        ]

        summary_results.append({
            "model": model_name,
            "n_subjects_evaluated": len(model_subject_rows),
            "pooled_accuracy": pooled_accuracy,
            "pooled_f1_score": pooled_f1,
            "mean_subject_accuracy": np.mean(subject_accuracies),
            "sd_subject_accuracy": np.std(
                subject_accuracies, ddof=1
            ) if len(subject_accuracies) > 1 else 0.0,
            "mean_subject_f1": np.mean(subject_f1s),
            "sd_subject_f1": np.std(
                subject_f1s, ddof=1
            ) if len(subject_f1s) > 1 else 0.0
        })

    return (
        pd.DataFrame(subject_results),
        pd.DataFrame(prediction_results),
        pd.DataFrame(summary_results)
    )


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("\nSearching for EEGLAB files...")

    records = discover_files(ROOT_DIR)

    if not records:
        raise RuntimeError(
            "No .set files found in engage/disengage folders. "
            "Check ROOT_DIR and folder names."
        )

    print(f"Found {len(records)} .set files.")

    file_summary = pd.DataFrame(records)
    file_summary["file_path"] = file_summary[
        "file_path"
    ].astype(str)

    file_summary.to_csv(
        OUTPUT_DIR / "discovered_files.csv",
        index=False
    )

    print("\nExtracting EEG features...")

    X, y, subjects, metadata, feature_names = load_dataset(
        records
    )

    print("\nDataset information")
    print("-------------------")
    print(f"Total EEG epochs: {len(y)}")
    print(f"Number of subjects: {len(np.unique(subjects))}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Engage epochs: {np.sum(y == 1)}")
    print(f"Disengage epochs: {np.sum(y == 0)}")

    subject_counts = pd.crosstab(
        metadata["subject"],
        metadata["label"]
    ).rename(columns={
        0: "disengage_epochs",
        1: "engage_epochs"
    })

    subject_counts.to_csv(
        OUTPUT_DIR / "subject_epoch_counts.csv"
    )

    pd.DataFrame({
        "feature": feature_names
    }).to_csv(
        OUTPUT_DIR / "feature_names.csv",
        index=False
    )

    metadata.to_csv(
        OUTPUT_DIR / "epoch_metadata.csv",
        index=False
    )

    print("\nRunning LOSO cross-validation...")

    subject_results, predictions, summary = run_loso(
        X, y, subjects, metadata
    )

    subject_results.to_csv(
        OUTPUT_DIR / "loso_subject_results.csv",
        index=False
    )

    predictions.to_csv(
        OUTPUT_DIR / "loso_predictions.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "loso_model_summary.csv",
        index=False
    )

    print("\nOVERALL LOSO RESULTS")
    print("====================")
    print(summary.round(4).to_string(index=False))

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
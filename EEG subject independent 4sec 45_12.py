# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:31:16 2026

@author: Raji
"""

import os
import numpy as np
import mne
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ============================================================
# CONFIGURATION
# ============================================================

# Folder structure example:
# dataset/
#    Subject_01/
#         engage/
#              file1.set
#              file2.set
#         disengage/
#              file1.set
#              file2.set
#
# Each subject contains separate engage/disengage folders.

DATASET_PATH = r"E:\EEG\STUD ENGAGE\All_files_16sec_epoched_45_12_model_sivapriya"

SAMPLING_RATE = 250
WINDOW_SIZE =16        # seconds (original EEG epoch length)
WINDOW_OVERLAP = 0     # IMPORTANT: use 0 to avoid leakage concerns

N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================
# FEATURE EXTRACTION
# ============================================================


def extract_features(epoch_data, sfreq):
    """
    Extract PSD-based EEG features from one epoch.

    Parameters
    ----------
    epoch_data : ndarray
        Shape -> (channels, samples)

    sfreq : float
        Sampling frequency

    Returns
    -------
    features : ndarray
    """

    # EEG frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }

    features = []

    for ch in epoch_data:

        # Compute PSD using Welch method
        psd, freqs = mne.time_frequency.psd_array_welch(
            ch,
            sfreq=sfreq,
            fmin=1,
            fmax=40,
            verbose=False
        )

        # Band power extraction
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.mean(psd[idx])
            features.append(band_power)

    return np.array(features)


# ============================================================
# EEG EPOCH HANDLING
# ============================================================
# Since each .set file already contains pre-epoched
# 4-second EEG windows, additional segmentation is NOT applied.
# This avoids overlapping-window leakage concerns.
# ============================================================


def get_epochs(data):
    """
    Return epochs directly.

    Expected shape:
    ----------------
    (epochs, channels, samples)

    If data is continuous:
    ----------------------
    Converts to single epoch.
    """

    if len(data.shape) == 3:
        return data

    elif len(data.shape) == 2:
        return np.expand_dims(data, axis=0)

    else:
        raise ValueError("Unexpected EEG data shape")


# ============================================================
# LOAD DATA
# ============================================================



def load_dataset(dataset_path):
    """
    Load EEG epochs and count total Engage and Disengage epochs.
    """

    X = []
    y = []
    groups = []

    total_engage_epochs = 0
    total_disengage_epochs = 0

    subjects = sorted(os.listdir(dataset_path))

    for subject_idx, subject in enumerate(subjects):

        subject_path = os.path.join(dataset_path, subject)

        if not os.path.isdir(subject_path):
            continue

        print(f"\nProcessing {subject}...")

        class_mapping = {
            'Engage': 1,
            'Disengage': 0
        }

        for class_name, label in class_mapping.items():

            class_path = os.path.join(subject_path, class_name)

            if not os.path.isdir(class_path):
                continue

            files = sorted(
                f for f in os.listdir(class_path)
                if f.lower().endswith('.set')
            )

            for file in files:

                file_path = os.path.join(class_path, file)

                try:
                    # Load already-epoched EEGLAB dataset
                    epochs = mne.read_epochs_eeglab(
                        file_path,
                        verbose=False
                    )

                    # Count epochs in this file
                    n_epochs = len(epochs)

                    if label == 1:
                        total_engage_epochs += n_epochs
                    else:
                        total_disengage_epochs += n_epochs

                    print(
                        f"  {class_name}: {file} "
                        f"-> {n_epochs} epochs"
                    )

                    # Extract features from every epoch
                    for epoch in epochs:
                        feat = extract_features(
                            epoch,
                            epochs.info['sfreq']
                        )

                        X.append(feat)
                        y.append(label)
                        groups.append(subject_idx)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Print total epoch counts
    print("\n======================================")
    print("       TOTAL EEG EPOCH COUNTS")
    print("======================================")
    print(f"Total Engage epochs    : {total_engage_epochs}")
    print(f"Total Disengage epochs : {total_disengage_epochs}")
    print(
        f"Grand total epochs    : "
        f"{total_engage_epochs + total_disengage_epochs}"
    )
    print("======================================")

    return np.array(X), np.array(y), np.array(groups)



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading dataset...")

    X, y, groups = load_dataset(DATASET_PATH)

    print("\nDataset Shape")
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)

    # ========================================================
    # LEAKAGE-FREE 5-FOLD CROSS VALIDATION
    # ========================================================
    # GroupKFold ensures:
    # - same subject NEVER appears in both train and test
    # - prevents subject leakage
    # ========================================================

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_accuraciesrf = []
    

    all_truerf = []
    all_predrf = []
    
    fold_accuraciesrfbalance = []
    fold_balance_f1=[]
    all_truerfbalance = []
    all_predrfbalance = []

    fold_number = 1

    for train_idx, test_idx in gkf.split(X, y, groups):

        print(f"\n========== Fold {fold_number} ==========")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])

        print("Train Subjects:", train_subjects)
        print("Test Subjects:", test_subjects)

        # ====================================================
        # PIPELINE PREVENTS SCALING LEAKAGE
        # ====================================================
        # StandardScaler is fit ONLY on training data
        # ====================================================

        # clfrf = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        #       ])
        # clfrfbalance = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('rf', RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42))
        #       ])
        
#         clfrf =  Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm', SVC(
#         kernel='rbf',
#         C=1.0,
#         gamma='scale',
#         random_state=42
#     ))
# ])
#         clfrfbalance =  Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm', SVC(
#         kernel='rbf',
#         C=1.0,
#         gamma='scale',
#         class_weight='balanced',
#         random_state=42
#     ))
# ])

# clf = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm', SVC(kernel='rbf'))
#         ])
        clfrf = Pipeline([
        ('scaler', StandardScaler()),
          ('lr', LogisticRegression())
          ])
        
        clfrfbalance = Pipeline([
              ('scaler', StandardScaler()),
              ('lr', LogisticRegression(
                  C=1.0,
                  class_weight='balanced',
                  max_iter=2000,
                  random_state=42
                  ))
              ])
        
         
        
        # clf = Pipeline([
        #      ('scaler', StandardScaler()),
        #      ('lr', LogisticRegression())
        #        ])
        clfrf.fit(X_train, y_train)

        y_predrf = clfrf.predict(X_test)

        accrf = accuracy_score(y_test, y_predrf)
        
        
        clfrfbalance.fit(X_train, y_train)

        y_predrfbalance = clfrfbalance.predict(X_test)

        accrfbalance = accuracy_score(y_test, y_predrfbalance)
        balancef1= f1_score(
    y_test,
    y_predrfbalance,
    average='weighted',
    zero_division=0
)

        print(f"Fold Accuracy: {accrf:.4f}")

        fold_accuraciesrf.append(accrf)

        all_truerf.extend(y_test)
        all_predrf.extend(y_predrf)
        
        fold_accuraciesrfbalance.append(accrfbalance)
        fold_balance_f1.append(balancef1)

        all_truerfbalance.extend(y_test)
        all_predrfbalance.extend(y_predrfbalance)

        fold_number += 1

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")

    print(f"Mean Accuracy: {np.mean(fold_accuraciesrf):.4f}")
    print(f"Std Accuracy : {np.std(fold_accuraciesrf):.4f}")

    print("\nClassification Report")
    print(classification_report(all_truerf, all_predrf))

    print("\nConfusion Matrix")
    print(confusion_matrix(all_truerf, all_predrf))

    # ========================================================
    # IMPORTANT NOTES FOR PAPER
    # ========================================================
    # 1. GroupKFold prevents subject leakage.
    # 2. Same subject never appears in train and test.
    # 3. StandardScaler fitted only on training data.
    # 4. No additional overlapping segmentation applied.
    # 5. Original 4-second epochs used directly.
    # ========================================================
    
    print("\n====================================")
    print("FINAL RESULTS BALANCED")
    print("====================================")

    print(f"Mean Accuracy: {np.mean(fold_accuraciesrfbalance):.4f}")
    print(f"Std Accuracy : {np.std(fold_accuraciesrfbalance):.4f}")


    print(f"Mean F1-score : {np.mean(fold_balance_f1):.4f}")
    print(f"SD F1-score   : {np.std(fold_balance_f1):.4f}")
    print("\nClassification Report")
    print(classification_report(all_truerfbalance, all_predrfbalance))

    print("\nConfusion Matrix")
    print(confusion_matrix(all_truerfbalance, all_predrfbalance))
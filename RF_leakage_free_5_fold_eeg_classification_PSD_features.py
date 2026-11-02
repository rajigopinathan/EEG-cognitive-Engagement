import os
import numpy as np
import mne
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score,classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import skew, kurtosis

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

DATASET_PATH = r"E:\EEG\STUD ENGAGE\All_files_16sec_epoched_organized_used_in_pgm"

SAMPLING_RATE = 250
WINDOW_SIZE = 16        # seconds (original EEG epoch length)
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
    IMPORTANT:
    Subject IDs are preserved using GROUP labels.
    This prevents data leakage across folds.
    """

    X = []
    y = []
    groups = []

    subjects = sorted(os.listdir(dataset_path))

    for subject_idx, subject in enumerate(subjects):

        subject_path = os.path.join(dataset_path, subject)

        if not os.path.isdir(subject_path):
            continue

        print(f"Processing {subject}...")

        class_mapping = {
            'Engage': 1,
            'Disengage': 0
        }

        for class_name, label in class_mapping.items():

            class_path = os.path.join(subject_path, class_name)

            if not os.path.exists(class_path):
                continue

            files = [f for f in os.listdir(class_path) if f.endswith('.set')]

            for file in files:

                file_path = os.path.join(class_path, file)

                try:
                    # Load EEGLAB .set file
                    #raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

                    # Basic preprocessing
                    #raw.filter(1., 40., verbose=False)

                    #data = raw.get_data()
                    sfreq = 250

                    # Since files are already epoched,
                    # no further segmentation is performed.
                    #epochs = get_epochs(data)
                    epochs = mne.read_epochs_eeglab(file_path, verbose=False)
                    for epoch in epochs:
                        feat = extract_features(epoch, sfreq)

                        X.append(feat)
                        y.append(label)
                        groups.append(subject_idx)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

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

    fold_accuracies = []
    fold_F1=[]
    all_true = []
    all_pred = []

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

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
              ])
        

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1=f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

        print(f"Fold Accuracy: {acc:.4f}")

        fold_accuracies.append(acc)
        fold_F1.append(f1)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

        fold_number += 1

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")

    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Accuracy : {np.std(fold_accuracies):.4f}")
    
    print(f"Mean F1-score : {np.mean(fold_F1):.4f}")
    print(f"SD F1-score   : {np.std(fold_F1):.4f}")

    print("\nClassification Report")
    print(classification_report(all_true, all_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(all_true, all_pred))

    # ========================================================
    # IMPORTANT NOTES FOR PAPER
    # ========================================================
    # 1. GroupKFold prevents subject leakage.
    # 2. Same subject never appears in train and test.
    # 3. StandardScaler fitted only on training data.
    # 4. No additional overlapping segmentation applied.
    # 5. Original 4-second epochs used directly.
    # ========================================================
 
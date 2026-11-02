# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:12:36 2026

@author: Raji
"""

# -*- coding: utf-8 -*-

"""
=============================================================
SUBJECT WISE TRIAL and condition-specific EEG CLASSIFICATION
=============================================================

Dataset structure:

E:\4 sec\subjectsGPT\
    |
    |-- CNT_01\
    |      |-- engage\
    |      |     |-- *.set
    |      |
    |      |-- disengage\
    |            |-- *.set
    |
    |-- CNT_02\
    |      |-- engage\
    |      |-- disengage\
    |
    |-- ...

IMPORTANT
---------
ONE .SET FILE = ONE TRIAL

All epochs inside one .set file are kept together.

Cross-validation is performed at the TRIAL level.

Therefore:

    Trial -> Train OR Test

never:

    Trial -> some epochs Train + some epochs Test


Classification:
    Engage    = 1
    Disengage = 0

Features:
    Delta  1-4 Hz
    Theta  4-8 Hz
    Alpha  8-13 Hz
    Beta   13-30 Hz
    Gamma  30-40 Hz

Classifier:
    StandardScaler
    Random Forest

Cross-validation:
    StratifiedKFold
    N_SPLITS = 2

Environment:
    Python 3.11
    NumPy 1.26.4
    SciPy 1.17.1
    MNE 1.7.0
"""

# ============================================================
# SCIPY COMPATIBILITY PATCH
# ============================================================

import scipy
import scipy.special

print("SciPy version:", scipy.__version__)


# SciPy 1.17 removed scipy.special.sph_harm.
#
# MNE 1.7 may still request sph_harm.
#
# Therefore create a compatibility wrapper BEFORE importing MNE.

if not hasattr(scipy.special, "sph_harm"):

    if hasattr(scipy.special, "sph_harm_y"):

        def sph_harm(m, n, theta, phi, out=None):

            result = scipy.special.sph_harm_y(
                n,
                m,
                phi,
                theta
            )

            if out is not None:
                out[...] = result
                return out

            return result

        scipy.special.sph_harm = sph_harm

        print(
            "SciPy compatibility patch applied:"
        )

        print(
            "scipy.special.sph_harm -> "
            "scipy.special.sph_harm_y"
        )

    else:

        raise ImportError(
            "SciPy does not contain sph_harm "
            "or sph_harm_y."
        )


# ============================================================
# IMPORTS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

import mne

from scipy.signal import welch

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline


# ============================================================
# VERSION INFORMATION
# ============================================================

print()
print("========================================")
print("EEG CLASSIFICATION ENVIRONMENT")
print("========================================")

print("NumPy :", np.__version__)
print("SciPy :", scipy.__version__)
print("MNE   :", mne.__version__)

print("========================================")
print()

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"E:\EEG\STUD ENGAGE\All_files_4sec_epoched_organized_used_in_pgm"

EXPECTED_SAMPLING_RATE = 250

WINDOW_SIZE = 4

WINDOW_OVERLAP = 0

# ------------------------------------------------------------
# IMPORTANT:
#
# We now perform CV at the TRIAL level.
#
# Because your subjects have only a small number of trials,
# use 2 folds.
# ------------------------------------------------------------

N_SPLITS = 4

RANDOM_STATE = 42


# ============================================================
# EEG FREQUENCY BANDS
# ============================================================

BANDS = {

    "delta": (1, 4),

    "theta": (4, 8),

    "alpha": (8, 13),

    "beta": (13, 30),

    "gamma": (30, 40)

}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(epoch_data, sfreq):

    """
    Extract PSD band-power features from one EEG epoch.

    Input:
        epoch_data:
            channels x samples

    Output:
        One feature vector.

    For each channel:

        Delta
        Theta
        Alpha
        Beta
        Gamma

    Therefore:

        features =
        number_of_channels x 5
    """

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if epoch_data.ndim != 2:

        raise ValueError(
            "Expected epoch shape "
            "(channels, samples), "
            f"got {epoch_data.shape}"
        )

    n_channels, n_samples = epoch_data.shape


    # --------------------------------------------------------
    # Welch parameters
    # --------------------------------------------------------

    nperseg = min(
        500,
        n_samples
    )

    if nperseg < 8:

        raise ValueError(
            f"Too few samples: {n_samples}"
        )


    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    features = []


    # ========================================================
    # CHANNEL LOOP
    # ========================================================

    for ch_index in range(n_channels):

        channel_data = epoch_data[ch_index]


        # ----------------------------------------------------
        # Welch PSD
        # ----------------------------------------------------

        freqs, psd = welch(

            channel_data,

            fs=sfreq,

            window="hann",

            nperseg=nperseg,

            noverlap=nperseg // 2,

            detrend="constant",

            scaling="density"

        )


        # ----------------------------------------------------
        # Frequency-band power
        # ----------------------------------------------------

        for band_name, (low, high) in BANDS.items():

            idx = np.logical_and(

                freqs >= low,

                freqs <= high

            )


            if not np.any(idx):

                band_power = 0.0

            else:

                # NumPy 1.26 compatible
                band_power = np.trapz(

                    psd[idx],

                    freqs[idx]

                )


            features.append(
                band_power
            )


    # --------------------------------------------------------
    # Convert to array
    # --------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(features)
    ):

        raise ValueError(
            "Feature vector contains "
            "NaN or infinite values."
        )


    return features


# ============================================================
# LOAD ONE EEGLAB FILE
# ============================================================

def load_set_file(file_path):

    """
    Load one EEGLAB epoched .set file.

    Returns:

        data
            shape:
            epochs x channels x samples

        sfreq
            sampling frequency
    """

    print()
    print("Loading:")

    print(file_path)


    # --------------------------------------------------------
    # Read EEGLAB epochs
    # --------------------------------------------------------

    epochs = mne.read_epochs_eeglab(

        file_path,

        verbose=False,

        montage_units="auto"

    )


    # --------------------------------------------------------
    # Sampling frequency
    # --------------------------------------------------------

    sfreq = float(
        epochs.info["sfreq"]
    )


    print(
        f"  Sampling frequency: "
        f"{sfreq:.2f} Hz"
    )


    # --------------------------------------------------------
    # Check sampling rate
    # --------------------------------------------------------

    if not np.isclose(

        sfreq,

        EXPECTED_SAMPLING_RATE,

        atol=1

    ):

        warnings.warn(

            f"Expected approximately "
            f"{EXPECTED_SAMPLING_RATE} Hz, "
            f"but found {sfreq:.2f} Hz:\n"
            f"{file_path}"

        )


    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------

    data = epochs.get_data()


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if data.ndim != 3:

        raise ValueError(

            f"Unexpected EEG shape: "
            f"{data.shape}"

        )


    print(
        f"  Number of epochs : "
        f"{data.shape[0]}"
    )

    print(
        f"  Number of channels: "
        f"{data.shape[1]}"
    )

    print(
        f"  Samples/epoch    : "
        f"{data.shape[2]}"
    )

    print(
        f"  Epoch duration   : "
        f"{data.shape[2] / sfreq:.3f} seconds"
    )

    
    return data, sfreq


# ============================================================
# LOAD ONE SUBJECT
# ============================================================

def load_subject_data(subject_path):

    """
    Load all .set files belonging to one subject.

    IMPORTANT:

        ONE .SET FILE = ONE TRIAL

    Each epoch in the same .set file receives
    the same trial ID.

    Returns:

        X
            feature matrix

        y
            epoch labels

        groups
            trial ID for every epoch
    """


    X = []

    y = []

    groups = []


    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_mapping = {

        "engage": 1,

        "disengage": 0

    }


    # --------------------------------------------------------
    # Trial counter
    # --------------------------------------------------------

    trial_counter = 0


    # ========================================================
    # CLASS LOOP
    # ========================================================

    for class_name, label in class_mapping.items():


        class_path = os.path.join(

            subject_path,

            class_name

        )


        # ----------------------------------------------------
        # Check folder
        # ----------------------------------------------------

        if not os.path.isdir(
            class_path
        ):

            print()

            print(
                "WARNING: Folder not found:"
            )

            print(
                class_path
            )

            continue


        # ----------------------------------------------------
        # Find SET files
        # ----------------------------------------------------

        files = sorted(

            [

                f

                for f in os.listdir(
                    class_path
                )

                if f.lower().endswith(
                    ".set"
                )

            ]

        )


        print()

        print(
            f"Class: {class_name}"
        )

        print(
            f"Number of .set files: "
            f"{len(files)}"
        )


        # ====================================================
        # FILE / TRIAL LOOP
        # ====================================================

        for file in files:


            file_path = os.path.join(

                class_path,

                file

            )


            # ------------------------------------------------
            # Assign ONE trial ID
            # ------------------------------------------------

            trial_id = trial_counter

            trial_counter += 1


            print()

            print(
                "----------------------------------------"
            )

            print(
                f"Trial Group {trial_id}"
            )

            print(
                f"Class       : {class_name}"
            )

            print(
                f"File        : {file}"
            )

            print(
                "----------------------------------------"
            )


            try:


                # --------------------------------------------
                # Load SET file
                # --------------------------------------------

                epoch_data, sfreq = load_set_file(

                    file_path

                )


                # --------------------------------------------
                # Number of epochs
                # --------------------------------------------

                n_epochs = epoch_data.shape[0]


                valid_epochs = 0


                # ============================================
                # EPOCH LOOP
                # ============================================

                for epoch_index in range(
                    n_epochs
                ):


                    epoch = epoch_data[
                        epoch_index
                    ]


                    # ----------------------------------------
                    # Extract features
                    # ----------------------------------------

                    features = extract_features(

                        epoch,

                        sfreq

                    )


                    # ----------------------------------------
                    # Store feature vector
                    # ----------------------------------------

                    X.append(
                        features
                    )


                    # ----------------------------------------
                    # Store class
                    # ----------------------------------------

                    y.append(
                        label
                    )


                    # ----------------------------------------
                    # Store TRIAL ID
                    #
                    # Every epoch from this .set file
                    # receives exactly the same trial ID.
                    # ----------------------------------------

                    groups.append(
                        trial_id
                    )


                    valid_epochs += 1


                print()

                print(
                    f"Features extracted: "
                    f"{valid_epochs}"
                )


            except Exception as e:


                print()

                print(
                    "ERROR processing file:"
                )

                print(
                    file_path
                )

                print(
                    f"Error: "
                    f"{type(e).__name__}: {e}"
                )

                print(
                    "This trial will be skipped."
                )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    if len(X) == 0:

        return (

            np.empty(
                (0, 0)
            ),

            np.array([]),

            np.array([])

        )


    X = np.asarray(

        X,

        dtype=np.float64

    )


    y = np.asarray(

        y,

        dtype=np.int64

    )


    groups = np.asarray(

        groups,

        dtype=np.int64

    )


    return X, y, groups


# ============================================================
# CREATE TRIAL-LEVEL FOLDS
# ============================================================

def create_trial_level_folds(
        y,
        groups,
        n_splits=2,
        random_state=42):

    """
    Create cross-validation folds using TRIALS as
    the independent observations.

    This is different from directly applying
    StratifiedGroupKFold to all epochs.

    ----------------------------------------------------------
    STEP 1
    ----------------------------------------------------------

    Find unique trials.

    Example:

        Trial 0 -> Engage
        Trial 1 -> Engage
        Trial 2 -> Engage
        Trial 3 -> Engage
        Trial 4 -> Disengage
        Trial 5 -> Disengage
        Trial 6 -> Disengage

    ----------------------------------------------------------
    STEP 2
    ----------------------------------------------------------

    Apply StratifiedKFold to those 7 trials.

    ----------------------------------------------------------
    STEP 3
    ----------------------------------------------------------

    Convert the selected trial IDs back into
    epoch indices.

    ----------------------------------------------------------

    Therefore ALL epochs from one trial always
    remain in one fold.
    """


    # ========================================================
    # UNIQUE TRIALS
    # ========================================================

    unique_trials = np.unique(
        groups
    )


    # ========================================================
    # GET ONE LABEL PER TRIAL
    # ========================================================

    trial_labels = []


    for trial_id in unique_trials:


        # --------------------------------------------
        # All epochs belonging to this trial
        # --------------------------------------------

        trial_y = y[
            groups == trial_id
        ]


        # --------------------------------------------
        # Unique class labels
        # --------------------------------------------

        unique_labels = np.unique(
            trial_y
        )


        # --------------------------------------------
        # Each trial must have exactly one class
        # --------------------------------------------

        if len(unique_labels) != 1:

            raise ValueError(

                f"Trial {trial_id} contains "
                f"multiple classes: "
                f"{unique_labels}"

            )


        trial_labels.append(
            trial_y[0]
        )


    trial_labels = np.asarray(

        trial_labels,

        dtype=np.int64

    )


    # ========================================================
    # DISPLAY TRIAL INFORMATION
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "TRIAL-LEVEL DATA"
    )

    print(
        "========================================"
    )


    total_engaged_trials = np.sum(
        trial_labels == 1
    )

    total_disengaged_trials = np.sum(
        trial_labels == 0
    )


    print(
        f"Total trials       : "
        f"{len(unique_trials)}"
    )

    print(
        f"Engaged trials     : "
        f"{total_engaged_trials}"
    )

    print(
        f"Disengaged trials  : "
        f"{total_disengaged_trials}"
    )


    print()

    print(
        "Trial ID | Class      | Epochs"
    )

    print(
        "--------------------------------"
    )


    # ========================================================
    # PRINT EVERY TRIAL
    # ========================================================

    for trial_id, label in zip(

        unique_trials,

        trial_labels

    ):


        if label == 1:

            class_name = "Engaged"

        else:

            class_name = "Disengaged"


        n_epochs = np.sum(

            groups == trial_id

        )


        print(

            f"{trial_id:8d} | "
            f"{class_name:10s} | "
            f"{n_epochs:6d}"

        )


    # ========================================================
    # CHECK NUMBER OF TRIALS PER CLASS
    # ========================================================

    if total_engaged_trials < n_splits:

        raise ValueError(

            f"Only {total_engaged_trials} "
            f"engaged trials are available, "
            f"but n_splits={n_splits}."

        )


    if total_disengaged_trials < n_splits:

        raise ValueError(

            f"Only {total_disengaged_trials} "
            f"disengaged trials are available, "
            f"but n_splits={n_splits}."

        )


    # ========================================================
    # TRIAL-LEVEL STRATIFIED CV
    # ========================================================

    cv = StratifiedKFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=random_state

    )


    # ========================================================
    # CREATE FOLDS
    # ========================================================

    for trial_train_idx, trial_test_idx in cv.split(

        unique_trials,

        trial_labels

    ):


        # --------------------------------------------
        # Trial IDs for training
        # --------------------------------------------

        train_trials = unique_trials[
            trial_train_idx
        ]


        # --------------------------------------------
        # Trial IDs for testing
        # --------------------------------------------

        test_trials = unique_trials[
            trial_test_idx
        ]


        # --------------------------------------------
        # Convert trial IDs into epoch indices
        # --------------------------------------------

        train_idx = np.where(

            np.isin(

                groups,

                train_trials

            )

        )[0]


        test_idx = np.where(

            np.isin(

                groups,

                test_trials

            )

        )[0]


        yield (

            train_idx,

            test_idx,

            train_trials,

            test_trials

        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not os.path.isdir(
        DATASET_PATH
    ):

        raise FileNotFoundError(

            f"Dataset folder does not exist:\n"
            f"{DATASET_PATH}"

        )


    # ========================================================
    # FIND SUBJECTS
    # ========================================================

    subjects = sorted(

        [

            s

            for s in os.listdir(
                DATASET_PATH
            )

            if os.path.isdir(

                os.path.join(

                    DATASET_PATH,

                    s

                )

            )

        ]

    )


    # ========================================================
    # GENERAL INFORMATION
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "TRIAL-LEVEL EEG CLASSIFICATION USING SVM"
    )

    print(
        "========================================"
    )

    print(
        f"Dataset: "
        f"{DATASET_PATH}"
    )

    print(
        f"Subjects found: "
        f"{len(subjects)}"
    )

    print(
        f"Cross-validation folds: "
        f"{N_SPLITS}"
    )

    print()

    print(
        "Frequency bands:"
    )

    for name, limits in BANDS.items():

        print(
            f"  {name:8s}: "
            f"{limits[0]}-{limits[1]} Hz"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "One .set file = one independent trial."
    )

    print(
        "CV is performed at the trial level."
    )

    print(
        "========================================"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    subject_results = []

    subject_f1_results = []

    # One summary row per subject, including subjects whose
    # epoch counts are available but classification is skipped.
    subject_summary_rows = []


    # ========================================================
    # SUBJECT LOOP
    # ========================================================

    for subject in subjects:


        subject_path = os.path.join(

            DATASET_PATH,

            subject

        )


        print()

        print()

        print(
            "========================================"
        )

        print(
            f"PROCESSING SUBJECT: {subject}"
        )

        print(
            "========================================"
        )


        # ====================================================
        # LOAD SUBJECT
        # ====================================================

        X, y, groups = load_subject_data(

            subject_path

        )

        # Counts are based on epochs whose features were extracted
        # successfully by load_subject_data().
        engaged_epochs = int(np.sum(y == 1)) if len(y) else 0
        disengaged_epochs = int(np.sum(y == 0)) if len(y) else 0

        # Default metric values remain blank (NaN) if classification
        # cannot be performed for this subject.
        mean_accuracy = np.nan
        std_accuracy = np.nan
        mean_f1 = np.nan
        std_f1 = np.nan


        # ====================================================
        # CHECK DATA
        # ====================================================

        if len(X) == 0:

            print()

            print(
                "No valid EEG data "
                "found for this subject."
            )

            subject_summary_rows.append({
                "Subject": subject,
                "Eng_Epochs": engaged_epochs,
                "Diseng_Epochs": disengaged_epochs,
                "Accuracy_Mean": mean_accuracy,
                "Accuracy_Std": std_accuracy,
                "F1_Mean": mean_f1,
                "F1_Std": std_f1,
                "Status": "No valid EEG data"
            })
            continue


        # ====================================================
        # BASIC DATA SUMMARY
        # ====================================================

        print()

        print(
            "----------------------------------------"
        )

        print(
            "SUBJECT DATA SUMMARY"
        )

        print(
            "----------------------------------------"
        )


        unique_trials = np.unique(
            groups
        )


        print(
            f"Feature shape       : "
            f"{X.shape}"
        )

        print(
            f"Number of epochs    : "
            f"{len(X)}"
        )

        print(
            f"Number of trials    : "
            f"{len(unique_trials)}"
        )

        print(
            f"Engaged epochs      : "
            f"{np.sum(y == 1)}"
        )

        print(
            f"Disengaged epochs   : "
            f"{np.sum(y == 0)}"
        )


        # ====================================================
        # CHECK TRIAL CLASSIFICATION
        # ====================================================

        trial_classes = []


        for trial_id in unique_trials:


            trial_y = y[
                groups == trial_id
            ]


            unique_labels = np.unique(
                trial_y
            )


            if len(unique_labels) != 1:

                raise ValueError(

                    f"Trial {trial_id} contains "
                    f"multiple class labels: "
                    f"{unique_labels}"

                )


            trial_classes.append(
                trial_y[0]
            )


        trial_classes = np.asarray(
            trial_classes
        )


        print(
            f"Engaged trials      : "
            f"{np.sum(trial_classes == 1)}"
        )

        print(
            f"Disengaged trials   : "
            f"{np.sum(trial_classes == 0)}"
        )


        # ====================================================
        # CHECK TRIAL COUNTS
        # ====================================================

        if np.sum(
            trial_classes == 1
        ) < N_SPLITS:

            print()

            print(
                "WARNING:"
            )

            print(
                "Not enough engaged trials "
                "for the selected number "
                "of folds."
            )

            print(
                f"Engaged trials = "
                f"{np.sum(trial_classes == 1)}"
            )

            subject_summary_rows.append({
                "Subject": subject,
                "Eng_Epochs": engaged_epochs,
                "Diseng_Epochs": disengaged_epochs,
                "Accuracy_Mean": mean_accuracy,
                "Accuracy_Std": std_accuracy,
                "F1_Mean": mean_f1,
                "F1_Std": std_f1,
                "Status": f"Insufficient engaged trials for {N_SPLITS}-fold CV"
            })
            continue


        if np.sum(
            trial_classes == 0
        ) < N_SPLITS:

            print()

            print(
                "WARNING:"
            )

            print(
                "Not enough disengaged trials "
                "for the selected number "
                "of folds."
            )

            print(
                f"Disengaged trials = "
                f"{np.sum(trial_classes == 0)}"
            )

            subject_summary_rows.append({
                "Subject": subject,
                "Eng_Epochs": engaged_epochs,
                "Diseng_Epochs": disengaged_epochs,
                "Accuracy_Mean": mean_accuracy,
                "Accuracy_Std": std_accuracy,
                "F1_Mean": mean_f1,
                "F1_Std": std_f1,
                "Status": f"Insufficient disengaged trials for {N_SPLITS}-fold CV"
            })
            continue


        # ====================================================
        # METRICS
        # ====================================================

        fold_accuracies = []

        fold_f1_scores = []

        all_true = []

        all_pred = []


        # ====================================================
        # TRIAL-LEVEL CROSS VALIDATION
        # ====================================================

        for fold, (

            train_idx,

            test_idx,

            train_trials,

            test_trials

        ) in enumerate(

            create_trial_level_folds(

                y,

                groups,

                n_splits=N_SPLITS,

                random_state=RANDOM_STATE

            ),

            1

        ):


            print()

            print()

            print(
                "========================================"
            )

            print(
                f"FOLD {fold}"
            )

            print(
                "========================================"


            )


            # =================================================
            # CHECK TRAIN / TEST TRIALS
            # =================================================

            overlap = np.intersect1d(

                train_trials,

                test_trials

            )


            print()

            print(
                "Training trials:"
            )

            print(
                train_trials
            )


            print()

            print(
                "Testing trials:"
            )

            print(
                test_trials
            )


            print()

            print(
                f"Training trial count: "
                f"{len(train_trials)}"
            )

            print(
                f"Testing trial count : "
                f"{len(test_trials)}"
            )


            # =================================================
            # LEAKAGE CHECK
            # =================================================

            print()

            print(
                f"Shared trials between "
                f"train/test: "
                f"{len(overlap)}"
            )


            if len(overlap) != 0:

                raise RuntimeError(

                    "TRIAL LEAKAGE DETECTED!"

                )


            # =================================================
            # CHECK CLASS BALANCE AT TRIAL LEVEL
            # =================================================

            train_trial_labels = []

            test_trial_labels = []


            for trial_id in train_trials:

                train_trial_labels.append(

                    y[
                        groups == trial_id
                    ][0]

                )


            for trial_id in test_trials:

                test_trial_labels.append(

                    y[
                        groups == trial_id
                    ][0]

                )


            train_trial_labels = np.asarray(
                train_trial_labels
            )

            test_trial_labels = np.asarray(
                test_trial_labels
            )


            print()

            print(
                "Training trial classes:"
            )

            print(

                f"  Engaged    : "
                f"{np.sum(train_trial_labels == 1)}"

            )

            print(

                f"  Disengaged : "
                f"{np.sum(train_trial_labels == 0)}"

            )


            print()

            print(
                "Testing trial classes:"
            )

            print(

                f"  Engaged    : "
                f"{np.sum(test_trial_labels == 1)}"

            )

            print(

                f"  Disengaged : "
                f"{np.sum(test_trial_labels == 0)}"

            )


            # =================================================
            # EPOCH COUNTS
            # =================================================

            print()

            print(
                f"Training epochs: "
                f"{len(train_idx)}"
            )

            print(
                f"Testing epochs : "
                f"{len(test_idx)}"
            )


            # =================================================
            # CREATE TRAIN / TEST DATA
            # =================================================

            X_train = X[
                train_idx
            ]

            X_test = X[
                test_idx
            ]

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]


            # =================================================
            # CLASSIFIER
            # =================================================

            clf = Pipeline(

                [

                    (
                        "scaler",

                        StandardScaler()

                    ),

                    (

                        "svm",

                        SVC(

                            kernel="rbf",

                            C=1.0,

                            gamma="scale",

                            class_weight="balanced",

                            random_state=RANDOM_STATE

                        )

                    )

                ]

            )


            # =================================================
            # TRAIN
            # =================================================

            clf.fit(

                X_train,

                y_train

            )


            # =================================================
            # PREDICTION
            # =================================================

            y_pred = clf.predict(

                X_test

            )


            # =================================================
            # METRICS
            # =================================================

            accuracy = accuracy_score(

                y_test,

                y_pred

            )


            f1 = f1_score(

                y_test,

                y_pred,

                average="weighted"

            )


            fold_accuracies.append(

                accuracy

            )


            fold_f1_scores.append(

                f1

            )


            all_true.extend(

                y_test.tolist()

            )


            all_pred.extend(

                y_pred.tolist()

            )


            # =================================================
            # FOLD RESULT
            # =================================================

            print()

            print(
                "----------------------------------------"
            )

            print(
                f"Fold {fold} Results"
            )

            print(
                "----------------------------------------"
            )

            print(

                f"Accuracy : "
                f"{accuracy:.4f}"

            )

            print(

                f"F1 Score : "
                f"{f1:.4f}"

            )


        # ====================================================
        # SUBJECT PERFORMANCE
        # ====================================================

        mean_accuracy = np.mean(

            fold_accuracies

        )


        std_accuracy = np.std(

            fold_accuracies

        )


        mean_f1 = np.mean(

            fold_f1_scores

        )


        std_f1 = np.std(

            fold_f1_scores

        )


        # ====================================================
        # SAVE SUBJECT RESULT
        # ====================================================

        subject_results.append(

            mean_accuracy

        )


        subject_f1_results.append(

            mean_f1

        )

        subject_summary_rows.append({
            "Subject": subject,
            "Eng_Epochs": engaged_epochs,
            "Diseng_Epochs": disengaged_epochs,
            "Accuracy_Mean": mean_accuracy,
            "Accuracy_Std": std_accuracy,
            "F1_Mean": mean_f1,
            "F1_Std": std_f1,
            "Status": "Success"
        })


        # ====================================================
        # SUBJECT RESULTS
        # ====================================================

        print()

        print()

        print(
            "========================================"
        )

        print(
            f"SUBJECT RESULT: {subject}"
        )

        print(
            "========================================"
        )


        print()

        print(

            f"Mean Accuracy : "
            f"{mean_accuracy:.4f}"

        )

        print(

            f"Std Accuracy  : "
            f"{std_accuracy:.4f}"

        )

        print()

        print(

            f"Mean F1 Score : "
            f"{mean_f1:.4f}"

        )

        print(

            f"Std F1 Score  : "
            f"{std_f1:.4f}"

        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        print()

        print(
            "========================================"
        )

        print(
            "CLASSIFICATION REPORT"
        )

        print(
            "========================================"
        )


        print(

            classification_report(

                all_true,

                all_pred,

                labels=[0, 1],

                target_names=[

                    "Disengaged",

                    "Engaged"

                ],

                zero_division=0

            )

        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(

            all_true,

            all_pred,

            labels=[0, 1]

        )


        print()

        print(
            "Confusion Matrix"
        )

        print()

        print(
            "                  Predicted"
        )

        print(
            "              Disengaged  Engaged"
        )

        print(

            f"Actual "
            f"Disengaged   "
            f"{cm[0,0]:8d}  "
            f"{cm[0,1]:8d}"

        )

        print(

            f"Actual "
            f"Engaged      "
            f"{cm[1,0]:8d}  "
            f"{cm[1,1]:8d}"

        )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()

    print()

    print(
        "========================================"
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "========================================"


    )


    if len(subject_results) == 0:

        print()

        print(
            "No subjects produced valid results."
        )


    else:


        # ----------------------------------------------------
        # Average subject accuracy
        # ----------------------------------------------------

        overall_accuracy = np.mean(

            subject_results

        )


        overall_accuracy_std = np.std(

            subject_results

        )


        # ----------------------------------------------------
        # Average subject F1
        # ----------------------------------------------------

        overall_f1 = np.mean(

            subject_f1_results

        )


        overall_f1_std = np.std(

            subject_f1_results

        )


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        print(

            f"Subjects processed: "
            f"{len(subject_results)}"

        )

        print()

        print(

            f"Average Accuracy : "
            f"{overall_accuracy:.4f}"

        )

        print(

            f"Std Accuracy     : "
            f"{overall_accuracy_std:.4f}"

        )

        print()

        print(

            f"Average F1 Score : "
            f"{overall_f1:.4f}"

        )

        print(

            f"Std F1 Score     : "
            f"{overall_f1_std:.4f}"

        )


    # ========================================================
    # SAVE SUBJECT-WISE RESULTS
    # ========================================================

    results_df = pd.DataFrame(subject_summary_rows)

    # Keep the requested columns first; Status explains blank metrics.
    requested_columns = [
        "Subject",
        "Eng_Epochs",
        "Diseng_Epochs",
        "Accuracy_Mean",
        "Accuracy_Std",
        "F1_Mean",
        "F1_Std",
        "Status"
    ]
    results_df = results_df.reindex(columns=requested_columns)

    csv_path = os.path.join(DATASET_PATH, "subjectwise_EEG_classification_results.csv")
    xlsx_path = os.path.join(DATASET_PATH, "subjectwise_EEG_classification_results.xlsx")

    results_df.to_csv(csv_path, index=False)

    try:
        results_df.to_excel(xlsx_path, index=False)
        print(f"Excel results saved to: {xlsx_path}")
    except Exception as e:
        print(f"Could not save Excel file: {e}")
        print("The CSV results file was saved successfully.")

    print(f"CSV results saved to: {csv_path}")
    print("\nSUBJECT-WISE RESULTS TABLE")
    print(results_df.to_string(index=False))


    # ========================================================
    # END
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "PROGRAM FINISHED"
    )

    print(
        "========================================"
    )
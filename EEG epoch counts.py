# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:28:00 2026

@author: Raji
"""


import os
import numpy as np
import mne

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    r"E:\EEG\STUD ENGAGE"
    r"\All_files_4sec_epoched_organized_used_in_pgm"
)

# ============================================================
# LOAD DATASET AND COUNT EPOCHS
# ============================================================

def load_dataset(dataset_path):
    """
    Count the actual epochs stored in Engage and Disengage
    EEGLAB .set files for each subject.

    Expected folder structure:
        dataset/
            SUB_01/
                Engage/
                    file1.set
                Disengage/
                    file2.set

    Returns
    -------
    total_engage_epochs : int
    total_disengage_epochs : int
    """

    total_engage_epochs = 0
    total_disengage_epochs = 0

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    subjects = sorted(os.listdir(dataset_path))

    print("\nCounting EEG epochs...")
    print("=" * 65)

    for subject in subjects:

        subject_path = os.path.join(dataset_path, subject)

        if not os.path.isdir(subject_path):
            continue

        subject_engage = 0
        subject_disengage = 0

        class_mapping = {
            "Engage": "engage",
            "Disengage": "disengage"
        }

        for class_name, condition in class_mapping.items():

            class_path = os.path.join(subject_path, class_name)

            if not os.path.isdir(class_path):
                continue

            files = sorted(
                f for f in os.listdir(class_path)
                if f.lower().endswith(".set")
            )

            for file in files:

                file_path = os.path.join(class_path, file)

                try:
                    epochs = mne.read_epochs_eeglab(
                        file_path,
                        verbose=False
                    )

                    n_epochs = len(epochs)

                    if condition == "engage":
                        subject_engage += n_epochs
                    else:
                        subject_disengage += n_epochs

                    print(
                        f"{subject} | {condition.capitalize():10s} | "
                        f"{file} | Epochs: {n_epochs}"
                    )

                except Exception as e:
                    print(
                        f"ERROR loading {file_path}\n"
                        f"Reason: {e}"
                    )

        total_engage_epochs += subject_engage
        total_disengage_epochs += subject_disengage

        print("-" * 65)
        print(
            f"{subject} totals -> "
            f"Engage: {subject_engage}, "
            f"Disengage: {subject_disengage}"
        )
        print("-" * 65)

    return total_engage_epochs, total_disengage_epochs


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    engage_count, disengage_count = load_dataset(DATASET_PATH)

    print("\n")
    print("=" * 45)
    print("       FINAL EEG EPOCH COUNTS")
    print("=" * 45)
    print(f"Total Engage epochs    : {engage_count}")
    print(f"Total Disengage epochs : {disengage_count}")
    print(
        f"Grand total epochs     : "
        f"{engage_count + disengage_count}"
    )
    print("=" * 45)

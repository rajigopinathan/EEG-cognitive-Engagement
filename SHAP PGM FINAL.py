# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:13:34 2026

@author: Raji
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

# For this ChatGPT environment:
CSV_PATH = "E:\EEG\STUD ENGAGE\All_files_4sec_epoched_45_12 model\psd_outputs\SHAP_RF_4sec_results/epoch_level_PSD_features.csv"

# For your Windows computer, replace the path above, for example:
# CSV_PATH = r"E:\EEG\STUD ENGAGE\epoch_level_PSD_features.csv"

OUTPUT_DIR = "SHAP_LOSO_4sec_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 300
TOP_N = 10

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

metadata_cols = [
    "subject_folder_id",
    "file_name",
    "condition",
    "trial_number"
]

missing_cols = [
    col for col in metadata_cols if col not in df.columns
]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

feature_cols = [
    col for col in df.columns if col not in metadata_cols
]

X_original = df[feature_cols].apply(
    pd.to_numeric, errors="raise"
).to_numpy(dtype=np.float64)

y = pd.to_numeric(
    df["condition"], errors="raise"
).to_numpy(dtype=int)

groups = df["subject_folder_id"].astype(str).to_numpy()

if not np.isfinite(X_original).all():
    raise ValueError("Feature matrix contains NaN or infinite values.")

if set(np.unique(y)) != {0, 1}:
    raise ValueError(
        f"Expected binary labels 0 and 1; found {np.unique(y)}"
    )

if len(np.unique(groups)) < 2:
    raise ValueError("LOSO requires at least two subjects.")

# Scale very small PSD numbers for numerical stability.
# Scaling is fitted nowhere: this is a fixed constant applied
# equally to every feature and every subject.
SCALE_FACTOR = 1e12
X = X_original * SCALE_FACTOR

print("=" * 65)
print("DATA SUMMARY")
print("=" * 65)
print("Rows / epochs:", X.shape[0])
print("PSD features:", X.shape[1])
print("Subjects:", len(np.unique(groups)))
print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
print("Original feature range:",
      np.min(X_original), "to", np.max(X_original))
print("Scaled feature range:",
      np.min(X), "to", np.max(X))
print("Feature names:", feature_cols)

# Save the data summary.
summary = pd.DataFrame({
    "Metric": [
        "Number of epochs",
        "Number of subjects",
        "Number of PSD features",
        "Engaged epochs (condition=1)",
        "Disengaged epochs (condition=0)",
        "PSD scale factor"
    ],
    "Value": [
        len(df),
        len(np.unique(groups)),
        len(feature_cols),
        int(np.sum(y == 1)),
        int(np.sum(y == 0)),
        SCALE_FACTOR
    ]
})
summary.to_csv(
    os.path.join(OUTPUT_DIR, "Data_Summary.csv"),
    index=False
)

# ============================================================
# 3. LOSO RANDOM FOREST AND SHAP
# ============================================================

logo = LeaveOneGroupOut()

subject_importances = []
metrics = []
diagnostics = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    test_subjects = np.unique(groups[test_idx])
    if len(test_subjects) != 1:
        raise RuntimeError("A LOSO fold should hold out one subject.")

    held_out_subject = test_subjects[0]

    if len(np.unique(y_train)) < 2:
        print(
            f"Skipping {held_out_subject}: training data contain "
            "only one class."
        )
        continue

    # Train the same type of Random Forest model.
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # ----------------------------
    # LOSO performance
    # ----------------------------
    accuracy = accuracy_score(y_test, predictions)
    binary_f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )

    metrics.append({
        "Subject": held_out_subject,
        "Accuracy": accuracy,
        "Binary_F1_Condition1": binary_f1,
        "Balanced_Accuracy": balanced_acc,
        "Train_Epochs": len(train_idx),
        "Test_Epochs": len(test_idx),
        "Test_Engaged": int(np.sum(y_test == 1)),
        "Test_Disengaged": int(np.sum(y_test == 0))
    })

    for i, idx in enumerate(test_idx):
        all_predictions.append({
            "Subject": groups[idx],
            "True_Label": int(y[idx]),
            "Predicted_Label": int(predictions[i])
        })

    # ----------------------------
    # Random Forest diagnostics
    # ----------------------------
    rf_importances = rf.feature_importances_

    # Count features with positive impurity-based importance.
    n_nonzero = int(np.sum(rf_importances > 0))

    # Check the actual trained trees.
    total_tree_nodes = sum(
        tree.tree_.node_count for tree in rf.estimators_
    )
    total_split_nodes = sum(
        np.sum(tree.tree_.children_left !=
               tree.tree_.children_right)
        for tree in rf.estimators_
    )

    # ----------------------------
    # SHAP values for held-out subject
    # ----------------------------
    explainer = shap.TreeExplainer(rf)
    shap_output = explainer.shap_values(X_test)

    # Handle old and new SHAP output formats.
    if isinstance(shap_output, list):
        # Older SHAP: list of arrays, one per class.
        class_index = list(rf.classes_).index(1)
        shap_values = np.asarray(shap_output[class_index])

    elif hasattr(shap_output, "values"):
        shap_values = np.asarray(shap_output.values)

    else:
        shap_values = np.asarray(shap_output)

    # Newer SHAP versions can return:
    # (samples, features, classes)
    if shap_values.ndim == 3:
        class_index = list(rf.classes_).index(1)
        shap_values = shap_values[:, :, class_index]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP dimensions: {shap_values.shape}"
        )

    if shap_values.shape[1] != len(feature_cols):
        raise ValueError(
            "SHAP feature count does not match input feature count."
        )

    # Mean absolute SHAP per feature for this held-out subject.
    subject_shap_importance = np.mean(
        np.abs(shap_values), axis=0
    )

    subject_importances.append(subject_shap_importance)

    diagnostics.append({
        "Subject": held_out_subject,
        "RF_Importance_Sum": float(np.sum(rf_importances)),
        "RF_Nonzero_Features": n_nonzero,
        "Total_Tree_Nodes": int(total_tree_nodes),
        "Total_Split_Nodes": int(total_split_nodes),
        "SHAP_Min": float(np.min(shap_values)),
        "SHAP_Max": float(np.max(shap_values)),
        "Max_Absolute_SHAP": float(np.max(np.abs(shap_values))),
        "Mean_Absolute_SHAP": float(np.mean(np.abs(shap_values))),
        "SHAP_All_Zero": bool(np.all(shap_values == 0))
    })

    print(
        f"Fold {fold:02d}/{len(np.unique(groups))} | "
        f"Subject={held_out_subject} | "
        f"Accuracy={accuracy:.3f} | "
        f"F1={binary_f1:.3f} | "
        f"RF nonzero features={n_nonzero} | "
        f"Max |SHAP|={np.max(np.abs(shap_values)):.6e}"
    )

# ============================================================
# 4. SAVE METRICS AND DIAGNOSTICS
# ============================================================

if len(metrics) == 0:
    raise RuntimeError("No valid LOSO folds were completed.")

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "LOSO_Metrics_Recomputed.csv"),
    index=False
)

diagnostics_df = pd.DataFrame(diagnostics)
diagnostics_df.to_csv(
    os.path.join(OUTPUT_DIR, "SHAP_Diagnostics.csv"),
    index=False
)

predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "LOSO_Predictions.csv"),
    index=False
)

print("\n" + "=" * 65)
print("LOSO PERFORMANCE")
print("=" * 65)

print(
    metrics_df[
        ["Accuracy", "Binary_F1_Condition1", "Balanced_Accuracy"]
    ].agg(["mean", "std"]).round(4)
)

print("\nDiagnostic overview:")
print(
    diagnostics_df[
        [
            "RF_Nonzero_Features",
            "Total_Split_Nodes",
            "Max_Absolute_SHAP",
            "Mean_Absolute_SHAP"
        ]
    ].describe().round(8)
)

# ============================================================
# 5. AGGREGATE SHAP IMPORTANCE ACROSS SUBJECTS
# ============================================================

if len(subject_importances) == 0:
    raise RuntimeError("No SHAP values were collected.")

importance_matrix = np.vstack(subject_importances)

# Equal weighting of each held-out subject.
mean_abs_shap = np.mean(importance_matrix, axis=0)

if len(subject_importances) > 1:
    sd_abs_shap = np.std(
        importance_matrix, axis=0, ddof=1
    )
else:
    sd_abs_shap = np.zeros(len(feature_cols))

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Mean_Absolute_SHAP": mean_abs_shap,
    "SD_Across_Subjects": sd_abs_shap
}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0, "Rank", np.arange(1, len(importance_df) + 1)
)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "SHAP_All_Features.csv"),
    index=False
)

top10_df = importance_df.head(TOP_N).copy()

top10_df.to_csv(
    os.path.join(OUTPUT_DIR, "SHAP_Top10_Features.csv"),
    index=False
)

print("\n" + "=" * 65)
print("TOP 10 EEG PSD FEATURES")
print("=" * 65)
print(top10_df.to_string(index=False))

# ============================================================
# 6. PLOT TOP 10 SHAP FEATURES
# ============================================================

plot_df = top10_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=True
)

plt.figure(figsize=(11, 7))

plt.barh(
    plot_df["Feature"],
    plot_df["Mean_Absolute_SHAP"],
    xerr=plot_df["SD_Across_Subjects"],
    capsize=3
)

plt.xlabel("Mean absolute SHAP value for class 1")
plt.ylabel("EEG channel-band feature")
plt.title("Top 10 EEG PSD Features: RF, 4-second Concordant Trials")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "SHAP_Top10_Barplot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# 7. SHAP SUMMARY PLOT
# ============================================================

# Refit a model on all rows only for visualization of SHAP
# patterns. This is NOT used to calculate LOSO performance.
final_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_model.fit(X, y)

final_explainer = shap.TreeExplainer(final_model)
final_shap_output = final_explainer.shap_values(X)

if isinstance(final_shap_output, list):
    class_index = list(final_model.classes_).index(1)
    final_shap_values = np.asarray(
        final_shap_output[class_index]
    )
elif hasattr(final_shap_output, "values"):
    final_shap_values = np.asarray(final_shap_output.values)
else:
    final_shap_values = np.asarray(final_shap_output)

if final_shap_values.ndim == 3:
    class_index = list(final_model.classes_).index(1)
    final_shap_values = final_shap_values[:, :, class_index]

# Confirm values are not all zero.
print(
    "\nFull-data SHAP absolute maximum:",
    np.max(np.abs(final_shap_values))
)

if np.max(np.abs(final_shap_values)) > 0:
    shap.summary_plot(
        final_shap_values,
        X,
        feature_names=feature_cols,
        max_display=TOP_N,
        show=False
    )

    plt.title("SHAP Summary: RF, 4-second Concordant Trials")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "SHAP_Summary_Plot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
else:
    print(
        "WARNING: SHAP values remain all zero. "
        "Inspect SHAP_Diagnostics.csv and RF tree diagnostics."
    )

# ============================================================
# 8. FREQUENCY-BAND IMPORTANCE
# ============================================================

band_rows = []

for band in ["delta", "theta", "alpha", "beta", "gamma"]:
    band_features = importance_df[
        importance_df["Feature"].str.endswith(
            f"_{band}_power"
        )
    ]

    band_rows.append({
        "Band": band,
        "Total_Mean_Absolute_SHAP": band_features[
            "Mean_Absolute_SHAP"
        ].sum(),
        "Average_Mean_Absolute_SHAP": band_features[
            "Mean_Absolute_SHAP"
        ].mean()
    })

band_df = pd.DataFrame(band_rows).sort_values(
    "Total_Mean_Absolute_SHAP",
    ascending=False
)

band_df.to_csv(
    os.path.join(OUTPUT_DIR, "SHAP_Importance_By_Band.csv"),
    index=False
)

print("\nSHAP importance by frequency band:")
print(band_df.to_string(index=False))

print("\nAnalysis complete.")
print("Output folder:", os.path.abspath(OUTPUT_DIR))





import os
import pandas as pd
import numpy as np
from utils import *

filename = "apr_24_fine-tune_test_labels_all_samples.csv"#"tasty.csv" 
#filename = "modified_test_data_with_predictions.csv" # "tasty.csv"  # "fine-tune_test_labels_all_samples.csv"
#TEST_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", filename)
TEST_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs", filename)
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs")

if __name__ == "__main__":
    # load test outputs - sklearn
    df = pd.read_csv(TEST_PATH)
    print(df.head())

    y_labels = np.array(df["true_labels"])
    #y_labels = np.array(df["overall_class"])
    y_preds = np.array(df["predicted_labels"])
    #y_preds = np.array(df["predicted_label"])
    z_leaning = np.array(df["leaning"])
    print("-" * 80)

    # Get indices of left-leaning and right-leaning in the z_leaning groups array
    left_indices = np.where(z_leaning == 0)[0]
    right_indices = np.where(z_leaning == 1)[0]

    # Get sub-arrays of predictions and true labels for left-leaning and right-leaning
    left_preds = y_preds[left_indices]
    right_preds = y_preds[right_indices]

    left_labels = y_labels[left_indices]
    right_labels = y_labels[right_indices]

    (
        cm_alldata,
        tp,
        tn,
        fp,
        fn,
        accuracy,
        precision,
        recall,
        f1,
        tpr,
        fpr,
        precision_classwise,
        recall_classwise,
    ) = alldatascores(y_labels, y_preds)
    
    (
        left_cm,
        ltp,
        ltn,
        lfp,
        lfn,
        left_accuracy,
        left_precision,
        left_recall,
        left_f1,
        left_tpr,
        left_fpr,
        left_precision_classwise,
        left_recall_classwise,
    ) = leftscores(left_labels, left_preds)
    (
        right_cm,
        rtp,
        rtn,
        rfp,
        rfn,
        right_accuracy,
        right_precision,
        right_recall,
        right_f1,
        right_tpr,
        right_fpr,
        right_precision_classwise,
        right_recall_classwise,
    ) = rightscores(right_labels, right_preds)
    stat_par_diff, equal_opp_diff, dis_im_ratio, av_odds_diff = fairnessmetrics(
        y_labels, y_preds, z_leaning
    )


# Find indices of false negatives and false positives
false_negatives = np.where((y_labels == 1) & (y_preds == 0))[0]
false_positives = np.where((y_labels == 0) & (y_preds == 1))[0]

# Get corresponding rows from the dataframe
false_neg_df = df.iloc[false_negatives]
false_pos_df = df.iloc[false_positives]

# Save to csv files
false_neg_df.to_csv(os.path.join(OUTPUTS_PATH, "false_negatives.csv"), index=False)
false_pos_df.to_csv(os.path.join(OUTPUTS_PATH, "false_positives.csv"), index=False)

# Find indices of false negatives and false positives for left-leaning
left_false_negatives = np.where((left_labels == 1) & (left_preds == 0))[0]
left_false_positives = np.where((left_labels == 0) & (left_preds == 1))[0]

# Get corresponding rows from the dataframe
left_false_neg_df = df.iloc[left_indices[left_false_negatives]]
left_false_pos_df = df.iloc[left_indices[left_false_positives]]

# Save to csv files
left_false_neg_df.to_csv(
    os.path.join(OUTPUTS_PATH, "left_false_negatives.csv"), index=False
)
left_false_pos_df.to_csv(
    os.path.join(OUTPUTS_PATH, "left_false_positives.csv"), index=False
)

# Find indices of false negatives and false positives for right-leaning
right_false_negatives = np.where((right_labels == 1) & (right_preds == 0))[0]
right_false_positives = np.where((right_labels == 0) & (right_preds == 1))[0]

# Get corresponding rows from the dataframe
right_false_neg_df = df.iloc[right_indices[right_false_negatives]]
right_false_pos_df = df.iloc[right_indices[right_false_positives]]

# Save to csv files
right_false_neg_df.to_csv(
    os.path.join(OUTPUTS_PATH, "right_false_negatives.csv"), index=False
)
right_false_pos_df.to_csv(
    os.path.join(OUTPUTS_PATH, "right_false_positives.csv"), index=False
)

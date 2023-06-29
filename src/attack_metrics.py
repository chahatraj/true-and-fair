import os
import pandas as pd
import numpy as np
# from utils import *

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
)

filename = "modified_test_data_with_predictions.csv"
TEST_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", filename)
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs")

if __name__ == "__main__":
    # load test outputs - sklearn
    df = pd.read_csv(TEST_PATH)
    print(df.head())

    y_labels = np.array(df["overall_class"])
    y_preds = np.array(df["predicted_label"])
    #z_leaning = np.array(df["leaning"])
    print("-" * 80)

    cm_alldata = confusion_matrix(y_labels, y_preds)
    tn, fp, fn, tp = cm_alldata.ravel()
    accuracy = accuracy_score(y_labels, y_preds)
    precision = precision_score(y_labels, y_preds)
    recall = recall_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds)
    fpr, tpr, thresholds = roc_curve(y_labels, y_preds)
    precision_classwise = precision_score(y_labels, y_preds, average=None)
    recall_classwise = recall_score(y_labels, y_preds, average=None)

    # Calculate attack success rate
    total_attacks = tp + tn  # Total number of attacks (true label is "real")
    successful_attacks = fp  # Successful attacks (true label is "real" but predicted as "fake")
    attack_success_rate = (successful_attacks / total_attacks) * 100 if total_attacks > 0 else 0

    # Calculate delta accuracy 
    original_accuracy = 0.91  
    delta_accuracy = (original_accuracy - accuracy) * 100
    
    print("-" * 80)
    print("Confusion Matrix:")
    print(cm_alldata)

    # print("\n Here are the TN, FP, FN, TP for all data: \n")
    print(f"\nTrue negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives: {tp}")

    # print("\n Here are the Accuracy, Precision, Recall, and F1-score: \n")
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    # print("\n Here are the TPR and FPR: \n")
    print(f"TPR: {tpr[1]:.2f}")
    print(f"FPR: {fpr[1]:.2f}")

    print(f"\nPrecision for class 0 (fake): {precision_classwise[0]:.2f}")
    print(f"Precision for class 1 (real): {precision_classwise[1]:.2f}")
    print(f"Recall for class 0 (fake): {recall_classwise[0]:.2f}")
    print(f"Recall for class 1 (real): {recall_classwise[1]:.2f}\n")

    # Print attack success rate
    print(f"\nAttack Success Rate: {attack_success_rate:.2f}%\n")

    print(f"Delta Accuracy: {delta_accuracy:.2f}%")
    print("-" * 80)


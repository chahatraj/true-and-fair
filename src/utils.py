import os
import numpy as np
import pandas as pd

np.random.seed(42)
import warnings

warnings.filterwarnings("ignore")

from aif360.sklearn.metrics import (
    statistical_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
    average_odds_difference,
)

from aif360.sklearn.postprocessing import RejectOptionClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
)

OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs")


def debiasing_results(prob_labels_path, mode, experiments):
    # Load data
    prob_labels = pd.read_csv(prob_labels_path)
    labels = prob_labels[["true_label"]].copy()
    y_labels = np.array(labels["true_label"].tolist())
    z_leaning = np.array(prob_labels["z"].to_list())

    # Get indices of left-leaning and right-leaning in the z_leaning groups array
    left_indices = np.where(z_leaning == 0)[0]
    right_indices = np.where(z_leaning == 1)[0]
    left_labels = y_labels[left_indices]
    right_labels = y_labels[right_indices]
    left_z = z_leaning[left_indices]
    right_z = z_leaning[right_indices]

    method_name = "Reject Option Classifier"

    print("-" * 80)
    print(f"Debiasing on {mode} data using {method_name} ...")

    # Calculate accuracy and fairness metrics before debiasing
    pred_labels = (
        prob_labels[["class1_prob", "class2_prob"]]
        .idxmax(axis=1)
        .replace({"class1_prob": 0, "class2_prob": 1})
    )

    accuracy_before = accuracy_score(labels["true_label"].tolist(), list(pred_labels))
    y_preds = np.array(list(pred_labels))
    left_preds_before = y_preds[left_indices]
    right_preds_before = y_preds[right_indices]

    # print("-"*80)
    # print("Metrics Before Debiasing ...")
    # print("-"*80)
    (
        stat_par_diff_before,
        equal_opp_diff_before,
        dis_im_ratio_before,
        av_odds_diff_before,
    ) = fairnessmetrics(y_labels, y_preds, z_leaning, printing=False)
    (
        cm_before,
        tp_before,
        tn_before,
        fp_before,
        fn_before,
        accuracy_before,
        precision_before,
        recall_before,
        f1_before,
        tpr_before,
        fpr_before,
        precision_classwise_before,
        recall_classwise_before,
    ) = alldatascores(y_labels, y_preds)

    # Prepare data for debiasing
    prob_labels.drop(columns=["true_label"], inplace=True)
    prob_labels = prob_labels.reset_index(drop=True)
    prob_labels = prob_labels.set_index(["z", prob_labels.index])
    prob_labels.index.names = ["z", "sample_id"]

    """# De-bias using Reject Option Classifier (use this for custom threshold and margin value)
    ROC = RejectOptionClassifier(prot_attr="z", threshold=0.5, margin=0.01)
    ROC = ROC.fit(prob_labels, labels)
    debiased_y_pred = ROC.predict(prob_labels)"""

    # De-bias using Reject Option Classifier (use this for varying hyperparameters)

    results = []
    for threshold, margins in experiments.items():
        for margin in margins:
            # print(threshold, margin)
            # print("-"*80)
            ROC = RejectOptionClassifier(
                prot_attr="z", threshold=threshold, margin=margin
            )
            ROC = ROC.fit(prob_labels, labels)
            debiased_y_pred = ROC.predict(prob_labels)

            accuracy_after = accuracy_score(
                labels["true_label"].tolist(), list(debiased_y_pred)
            )
            y_preds = np.array(list(debiased_y_pred))
            left_preds_after = y_preds[left_indices]
            right_preds_after = y_preds[right_indices]

            # print("-"*80)
            # print("Metrics After Debiasing ...")
            # print("-"*80)
            (
                stat_par_diff_after,
                equal_opp_diff_after,
                dis_im_ratio_after,
                av_odds_diff_after,
            ) = fairnessmetrics(y_labels, y_preds, z_leaning, printing=False)
            (
                cm_after,
                tp_after,
                tn_after,
                fp_after,
                fn_after,
                accuracy_after,
                precision_after,
                recall_after,
                f1_after,
                tpr_after,
                fpr_after,
                precision_classwise_after,
                recall_classwise_after,
            ) = alldatascores(y_labels, y_preds)

            # Store results in a dataframe
            results_df = pd.DataFrame(
                {
                    "Threshold": threshold,
                    "Margin": margin,
                    "Accuracy": [accuracy_before, accuracy_after],
                    "Statistical Parity Difference": [
                        stat_par_diff_before,
                        stat_par_diff_after,
                    ],
                    "Equal Opportunity Difference": [
                        equal_opp_diff_before,
                        equal_opp_diff_after,
                    ],
                    "Disparate Impact Ratio": [dis_im_ratio_before, dis_im_ratio_after],
                    "Average Odds Difference": [
                        av_odds_diff_before,
                        av_odds_diff_after,
                    ],
                },
                index=["Before Debiasing", "After Debiasing"],
            )
            results.append(results_df)
            # print("-" * 80)
            # print(f"Results for Threshold: {threshold}, Margin: {margin}")
            # print(results_df)
            # print("-" * 80)

    # Combine results into a single DataFrame
    results_df = pd.concat(results)

    # Save the results to a CSV file
    filename = f"roc_results.csv"
    filepath = os.path.join(OUTPUTS_PATH, filename)
    results_df.to_csv(filepath, index=True)

    return results

    """    # Store results in a dataframe (uncomment and unindent for custom ROC threshold and margin)
        results_df = pd.DataFrame({
            'Accuracy': [accuracy_before, accuracy_after],
            'Statistical Parity Difference': [stat_par_diff_before, stat_par_diff_after],
            'Equal Opportunity Difference': [equal_opp_diff_before, equal_opp_diff_after],
            'Disparate Impact Ratio': [dis_im_ratio_before, dis_im_ratio_after],
            'Average Odds Difference': [av_odds_diff_before, av_odds_diff_after]
        }, index=['Before Debiasing', 'After Debiasing'])
        print("-"*80)
        print(results_df)
        print("-"*80)
        return results_df"""


def alldatascores(y_labels, y_preds):
    cm_alldata = confusion_matrix(y_labels, y_preds)
    tn, fp, fn, tp = cm_alldata.ravel()
    accuracy = accuracy_score(y_labels, y_preds)
    precision = precision_score(y_labels, y_preds)
    recall = recall_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds)
    fpr, tpr, thresholds = roc_curve(y_labels, y_preds)
    precision_classwise = precision_score(y_labels, y_preds, average=None)
    recall_classwise = recall_score(y_labels, y_preds, average=None)

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

    return (
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
    )


def leftscores(left_labels, left_preds):
    left_cm = confusion_matrix(left_labels, left_preds)
    ltn, lfp, lfn, ltp = left_cm.ravel()
    left_accuracy = accuracy_score(left_labels, left_preds)
    left_precision = precision_score(left_labels, left_preds)
    left_recall = recall_score(left_labels, left_preds)
    left_f1 = f1_score(left_labels, left_preds)
    left_fpr, left_tpr, left_thresholds = roc_curve(left_labels, left_preds)
    left_precision_classwise = precision_score(left_labels, left_preds, average=None)
    left_recall_classwise = recall_score(left_labels, left_preds, average=None)

    # print("-" * 80)
    # print("Results for left leaning: \n")
    # print("Confusion Matrix:")
    # print(left_cm)

    # print("\n Here are the TP, TN, FP, FN for the Left-leaning group: \n")
    # print(f"True negatives: {ltn}")
    # print(f"False positives: {lfp}")
    # print(f"False negatives: {lfn}")
    # print(f"True positives: {ltp}")

    # print(
    #     "\n Here are the Accuracy, Precision, Recall, and F1-score for the Left-leaning group: \n"
    # )
    # print(f"Accuracy: {left_accuracy:.2f}")
    # print(f"Precision: {left_precision:.2f}")
    # print(f"Recall: {left_recall:.2f}")
    # print(f"F1-score: {left_f1:.2f}")

    # print("\n Here are the TPR and FPR: \n")
    # print(f"TPR: {left_tpr[1]:.2f}")
    # print(f"FPR: {left_fpr[1]:.2f}")

    # print(f"\nPrecision for class 0 (fake): {left_precision_classwise[0]:.2f}")
    # print(f"Precision for class 1 (real): {left_precision_classwise[1]:.2f}")
    # print(f"Recall for class 0 (fake): {left_recall_classwise[0]:.2f}")
    # print(f"Recall for class 1 (real): {left_recall_classwise[1]:.2f}\n")

    return (
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
    )


def rightscores(right_labels, right_preds):
    right_cm = confusion_matrix(right_labels, right_preds)
    rtn, rfp, rfn, rtp = right_cm.ravel()
    right_accuracy = accuracy_score(right_labels, right_preds)
    right_precision = precision_score(right_labels, right_preds)
    right_recall = recall_score(right_labels, right_preds)
    right_f1 = f1_score(right_labels, right_preds)
    right_fpr, right_tpr, right_thresholds = roc_curve(right_labels, right_preds)
    right_precision_classwise = precision_score(right_labels, right_preds, average=None)
    right_recall_classwise = recall_score(right_labels, right_preds, average=None)

    # print("-" * 80)
    # print("Results for right leaning : \n")
    # print("Confusion Matrix:")
    # print(right_cm)

    # print("\n Here are the TP, TN, FP, FN for the Right-leaning group: \n")
    # print(f"True negatives: {rtn}")
    # print(f"False positives: {rfp}")
    # print(f"False negatives: {rfn}")
    # print(f"True positives: {rtp}")

    # print(
    #     "\n Here are the Accuracy, Precision, Recall, and F1-score for the Right-leaning group: \n"
    # )
    # print(f"Accuracy: {right_accuracy:.2f}")
    # print(f"Precision: {right_precision:.2f}")
    # print(f"Recall: {right_recall:.2f}")
    # print(f"F1-score: {right_f1:.2f}")

    # print("\n Here are the TPR and FPR: \n")
    # print(f"TPR: {right_tpr[1]:.2f}")
    # print(f"FPR: {right_fpr[1]:.2f}")

    # print(f"\nPrecision for class 0 (fake): {right_precision_classwise[0]:.2f}")
    # print(f"Precision for class 1 (real): {right_precision_classwise[1]:.2f}")
    # print(f"Recall for class 0 (fake): {right_recall_classwise[0]:.2f}")
    # print(f"Recall for class 1 (real): {right_recall_classwise[1]:.2f}\n")

    return (
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
    )


def fairnessmetrics(y_labels, y_preds, z_leaning, printing=True):
    stat_par_diff = statistical_parity_difference(
        y_labels,
        y_preds,
        prot_attr=z_leaning,
        priv_group=0,
        pos_label=1,
        sample_weight=None,
    )

    equal_opp_diff = equal_opportunity_difference(
        y_labels,
        y_preds,
        prot_attr=z_leaning,
        priv_group=0,
        pos_label=1,
        sample_weight=None,
    )

    dis_im_ratio = disparate_impact_ratio(
        y_labels,
        y_preds,
        prot_attr=z_leaning,
        priv_group=0,
        pos_label=1,
        sample_weight=None,
        zero_division="warn",
    )

    av_odds_diff = average_odds_difference(
        y_labels,
        y_preds,
        prot_attr=z_leaning,
        priv_group=0,
        pos_label=1,
        sample_weight=None,
    )

    if printing:
        print("-" * 80)
        print("Fairness Metrics using IBM's AIF360: \n")
        print(f"Statistical Parity Difference: {stat_par_diff} \n")
        print(f"Equal Opportunity Difference: {equal_opp_diff} \n")
        print(f"Disparate Impact Ratio: {dis_im_ratio} \n")
        print(f"Average Odds Difference: {av_odds_diff} \n")
        print("-" * 80)

    return stat_par_diff, equal_opp_diff, dis_im_ratio, av_odds_diff


# uncomment if you need to calculate fairness metrics without AIF360
def calculate_SPD(left_tpr, left_fpr, right_tpr, right_fpr):
    spd = abs(left_tpr - right_tpr) - abs(left_fpr - right_fpr)
    return spd


def calculate_EOD(left_tpr, left_fpr, right_tpr, right_fpr):
    eod = left_tpr - right_tpr
    return eod


def calculate_DIR(left_tpr, left_fpr, right_tpr, right_fpr):
    diratio = (left_tpr / left_fpr) / (right_tpr / right_fpr)
    return diratio


def calculate_AOD(left_tpr, left_fpr, right_tpr, right_fpr):
    aod = (left_fpr - right_fpr) + (left_tpr - right_tpr)
    aod = aod * 0.5
    return aod


# uncomment if you need to calculate fairness metrics without AIF360
# left_fpr, left_tpr, left_thresholds = roc_curve(left_labels, left_preds)
# right_fpr, right_tpr, right_thresholds = roc_curve(right_labels, right_preds)

# spd = calculate_SPD(left_tpr, left_fpr, right_tpr, right_fpr)
# eod = calculate_EOD(left_tpr, left_fpr, right_tpr, right_fpr)
# diratio = calculate_DIR(left_tpr, left_fpr, right_tpr, right_fpr)
# aod = calculate_AOD(left_tpr, left_fpr, right_tpr, right_fpr)

# print("-" * 80)
# print("Fairness Metrics calculated using formulae:")
# print("Statistical Parity Difference:", spd)
# print("The Equal Opportunity Difference (EOD) is:", eod)
# print("The Disparate Impact Ratio (DIR) is:", diratio)
# print("The Average Odds Difference (AOD) is:", aod)


# # IMPORTANT FOR DEBUGGIN debias.py : DONT DELETE # #

# # z_train = train_probs.index.get_level_values('z').copy()
# # train_probs['z'] = z_train
# # prot_attr = 'z'
# # X = train_probs
# # assert prot_attr in X.index.names
# # a = X.index.to_frame()
# # groups = a.set_index(prot_attr).index

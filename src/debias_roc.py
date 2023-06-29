import os
import numpy as np
from utils import fairnessmetrics, debiasing_results

TRAIN_PROB_LABELS_PATH = os.path.join(
    os.path.dirname(os.getcwd()), "outputs", "train_probs_labels_all_samples.csv"
)
TEST_PROB_LABELS_PATH = os.path.join(
    os.path.dirname(os.getcwd()), "outputs", "test_probs_labels_all_samples.csv"
)
VALID_PROB_LABELS_PATH = os.path.join(
    os.path.dirname(os.getcwd()), "outputs", "valid_probs_labels_all_samples.csv"
)

np.random.seed(42)

if __name__ == "__main__":
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    experiments = {}
    for threshold in thresholds:
        margin_list = []
        if threshold == 0.1 or threshold == 0.9:
            margin_list.append(0.05)
        else:
            for margin in range(1, 10):
                margin_decimal = margin * 0.1 ** len(str(margin))
                if 0 < margin_decimal < min(threshold, 1 - threshold):
                    margin_list.append(round(margin_decimal, 1))
        experiments[threshold] = margin_list
    print(experiments)

    debiasing_results(
        prob_labels_path=TEST_PROB_LABELS_PATH, mode="test", experiments=experiments
    )

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bert_model import BertClassifier
from data_load import NELADataProcessor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
NUM_SAMPLES = -1
NAME = "distilbert-base-uncased"
MODE = "fine-tune"  # pre-train or fine-tune
modelname = "19-03-23-03_26_final_model_fine-tune.pt"
MODELS_PATH = os.path.join(os.path.dirname(os.getcwd()), "models", modelname)
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs")
SPLIT = "train"  # "train" or "test" or "valid"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BertClassifier(mode=MODE)
checkpoint = torch.load(MODELS_PATH)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# load example data to use for analysis
data_processor = NELADataProcessor(
    name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
)
dataloaders = data_processor.process(num_samples=NUM_SAMPLES)

# Get probabilities for the training data and store in a dataframe
probs = []
true_labels = []
z_vals = []
model.eval()
with torch.no_grad():
    for data in dataloaders[SPLIT]:
        input_ids, attention_mask, labels, leaning = data
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs_batch = torch.softmax(outputs, dim=1)
        probs.extend(probs_batch.tolist())
        true_labels.extend(labels.tolist())
        z_vals.extend(leaning.tolist())

probs_df = pd.DataFrame(probs, columns=["class1_prob", "class2_prob"])
probs_df["true_label"] = true_labels
probs_df["z"] = z_vals
print(probs_df.head())

# Export the dataframe to a CSV file
if NUM_SAMPLES == -1:
    sample_size = "all"
else:
    sample_size = NUM_SAMPLES
probs_df.to_csv(
    os.path.join(OUTPUTS_PATH, f"{SPLIT}_probs_labels_{sample_size}_samples.csv"),
    index=False,
)

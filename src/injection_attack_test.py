import os
import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoTokenizer
from bert_model import BertClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Define constants
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data")
TEST_DATA_PATH = os.path.join(DATA_PATH, "random_inject_test_data.csv")
MODELS_PATH = os.path.join(os.path.dirname(os.getcwd()), "models")
MODEL_PATH = os.path.join(MODELS_PATH, "26-04-23-03_15_final_model_fine-tune.pt")
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
NAME = "distilbert-base-uncased"
MODE = "fine-tune"  # Set the mode for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the attack test data
df = pd.read_csv(TEST_DATA_PATH)

# Filter out rows with non-string values in the "content" column
df = df[df["content"].apply(lambda x: isinstance(x, str))]

# Tokenize the attack test data
tokenizer = AutoTokenizer.from_pretrained(NAME)
encoded = tokenizer.batch_encode_plus(
    df["content"].tolist(),
    max_length=MAX_SEQUENCE_LENGTH,
    truncation=True,
    padding="max_length"
)
input_ids = torch.tensor(encoded["input_ids"]).to(device)
attention_mask = torch.tensor(encoded["attention_mask"]).to(device)



# Create a DataLoader for the attack test data
attack_test_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
attack_test_dataloader = torch.utils.data.DataLoader(
    attack_test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# Load the model
model = BertClassifier(mode=MODE)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Use the model to make predictions on the attack test data
predicted_labels = []
with torch.no_grad():
    for batch in attack_test_dataloader:
        input_ids, attention_mask = batch
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        predicted_labels.extend(preds.cpu().numpy())

# Add the predicted labels to the DataFrame as a new column
df["predicted_label"] = predicted_labels

# Save the modified DataFrame to a new CSV file
OUTPUT_PATH = os.path.join(DATA_PATH, "modified_test_data_with_predictions.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("Predictions saved to", OUTPUT_PATH)

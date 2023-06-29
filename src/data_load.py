import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "finaldata.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "data")
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
NAME = "distilbert-base-uncased"  # "roberta-large" #"distilbert-base-uncased"
NUM_SAMPLES = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NELADataset:
    # set num_samples = -1 to run on all data
    def __init__(self, path=DATA_PATH, num_samples=NUM_SAMPLES):
        self.path = path
        self.num_samples = num_samples
        self.df = pd.read_csv(path)
        self.df = self.df[["source", "name", "content", "overall_class", "leaning"]]
        self.df["source"] = self.df["source"].map(str)
        self.df["name"] = self.df["name"].map(str)
        self.df["content"] = self.df["content"].map(str)
        self.df["leaning"] = self.df["leaning"].map({"left": 0, "right": 1})

        if num_samples != -1:
            self.df = self.df[:num_samples]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "source": sample["source"],
            "name": sample["name"],
            "content": sample["content"],
            "overall_class": sample["overall_class"],
            "leaning": sample["leaning"],
        }


class NELADataProcessor:
    def __init__(
        self, name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def tokenize(self, sentences, padding="max_length"):
        encoded = self.tokenizer.batch_encode_plus(
            sentences, max_length=self.max_seq_length, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(
            device
        )

    def process(self, data_path=DATA_PATH, num_samples=NUM_SAMPLES):
        self.dataset = NELADataset(data_path, num_samples)
        (
            X_train_val,
            X_test,
            y_train_val,
            y_test,
            z_train_val,
            z_test,
        ) = train_test_split(
            self.dataset.df["content"].tolist(),
            self.dataset.df["overall_class"].tolist(),
            self.dataset.df["leaning"].tolist(),
            test_size=0.15,
            shuffle=True,
        )
        X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(
            X_train_val, y_train_val, z_train_val, test_size=0.15, shuffle=True
        )

        train_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_train),
            torch.tensor(y_train).to(device),
            torch.tensor(z_train).to(device)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_val),
            torch.tensor(y_val).to(device),
            torch.tensor(z_val).to(device)
        )
        test_dataset = torch.utils.data.TensorDataset(
            *self.tokenize(X_test),
            torch.tensor(y_test).to(device),
            torch.tensor(z_test).to(device)
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.dataloaders = {
            "train": train_dataloader,
            "valid": valid_dataloader,
            "test": test_dataloader,
        }

        return self.dataloaders


if __name__ == "__main__":
    data_processor = NELADataProcessor()
    dataloaders = data_processor.process()

    # store training data
    train_dataset = dataloaders["train"].dataset
    X_train, attention_mask_train, y_train, z_train = train_dataset[:]
    train_data = pd.DataFrame(
        {
            "content": data_processor.tokenizer.batch_decode(
                X_train, skip_special_tokens=True
            ),
            "overall_class": y_train.cpu().numpy(),
            "leaning": z_train.cpu().numpy(),
        }
    )
    save_path = os.path.join(OUTPUT_PATH, "training_data.csv")
    train_data.to_csv(save_path, index=False)

    # store validation data
    valid_dataset = dataloaders["valid"].dataset
    X_valid, attention_mask_valid, y_valid, z_valid = valid_dataset[:]
    valid_data = pd.DataFrame(
        {
            "content": data_processor.tokenizer.batch_decode(
                X_valid, skip_special_tokens=True
            ),
            "overall_class": y_valid.cpu().numpy(),
            "leaning": z_valid.cpu().numpy(),
        }
    )
    save_path = os.path.join(OUTPUT_PATH, "validation_data.csv")
    valid_data.to_csv(save_path, index=False)

    # store testing data
    test_dataset = dataloaders["test"].dataset
    X_test, attention_mask_test, y_test, z_test = test_dataset[:]
    test_data = pd.DataFrame(
        {
            "content": data_processor.tokenizer.batch_decode(
                X_test, skip_special_tokens=True
            ),
            "overall_class": y_test.cpu().numpy(),
            "leaning": z_test.cpu().numpy(),
        }
    )
    save_path = os.path.join(OUTPUT_PATH, "testing_data.csv")
    test_data.to_csv(save_path, index=False)

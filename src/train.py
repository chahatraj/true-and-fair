from data_load import NELADataProcessor
from bert_model import BertClassifier

import os
import random
import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

SEED = 42
NUM_EPOCHS = 4
LEARNING_RATE = 5e-5  # 5e-6
EPS = 1e-8
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
NAME = "distilbert-base-uncased"  # "roberta-large" #"distilbert-base-uncased"
NUM_SAMPLES = (
    -1
)  # -1 will run on entire data, otherwise specify a certain number of data points like 1000
MODE = "fine-tune"  # "pre-train"  # pre-train or fine-tune
# CKPT = "05-03-23-13_58_final_model_pretrain.pt"
# "05-03-23-09_31_pretrain_model_epoch63.pt" -> overfits in 15 epochs
# "05-03-23-01_19_final_model_pretrain.pt" -> 84.99
# MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), "models", CKPT)

MODELS_PATH = os.path.join(os.path.dirname(os.getcwd()), "models")
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs")
LOGS_PATH = os.path.join(os.getcwd(), "logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    models_path,
):
    writer = SummaryWriter(log_dir=LOGS_PATH)
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        model.train()
        for i, data in enumerate(train_dataloader):
            # print("Batch {i}".format(i=i + 1))
            input_ids, attention_mask, labels, leaning = data
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == labels).sum().item()
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader.dataset)
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Train Accuracy", train_acc, epoch)
        print("-" * 50)
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(valid_dataloader):
                # print("Val Epoch {num}: ".format(num=epoch + 1))
                input_ids, attention_mask, labels, leaning = data
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_acc += (predicted == labels).sum().item()
        valid_loss /= len(valid_dataloader)
        valid_acc /= len(valid_dataloader.dataset)
        writer.add_scalar("Validation Loss", valid_loss, epoch)
        writer.add_scalar("Validation Accuracy", valid_acc, epoch)
        print("-" * 50)
        print(
            f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}"
        )
        filename = datetime.now().strftime(
            f"%d-%m-%y-%H_%M_{MODE}_model_epoch{epoch}.pt"
        )
        torch.save(model.state_dict(), os.path.join(models_path, filename))

    filename = datetime.now().strftime(f"%d-%m-%y-%H_%M_final_model_{MODE}.pt")
    torch.save(model.state_dict(), os.path.join(models_path, filename))
    writer.close()


def evaluate_model(model, test_dataloader):
    model.eval()
    test_acc = 0
    batch_count = 0
    all_labels = []
    all_preds = []
    all_leaning = []
    for i, data in enumerate(test_dataloader):
        input_ids, attention_mask, labels, leaning = data
        all_leaning.append(leaning.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            test_acc += (preds == labels).sum().item()
            batch_count += 1

    test_acc /= batch_count * test_dataloader.batch_size
    print(f"Test Accuracy: {test_acc} \n")
    return all_labels, all_preds, all_leaning


def save_test_as_dataframe(all_labels, all_preds, all_leaning):
    labels_df = pd.DataFrame(
        {
            "true_labels": [label for batch in all_labels for label in batch],
            "predicted_labels": [pred for batch in all_preds for pred in batch],
            "leaning": [leaning for batch in all_leaning for leaning in batch],
        }
    )
    print(labels_df.head())
    if NUM_SAMPLES == -1:
        sample_size = "all"
    else:
        sample_size = NUM_SAMPLES
    filename = f"apr_24_{MODE}_test_labels_{sample_size}_samples.csv"
    labels_df.to_csv(os.path.join(OUTPUTS_PATH, filename), index=False)


if __name__ == "__main__":
    data_processor = NELADataProcessor(
        name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    )
    dataloaders = data_processor.process(num_samples=NUM_SAMPLES)
    # print(len(dataloaders["train"].dataset))
    # print(len(dataloaders["valid"].dataset))
    # print(len(dataloaders["test"].dataset))
    print("Data Load done")

    model = BertClassifier(name=NAME, mode=MODE, pretrained_checkpoint=None)
    model = model.to(device)
    print("Model Load done")
    # print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPS)
    total_steps = len(dataloaders["train"]) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training/Validation
    print("Starting train/val loop now :p \n")
    train_model(
        model,
        dataloaders["train"],
        dataloaders["valid"],
        criterion,
        optimizer,
        scheduler,
        NUM_EPOCHS,
        MODELS_PATH,
    )
    print("Training Complete :p \n")

    # Evaluation
    print("Model Evaluation :p \n")
    all_labels, all_preds, all_leaning = evaluate_model(model, dataloaders["test"])
    save_test_as_dataframe(all_labels, all_preds, all_leaning)

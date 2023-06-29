import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bert_model import BertClassifier
from data_load import NELADataProcessor

import seaborn as sns
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
NUM_SAMPLES = 1000
NAME = "distilbert-base-uncased"
MODE = "fine-tune"  # pre-train or fine-tune
modelname = "23-02-23-02_29_final_model.pt"
MODELS_PATH = os.path.join(os.path.dirname(os.getcwd()), "models", modelname)
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # load trained model for analysis
    model = BertClassifier(mode=MODE)
    checkpoint = torch.load(MODELS_PATH)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    model.zero_grad()

    # load example data to use for analysis
    data_processor = NELADataProcessor(
        name=NAME, max_seq_length=MAX_SEQUENCE_LENGTH, batch_size=BATCH_SIZE
    )
    dataloaders = data_processor.process(num_samples=NUM_SAMPLES)
    tokenizer = data_processor.tokenizer
    data = data_processor.dataset
    example_sentence = "New covid wave spreads panic in Texas."  # data[0]["content"]
    example_target = 0  # data[0]["overall_class"]
    print("Example sentence     : ", example_sentence)
    print("Corresponding target : ", example_target)
    example_target = torch.tensor([example_target]).to(device)

    # Use captum instead of ecco for MLM BERT (mentioned by creator of ecco)
    # https://github.com/jalammar/ecco/issues/31#issuecomment-777578887

    # Follow this tutorial : https://captum.ai/tutorials/Bert_SQUAD_Interpret
    # (model, tokenizer) have already been loaded above. follow the rest of the tutorial
    # and match it to the requirements for this particular dataset which is different from
    # the Question Answering dataset they have in the tutorial.

    # Create an instance of the LayerIntegratedGradients class
    lig = LayerIntegratedGradients(model, model.bert.embeddings)

    # Tokenize the example sentence
    inputs = tokenizer.encode_plus(
        example_sentence,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Compute attributions
    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),
        additional_forward_args=(attention_mask,),
        target=example_target,
        internal_batch_size=BATCH_SIZE,
        return_convergence_delta=True,
    )
    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    attributions_norm = attributions_sum / torch.norm(attributions_sum)

    # Generate heatmap visualization of attributions
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    heatmap = attributions_norm.squeeze().tolist()
    plt.figure(figsize=(16, 8))
    sns.set(font_scale=1.5)
    sns.heatmap([heatmap], annot=[tokens], fmt="", cmap="viridis")
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks([])
    plt.title("BERT Input Attribution Heatmap")
    plt.savefig(
        os.path.join(OUTPUTS_PATH, "example_heatmap.png"), dpi=300, bbox_inches="tight"
    )

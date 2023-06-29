import os
import random
import pandas as pd
import numpy as np
from utils import *
import torch
import shap
import json
import nltk
import re
from nltk.corpus import stopwords
from bert_model import BertClassifier
from transformers import AutoTokenizer
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

nltk.download("punkt")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

filename = "left_false_negatives.csv"
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "outputs/shap_phrase_outputs")
TEST_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs/shap_inputs", filename)
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs", "imgs")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tokenize(tokenizer, sentences, padding="max_length"):
    encoded = tokenizer.batch_encode_plus(
        sentences, max_length=512, truncation=True, padding=padding
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


def get_model_output(sentences):
    sentences = list(sentences)
    input_ids, attention_mask = tokenize(tokenizer, sentences)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probabilities = torch.softmax(output, dim=-1)
    return probabilities.cpu().numpy()


def preprocess_text(text):
    text = re.sub(r"#", "", text)
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    text = " ".join(tokens)
    return text


def get_k_most_important_phrases(text, output_class, k):
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    tokenized_sentences = [
        nltk.word_tokenize(sentence) for sentence in preprocessed_sentences
    ]

    phrases = Phrases(tokenized_sentences, min_count=2, threshold=10)
    phraser = Phraser(phrases)
    phrase_sentences = [" ".join(phraser[sentence]) for sentence in tokenized_sentences]

    explainer = shap.Explainer(get_model_output, shap.maskers.Text(tokenizer))
    shap_values = explainer(phrase_sentences)
    importance_values = shap_values[:, :, output_class].values

    phrase_importance = []
    for i, phrase in enumerate(phrase_sentences):
        phrase_mean_importance = np.mean(importance_values[i])
        phrase_importance.append((phrase, phrase_mean_importance))

    phrase_importance.sort(key=lambda x: x[1], reverse=True)

    k_most_important_phrases = phrase_importance[:k]

    return k_most_important_phrases


if __name__ == "__main__":
    model = BertClassifier(mode="fine-tune")
    model.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.getcwd()),
                "models",
                "19-03-23-03_26_final_model_fine-tune.pt",
            ),
            map_location=device,
        )
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    stop_words = set(stopwords.words("english"))
    df = pd.read_csv(TEST_PATH)
    print(df.head())
    df = df.dropna(subset=["sentence"])
    input_texts = df["sentence"].tolist()
    # input_texts = input_texts[:1]

    output_class = 1
    k = 5
    most_important_phrases = []  # Corrected variable name
    for text in input_texts:
        k_most_important = get_k_most_important_phrases(text, output_class, k)
        most_important_phrases.append(k_most_important)
        print(
            f"The {k} most important phrases for output class {output_class} are:",
            k_most_important,
        )

    filename = f"{k}_shap_phrases_lfn.json"
    filepath = os.path.join(OUTPUTS, filename)
    with open(filepath, "w") as f:
        json.dump(most_important_phrases, f, indent=4)

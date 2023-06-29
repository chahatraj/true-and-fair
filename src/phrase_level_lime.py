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
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from multiprocessing import Pool

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


def tokenize(tokenizer, sentences, padding="max_length", batch_size=16):
    all_input_ids = []
    all_attention_mask = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        encoded = tokenizer.batch_encode_plus(
            batch_sentences, max_length=512, truncation=True, padding=padding
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        all_input_ids.extend(input_ids)
        all_attention_mask.extend(attention_mask)
    return torch.tensor(all_input_ids).to(device), torch.tensor(all_attention_mask).to(
        device
    )


def get_model_output(sentences, batch_size=16):
    all_probabilities = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        input_ids, attention_mask = tokenize(
            tokenizer, batch_sentences, batch_size=batch_size
        )
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probabilities = torch.softmax(output, dim=-1)
        all_probabilities.extend(probabilities.cpu().numpy())
    return all_probabilities


def get_lime_explanation(
    text, model, tokenizer, num_features=5, num_samples=100, batch_size=16
):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    exp = explainer.explain_instance(
        text,
        lambda x: get_model_output(x, batch_size=batch_size),
        num_features=num_features,
        num_samples=num_samples,
    )
    return exp.as_list()


def get_lime_explanations_parallel(
    phrase_sentences, model, tokenizer, num_features, output_class, k
):
    with Pool() as pool:
        lime_explanations = pool.starmap(
            get_lime_explanation,
            [
                (phrase_sentence, model, tokenizer, num_features)
                for phrase_sentence in phrase_sentences
            ],
        )
    return lime_explanations


def preprocess_text(text):
    text = re.sub(r"#", "", text)
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    text = " ".join(tokens)
    return text


def get_k_most_important_phrases_lime(text, output_class, k, batch_size=16):
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    tokenized_sentences = [
        nltk.word_tokenize(sentence) for sentence in preprocessed_sentences
    ]

    phrases = Phrases(
        tokenized_sentences, min_count=2, threshold=10
    )  # Increase the threshold for meaningful phrases
    phraser = Phraser(phrases)
    phrase_sentences = [" ".join(phraser[sentence]) for sentence in tokenized_sentences]

    lime_explanations = [
        get_lime_explanation(
            phrase_sentence, model, tokenizer, num_features=k, batch_size=batch_size
        )
        for phrase_sentence in phrase_sentences
    ]

    phrase_importance = []
    for i, explanation in enumerate(lime_explanations):
        importance_sum = sum([abs(feature[1]) for feature in explanation])
        phrase_importance.append((phrase_sentences[i], importance_sum))

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
    input_texts = input_texts[:1]

    output_class = 1
    k = 5
    most_important_phrases = []  # Corrected variable name
    for text in tqdm(input_texts):
        # Get LIME explanations
        lime_explanation = get_k_most_important_phrases_lime(text, output_class, k)
        most_important_phrases.append(lime_explanation)
        print(
            f"The {k} most important phrases for output class {output_class} using LIME are:",
            lime_explanation,
        )

    filename = f"{k}_lime_phrases_lfn.json"
    filepath = os.path.join(OUTPUTS, filename)
    with open(filepath, "w") as f:
        json.dump(most_important_phrases, f, indent=4)

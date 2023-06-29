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
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords
from bert_model import BertClassifier
from transformers import AutoTokenizer
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

nltk.download("punkt")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

filename = "real_val.csv"  # change files from output to run on all misclassified items
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "outputs/shap_outputs")
TEST_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", filename)
OUTPUTS_PATH = os.path.join(os.path.dirname(os.getcwd()), "outputs", "imgs")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stop_words = set(stopwords.words("english"))


def tokenize(tokenizer, sentences, padding="max_length"):
    encoded = tokenizer.batch_encode_plus(
        sentences, max_length=512, truncation=True, padding=padding
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


def get_model_output(sentences, tokenizer, batch_size=16):
    sentences = list(sentences)
    n = len(sentences)
    num_batches = (n + batch_size - 1) // batch_size
    probabilities_all = []

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, n)
        batch_sentences = sentences[batch_start:batch_end]

        input_ids, attention_mask = tokenize(tokenizer, batch_sentences)
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probabilities = torch.softmax(output, dim=-1)
            probabilities_all.extend(probabilities.cpu().numpy())

    return np.array(probabilities_all)


def preprocess_text(text):
    text = re.sub(r"#", "", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    text = " ".join(tokens)
    return text


def preprocess_text_parallel(sentences):
    max_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        preprocessed_sentences = list(
            tqdm(executor.map(preprocess_text, sentences), total=len(sentences))
        )
    return preprocessed_sentences


# def get_k_most_important_words_helper(
#     idx, sentence, output_class, k, tokenizer, batch_size=8
# ):
#     tokenized_sentence = tokenizer.tokenize(sentence)

#     input_ids, attention_mask = tokenize(tokenizer, [sentence])
#     explainer = shap.Explainer(
#         lambda x: get_model_output(x, tokenizer, batch_size=batch_size),
#         shap.maskers.Text(tokenizer),
#     )
#     shap_values = explainer([sentence])
#     importance_values = shap_values[:, :, output_class].values
#     token_importance = list(zip(tokenized_sentence, importance_values[0]))

#     # Filter out words beginning with "##" as these come from sub-words due to tokenization
#     token_importance = [
#         (token, float(score))
#         for token, score in token_importance
#         if not token.startswith("##")
#     ]

#     positive_token_importance = [item for item in token_importance if item[1] > 0]
#     negative_token_importance = [item for item in token_importance if item[1] < 0]

#     positive_token_importance.sort(key=lambda x: x[1], reverse=True)
#     k_most_positive_important_words = positive_token_importance[
#         : min(k, len(positive_token_importance))
#     ]

#     negative_token_importance.sort(key=lambda x: x[1])
#     k_most_negative_important_words = negative_token_importance[
#         : min(k, len(negative_token_importance))
#     ]

#     return idx, {
#         "sentence": sentence,
#         "positive_important_words": k_most_positive_important_words,
#         "negative_important_words": k_most_negative_important_words,
#     }

def get_k_most_important_words_helper(
    idx, sentence, output_class, k, tokenizer, batch_size=8
):
    tokenized_sentence = tokenizer.tokenize(sentence)

    # Perform NER and create a mapping from named entities to their tokens
    entity_to_tokens = {}
    doc = nlp(sentence)
    for ent in doc.ents:
        entity_to_tokens[ent.text] = ent.text.split()

    input_ids, attention_mask = tokenize(tokenizer, [sentence])
    explainer = shap.Explainer(
        lambda x: get_model_output(x, tokenizer, batch_size=batch_size),
        shap.maskers.Text(tokenizer),
    )
    shap_values = explainer([sentence])
    importance_values = shap_values[:, :, output_class].values
    token_importance = list(zip(tokenized_sentence, importance_values[0]))

    # Aggregate salience scores for named entities
    aggregated_token_importance = []
    for token, score in token_importance:
        if token.startswith("##"):
            continue
        for entity, tokens in entity_to_tokens.items():
            if token in tokens:
                # Aggregate scores for tokens belonging to the same entity
                aggregated_score = sum(
                    score for t, score in token_importance if t in tokens
                )
                average_score = aggregated_score / len(tokens)
                aggregated_token_importance.append((entity, average_score))
                break
        else:
            aggregated_token_importance.append((token, score))

    # Remove duplicates
    aggregated_token_importance = list(set(aggregated_token_importance))

    positive_token_importance = [item for item in aggregated_token_importance if item[1] > 0]
    negative_token_importance = [item for item in aggregated_token_importance if item[1] < 0]

    positive_token_importance.sort(key=lambda x: x[1], reverse=True)
    k_most_positive_important_words = positive_token_importance[
        : min(k, len(positive_token_importance))
    ]

    negative_token_importance.sort(key=lambda x: x[1])
    k_most_negative_important_words = negative_token_importance[
        : min(k, len(negative_token_importance))
    ]

    return idx, {
        "sentence": sentence,
        "positive_important_words": k_most_positive_important_words,
        "negative_important_words": k_most_negative_important_words,
    }


def get_k_most_important_words(sentences, output_class, k):
    result = {}

    max_workers = os.cpu_count()

    # Filter out sentences with "forbidden nginx"
    filtered_sentences = [
        (idx, sentence)
        for idx, sentence in enumerate(sentences)
        if "forbidden nginx" not in sentence.lower()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                get_k_most_important_words_helper,
                idx,
                sentence,
                output_class,
                k,
                AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True),
                batch_size=8,
            )
            for idx, sentence in filtered_sentences
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(filtered_sentences)
        ):
            idx, res = future.result()
            result[idx] = res
    return result


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
    model.eval()
    model.to(device)

    df = pd.read_csv(TEST_PATH)
    print(df.head())
    df = df.dropna(subset=["content"])
    sentences = df["content"].tolist()
    sentences = preprocess_text_parallel(sentences)
    # sentences = sentences[:100]

    output_class = 1
    k = 10
    k_most_important_words = get_k_most_important_words(
        sentences, output_class, k
    )
    print(
        f"The {k} most important words for output class {output_class} are:",
        k_most_important_words,
    )
    out_filename = f"May3_{k}_token_shap_real_val.json"  # change
    filepath = os.path.join(OUTPUTS, out_filename)
    with open(filepath, "w") as f:
        json.dump(k_most_important_words, f, indent=4)

# sentences = [preprocess_text(sentence) for sentence in sentences]
# visualize_random_sentences(sentences, output_class, num_sentences=10)

# def visualize_random_sentences(sentences, output_class, num_sentences=5):
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
#     selected_indices = random.sample(range(len(sentences)), num_sentences)
#     selected_sentences = [sentences[i] for i in selected_indices]

#     explainer = shap.Explainer(
#         lambda x: get_model_output(x, tokenizer, batch_size), shap.maskers.Text(tokenizer)
#     )
#     shap_values = explainer(selected_sentences)

#     for idx, sentence, shap_value in zip(
#         selected_indices, selected_sentences, shap_values
#     ):
#         html_str = shap.plots.text(shap_value[:, output_class], display=False)
#         filename = f"shap_plot_sentence_{idx}.html"
#         filepath = os.path.join(OUTPUTS_PATH, filename)
#         with open(filepath, "w") as f:
#             f.write(html_str)

# def get_model_output(sentences):
#     sentences = list(sentences)
#     input_ids, attention_mask = tokenize(tokenizer, sentences)
#     with torch.no_grad():
#         output = model(input_ids, attention_mask)
#         probabilities = torch.softmax(output, dim=-1)
#     return probabilities.cpu().numpy()

"""
def get_k_most_important_words(sentences, output_class, k):
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    input_ids, attention_mask = tokenize(tokenizer, sentences)
    explainer = shap.Explainer(get_model_output, shap.maskers.Text(tokenizer))
    shap_values = explainer(sentences)
    importance_values = shap_values[:, :, output_class].values
    token_importance = list(zip(tokenized_sentences[0], importance_values[0]))
    token_importance.sort(key=lambda x: x[1], reverse=True)
    k_most_important_words = {token: importance for token, importance in token_importance[:k]}
    return k_most_important_words
"""

# def get_k_most_important_words(sentences, output_class, k):
#     result = {}

#     for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
#         # Preprocess the sentence
#         sentence = preprocess_text(sentence)
#         tokenized_sentence = tokenizer.tokenize(sentence)

#         input_ids, attention_mask = tokenize(tokenizer, [sentence])
#         explainer = shap.Explainer(get_model_output, shap.maskers.Text(tokenizer))
#         shap_values = explainer([sentence])
#         importance_values = shap_values[:, :, output_class].values
#         token_importance = list(zip(tokenized_sentence, importance_values[0]))
#         token_importance.sort(key=lambda x: x[1], reverse=True)

#         k_most_important_words = token_importance[:k]
#         result[idx] = {
#             "sentence": sentence,
#             "important_words": [(token, float(score)) for token, score in k_most_important_words]
#         }

#     return result

# def get_k_most_important_words(sentences, output_class, k):
#     result = {}

#     for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
#         # Preprocess the sentence
#         sentence = preprocess_text(sentence)
#         tokenized_sentence = tokenizer.tokenize(sentence)

#         input_ids, attention_mask = tokenize(tokenizer, [sentence])
#         explainer = shap.Explainer(get_model_output, shap.maskers.Text(tokenizer))
#         shap_values = explainer([sentence])
#         importance_values = shap_values[:, :, output_class].values
#         token_importance = list(zip(tokenized_sentence, importance_values[0]))

#         positive_token_importance = [item for item in token_importance if item[1] > 0]
#         negative_token_importance = [item for item in token_importance if item[1] < 0]

#         positive_token_importance.sort(key=lambda x: x[1], reverse=True)
#         k_most_positive_important_words = positive_token_importance[:min(k, len(positive_token_importance))]

#         negative_token_importance.sort(key=lambda x: x[1])
#         k_most_negative_important_words = negative_token_importance[:min(k, len(negative_token_importance))]

#         result[idx] = {
#             "sentence": sentence,
#             "positive_important_words": [(token, float(score)) for token, score in k_most_positive_important_words],
#             "negative_important_words": [(token, float(score)) for token, score in k_most_negative_important_words],
#         }

#     return result

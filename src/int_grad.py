import os
import random
import pandas as pd
import numpy as np
import torch
import json
import nltk
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords
from bert_model import BertClassifier
from transformers import AutoTokenizer
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

nltk.download("punkt")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

filename = "real_val.csv"  # change files from output to run on all misclassified items
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "outputs/intgrad_outputs")
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


def get_model_output(sentences, tokenizer):
    sentences = list(sentences)
    input_ids, attention_mask = tokenize(tokenizer, sentences)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probabilities = torch.softmax(output, dim=-1)
    return probabilities.cpu().numpy()


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

# def get_k_most_important_words(sentences, output_class, k, target, tokenizer):
#     result = {}

#     lig = LayerIntegratedGradients(model, model.bert.embeddings)

#     for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
#         tokenized_sentence = tokenizer.tokenize(sentence)

#         input_ids, attention_mask = tokenize(tokenizer, [sentence])
#         model_output = get_model_output([sentence], tokenizer)[0]

#         # Change this block
#         baseline = input_ids * 0
#         baseline[0][0] = 101  # [CLS] token
#         baseline[0][-1] = 102  # [SEP] token
#         attributions, delta = lig.attribute(
#             input_ids,
#             additional_forward_args=attention_mask,
#             baselines=baseline,
#             return_convergence_delta=True,
#             n_steps=50,
#             target=target,  # Add the target parameter here
#         )
#         importance_values = attributions.sum(dim=-1).squeeze(0)
#         token_importance = list(zip(tokenized_sentence, importance_values.tolist()))

#         # Filter out words beginning with "##" as these come from sub-words due to tokenization
#         token_importance = [
#             (token, float(score))
#             for token, score in token_importance
#             if not token.startswith("##")
#         ]

#         positive_token_importance = [item for item in token_importance if item[1] > 0]
#         negative_token_importance = [item for item in token_importance if item[1] < 0]

#         positive_token_importance.sort(key=lambda x: x[1], reverse=True)
#         k_most_positive_important_words = positive_token_importance[
#             : min(k, len(positive_token_importance))
#         ]

#         negative_token_importance.sort(key=lambda x: x[1])
#         k_most_negative_important_words = negative_token_importance[
#             : min(k, len(negative_token_importance))
#         ]

#         result[idx] = {
#             "sentence": sentence,
#             "positive_important_words": [
#                 (token, float(score))
#                 for token, score in k_most_positive_important_words
#             ],
#             "negative_important_words": [
#                 (token, float(score))
#                 for token, score in k_most_negative_important_words
#             ],
#         }

#     return result


def get_k_most_important_words(sentences, output_class, k, target, tokenizer):
    result = {}

    lig = LayerIntegratedGradients(model, model.bert.embeddings)

    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        tokenized_sentence = tokenizer.tokenize(sentence)

        # Perform NER and create a mapping from named entities to their tokens
        entity_to_tokens = {}
        doc = nlp(sentence)
        for ent in doc.ents:
            entity_to_tokens[ent.text] = ent.text.split()

        input_ids, attention_mask = tokenize(tokenizer, [sentence])
        model_output = get_model_output([sentence], tokenizer)[0]

        # Change this block
        baseline = input_ids * 0
        baseline[0][0] = 101  # [CLS] token
        baseline[0][-1] = 102  # [SEP] token
        attributions, delta = lig.attribute(
            input_ids,
            additional_forward_args=attention_mask,
            baselines=baseline,
            return_convergence_delta=True,
            n_steps=50,
            target=target,  # Add the target parameter here
        )
        importance_values = attributions.sum(dim=-1).squeeze(0)
        token_importance = list(zip(tokenized_sentence, importance_values.tolist()))

        # Aggregate importance scores for named entities
        aggregated_importance = []
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
                    aggregated_importance.append((entity, average_score))
                    break
            else:
                aggregated_importance.append((token, score))

        # Remove duplicates
        aggregated_importance = list(set(aggregated_importance))

        positive_importance = [item for item in aggregated_importance if item[1] > 0]
        negative_importance = [item for item in aggregated_importance if item[1] < 0]

        positive_importance.sort(key=lambda x: x[1], reverse=True)
        k_most_positive_important_words = positive_importance[
            : min(k, len(positive_importance))
        ]

        negative_importance.sort(key=lambda x: x[1])
        k_most_negative_important_words = negative_importance[
            : min(k, len(negative_importance))
        ]

        result[idx] = {
            "sentence": sentence,
            "positive_important_words": [
                (token, float(score))
                for token, score in k_most_positive_important_words
            ],
            "negative_important_words": [
                (token, float(score))
                for token, score in k_most_negative_important_words
            ],
        }

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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    df = pd.read_csv(TEST_PATH)
    print(df.head())
    df = df.dropna(subset=["content"])
    sentences = df["content"].tolist()
    sentences = preprocess_text_parallel(sentences)
    # sentences = sentences[:20]
    sentences = [
        sentence
        for sentence in sentences
        if "forbidden nginx" not in sentence.lower()
    ]

    output_class = 1
    k = 10
    k_most_important_words = get_k_most_important_words(
        sentences, output_class, k, output_class, tokenizer
    )
    print(
        f"The {k} most important words for output class {output_class} are:",
        k_most_important_words,
    )
    filename = f"May3_{k}_token_intgrad_real_val.json"  # change
    filepath = os.path.join(OUTPUTS, filename)
    with open(filepath, "w") as f:
        json.dump(k_most_important_words, f, indent=4)

# sentences = [preprocess_text(sentence) for sentence in sentences]
# visualize_random_sentences(sentences, output_class, num_sentences=10)

# def visualize_random_sentences(sentences, output_class, num_sentences=5):
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
#     selected_indices = random.sample(range(len(sentences)), num_sentences)
#     selected_sentences = [sentences[i] for i in selected_indices]

#     for idx, sentence in zip(selected_indices, selected_sentences):
#         # Preprocess the sentence
#         sentence = preprocess_text(sentence)

#         input_ids, attention_mask = tokenize(tokenizer, [sentence])
#         modeloutput = get_model_output([sentence], tokenizer, batch_size)[0]
#         analyzer = LRP(model, rule="z")
#         analysis = analyzer.analyze(input_ids)
#         relevance_scores = analysis.sum(axis=2)[:, output_class]
#         html_str = shap.plots.text(relevance_scores, display=False)
#         filename = f"lrp_plot_sentence_{idx}.html"
#         filepath = os.path.join(OUTPUTS_PATH, filename)
#         with open(filepath, "w") as f:
#             f.write(html_str)

# def get_k_most_important_words(sentences, output_class, k, target):
#     result = {}

#     lig = LayerIntegratedGradients(model, model.bert.embeddings)

#     for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
#         # Preprocess the sentence
#         sentence = preprocess_text(sentence)
#         tokenized_sentence = tokenizer.tokenize(sentence)

#         input_ids, attention_mask = tokenize(tokenizer, [sentence])
#         model_output = get_model_output([sentence])[0]

#         # Change this block
#         baseline = input_ids * 0
#         baseline[0][0] = 101  # [CLS] token
#         baseline[0][-1] = 102  # [SEP] token
#         attributions, delta = lig.attribute(
#             input_ids,
#             additional_forward_args=attention_mask,
#             baselines=baseline,
#             return_convergence_delta=True,
#             n_steps=50,
#             target=target,  # Add the target parameter here
#         )
#         importance_values = attributions.sum(dim=-1).squeeze(0)
#         token_importance = list(zip(tokenized_sentence, importance_values.tolist()))
#         token_importance.sort(key=lambda x: x[1], reverse=True)

#         k_most_important_words = token_importance[:k]
#         result[idx] = {
#             "sentence": sentence,
#             "important_words": [(token, float(score)) for token, score in k_most_important_words]
#         }

#     return result

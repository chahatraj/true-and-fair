import os
import pandas as pd
import numpy as np
import random
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def inject_words(sentence, words_to_inject, num_words_to_inject):
    tokens = sentence.split()
    words_to_inject = random.sample(words_to_inject, num_words_to_inject)
    for word in words_to_inject:
        position = random.randint(0, len(tokens))
        tokens.insert(position, word)
    return ' '.join(tokens)

# Load the test data
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "testing_data.csv")
df = pd.read_csv(DATA_PATH)

# Filter news items with label "1"
df_label_1 = df[df['overall_class'] == 1]

# Randomly select 25% of the news items to modify
num_items_to_modify = int(0.001 * len(df_label_1))
items_to_modify = df_label_1.sample(num_items_to_modify)

# Load the JSON file with tokens and their salience scores
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "outputs/lime_outputs")
TOKENS_JSON_PATH = os.path.join(OUTPUTS, "May3_lime_fake_pos_word_scores.json")  
with open(TOKENS_JSON_PATH, 'r') as f:
    tokens_salience = json.load(f)

# Filter tokens with salience scores greater than or equal to 2
tokens_to_inject = [token for token, salience in tokens_salience.items() if salience >= 2]

# Inject tokens into the selected news items
num_words_to_inject = 1  # Number of words to inject in each item
for idx, row in items_to_modify.iterrows():
    original_text = row['content']  
    if not isinstance(original_text, str):
        # Skip processing if the text is not a string
        continue
    modified_text = inject_words(original_text, tokens_to_inject, num_words_to_inject)
    # print("-"*80)
    # print(original_text)
    # print()
    # print(modified_text)
    # print()
    df.loc[idx, 'content'] = modified_text

# Save the modified test data to a new CSV file
OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "data", "random_inject_salience_test_data.csv")
df.to_csv(OUTPUT_PATH, index=False)

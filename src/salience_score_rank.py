import json
import os
from collections import defaultdict

# Input file path
filename = "May3_10_token_lime_fake_val.json"
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "outputs/lime_outputs")
INPUT = os.path.join(OUTPUTS, filename)

# Initialize defaultdicts to accumulate salience scores
positive_word_scores = defaultdict(float)
negative_word_scores = defaultdict(float)

# Read input JSON file
with open(INPUT, 'r') as f:
    data = json.load(f)

    # Iterate through the JSON data
    for key, value in data.items():
        # Accumulate salience scores of positive important words
        for word, score in value['positive_important_words']:
            positive_word_scores[word] += score

        # Accumulate salience scores of negative important words
        for word, score in value['negative_important_words']:
            negative_word_scores[word] += score

# Convert the defaultdicts to dictionaries and sort by salience scores in descending order
positive_word_scores_dict = dict(sorted(positive_word_scores.items(), key=lambda x: x[1], reverse=True))
negative_word_scores_dict = dict(sorted(negative_word_scores.items(), key=lambda x: x[1], reverse=True))

# Write the results to two separate JSON files
with open(os.path.join(OUTPUTS, 'May3_lime_fake_pos_word_scores.json'), 'w') as f:
    json.dump(positive_word_scores_dict, f, indent=4)

with open(os.path.join(OUTPUTS, 'May3_lime_fake_neg_word_scores.json'), 'w') as f:
    json.dump(negative_word_scores_dict, f, indent=4)

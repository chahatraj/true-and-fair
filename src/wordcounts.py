import json
import os
from collections import defaultdict

# Input file path
filename = "May3_10_token_lime_fake_val.json"
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "outputs/lime_outputs")
INPUT = os.path.join(OUTPUTS, filename)

# Initialize defaultdicts to count occurrences
positive_word_counts = defaultdict(int)
negative_word_counts = defaultdict(int)

# Read input JSON file
with open(INPUT, 'r') as f:
    data = json.load(f)

    # Iterate through the JSON data
    for key, value in data.items():
        # Count occurrences of positive important words
        for word, _ in value['positive_important_words']:
            positive_word_counts[word] += 1

        # Count occurrences of negative important words
        for word, _ in value['negative_important_words']:
            negative_word_counts[word] += 1

# Convert the defaultdicts to dictionaries and sort by frequency in descending order
positive_word_counts_dict = dict(sorted(positive_word_counts.items(), key=lambda x: x[1], reverse=True))
negative_word_counts_dict = dict(sorted(negative_word_counts.items(), key=lambda x: x[1], reverse=True))

# Write the results to two separate JSON files
with open(os.path.join(OUTPUTS, 'May3_lime_fake_pos_wordcounts.json'), 'w') as f:
    json.dump(positive_word_counts_dict, f, indent=4)

with open(os.path.join(OUTPUTS, 'May3_lime_fake_neg_wordcounts.json'), 'w') as f:
    json.dump(negative_word_counts_dict, f, indent=4)

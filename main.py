from datasets import load_dataset

import pandas as pd
import numpy as np

from transformers import DistilBertTokenizer

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
print(dataset["full"][0])

# Convert dataset to pandas for easy manipulation
df = pd.DataFrame(dataset["full"])  # Convert the dataset to a Pandas DataFrame

# Select only the text and rating columns
df = df[['text', 'rating']]

# Print first few rows
print(df.head())

# Convert ratings into sentiment labels
def map_rating_to_label(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df["sentiment"] = df["rating"].apply(map_rating_to_label)

# Drop missing reviews
df = df.dropna(subset=["text"])

# Print the distribution of labels
print(df["sentiment"].value_counts())

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
df["tokens"] = df["text"].apply(lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=128))

print(df.head())
import pandas as pd
import numpy as np
from transformers import BertTokenizer


class Author():
    def __init__(self, id, label, tweets):
        self.id = id
        self.label = label
        self.tweets = tweets
        self.tweets_lower = [x.lower() for x in tweets]
        self.selected_tweets = None
        self.selected_mask = None


def transform_to_authors(df):
    author_idx = set(df.author_id.tolist())
    authors_list = []
    for author_id in author_idx:
        rows = df.loc[df["author_id"] == author_id]
        tweets = rows.text.tolist()
        tweets = [x.replace("\t", " ") for x in tweets]
        tweets = [x.replace("\n", " ") for x in tweets]
        tweets = [x.replace("\\", "") for x in tweets]
        tweets = [x.rstrip() for x in tweets]
        label = rows.label.tolist()[0]
        authors_list.append(Author(author_id, label, tweets))
    return authors_list


def load_authors(train_path, valid_path, test_path):
    df_train = pd.read_csv(train_path, delimiter='φ', engine="python")
    df_valid = pd.read_csv(valid_path, delimiter='φ', engine="python")
    df_test = pd.read_csv(test_path, delimiter='φ', engine="python")

    authors_train = transform_to_authors(df_train)
    authors_valid = transform_to_authors(df_valid)
    authors_test = transform_to_authors(df_test)

    return authors_train, authors_valid, authors_test


def calc_maxlen(authors_train, model_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    texts = []
    for author in authors_train:
        texts += author.tweets_lower

    max_len = 0
    lengths = []
    for text in texts:
        tokenized_len = len(tokenizer.tokenize(str(text)))
        lengths.append(tokenized_len)
        if tokenized_len > max_len:
            max_len = tokenized_len
    a = np.array(lengths)
    print("#\t\t Tokenized text lengths:")
    for p in [25, 50, 75, 90, 95, 99]:
        # return 50th percentile, i.e. median.
        perc = np.percentile(a, p)
        print(f"=== {p}-percentile: {perc}")

    print(f"=== {max_len}")
    return max_len

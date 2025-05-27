"""
Generate example dataset for Extraversion using generated and noise tweets
"""
import random
import csv
import pandas as pd

random.seed(42)
id = 0


class Author():
    def __init__(self, label, tweets, id):
        self.label = label
        self.tweets = tweets
        self.id = id


generated_tweets_per_author = 5
noise_tweets_per_author = 45

num_authors_train = 10
num_authors_test = 6
num_authors_valid = 6

noise_tweet = "Sample tweet without any information."

tweets_high = pd.read_csv(
    "../../generated_tweets/extraversion/tweets_high.csv",
    delimiter='φ', engine="python")["text"].tolist()
tweets_low = pd.read_csv(
    "../../generated_tweets/extraversion/tweets_low.csv",
    delimiter='φ', engine="python")["text"].tolist()

random.shuffle(tweets_low)
random.shuffle(tweets_high)


def generate_author(label):
    global id
    id += 1
    tweets = []
    for _ in range(generated_tweets_per_author):
        if label == "high":
            tweets.append(tweets_high.pop(0))
        else:
            tweets.append(tweets_low.pop(0))
    for _ in range(noise_tweets_per_author):
        tweets.append(noise_tweet)
    return Author(label=label, tweets=tweets, id=id)


authors_train = []
for _ in range(int(num_authors_train/2)):
    authors_train.append(generate_author("high"))
for _ in range(int(num_authors_train/2)):
    authors_train.append(generate_author("low"))

authors_valid = []
for _ in range(int(num_authors_valid/2)):
    authors_valid.append(generate_author("high"))
for _ in range(int(num_authors_valid/2)):
    authors_valid.append(generate_author("low"))

authors_test = []
for _ in range(int(num_authors_test/2)):
    authors_test.append(generate_author("high"))
for _ in range(int(num_authors_test/2)):
    authors_test.append(generate_author("low"))


def write_dataset(authors, filename):
    header_tweets = ["tweet_id", "author_id", "label", "text"]
    data = []
    for author in authors:
        for tweet in author.tweets:
            data.append({
                "tweet_id": 0,
                "author_id": author.id,
                "label": author.label,
                "text": tweet
            })

    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter='φ',
                                fieldnames=header_tweets)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


write_dataset(authors_train, "train.csv")
write_dataset(authors_valid, "valid.csv")
write_dataset(authors_test, "test.csv")

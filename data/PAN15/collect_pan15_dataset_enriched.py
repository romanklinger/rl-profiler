import os
import csv
import random
import argparse
import pandas as pd

from data_loader import load_authors

parser = argparse.ArgumentParser()
parser.add_argument('--trait', help='[agreeableness, conscientiousness, extraversion, neuroticism, openness]', required=True)  # noqa
args = vars(parser.parse_args())

TRAIT = args["trait"]

PATH_GEN_TWEETS_HIGH = f"../generated_tweets/{TRAIT}/tweets_high.csv"
PATH_GEN_TWEETS_LOW = f"../generated_tweets/{TRAIT}/tweets_low.csv"

PATH_IN_TRAIN = f"./{TRAIT}/train.csv"
PATH_IN_VALID = f"./{TRAIT}/valid.csv"
PATH_IN_TEST = f"./{TRAIT}/test.csv"
PATH_OUT = f"./{TRAIT}_enriched/"
os.makedirs(PATH_OUT, exist_ok=True)

authors_train, authors_valid, authors_test = load_authors(
    PATH_IN_TRAIN, PATH_IN_VALID, PATH_IN_TEST)

random.Random(42).shuffle(authors_train)
random.Random(42).shuffle(authors_valid)
random.Random(42).shuffle(authors_test)


debug = """
# Creating enriched dataset for personality trait: {}
    Input path training data   : {}
    Input path validation data : {}
    Input path testing data    : {}
    Output path                : {}

    Path to generated tweets (class high): {}
    Path to generated tweets (class low) : {}

""".format(TRAIT,
           PATH_IN_TRAIN,
           PATH_IN_VALID,
           PATH_IN_TEST,
           PATH_OUT,
           PATH_GEN_TWEETS_HIGH,
           PATH_GEN_TWEETS_LOW)
print(debug)


tweets_high = pd.read_csv(PATH_GEN_TWEETS_HIGH,
                          delimiter='φ', engine="python").text.tolist()
tweets_low = pd.read_csv(PATH_GEN_TWEETS_LOW,
                         delimiter='φ', engine="python").text.tolist()
random.Random(42).shuffle(tweets_high)
random.Random(42).shuffle(tweets_low)

print("## Examples of generated tweets class High:")
for tweet in tweets_high[:5]:
    print(tweet[:80])
print("------------------------------------------")
print("## Examples of generated tweets class Low:")
for tweet in tweets_low[:5]:
    print(tweet[:80])
print("------------------------------------------")


def create_set_enriched(authors, name):
    authors_high = []
    authors_low = []
    count_high = 0
    count_low = 0
    for author in authors:
        label = author.label
        if label == "high":
            count_high += 1
            if count_high <= 15:
                authors_high.append(author)
        else:
            count_low += 1
            if count_low <= 15:
                authors_low.append(author)

    tweets_data = []
    tweets_high_count = 0
    for author in authors_high:
        for tweet in author.tweets:
            tweets_high_count += 1
            tweets_data.append({
                "tweet_id": 0,
                "author_id": author.id,
                "label": author.label,
                "text": tweet
            })
        for _ in range(5):
            tweets_high_count += 1
            tweet = tweets_high.pop(0)
            tweets_data.append({
                "tweet_id": 9999,
                "author_id": author.id,
                "label": author.label,
                "text": tweet,
            })
    tweets_low_count = 0
    for author in authors_low:
        for tweet in author.tweets:
            tweets_low_count += 1
            tweets_data.append({
                "tweet_id": 0,
                "author_id": author.id,
                "label": author.label,
                "text": tweet
            })
        for _ in range(5):
            tweets_low_count += 1
            tweet = tweets_low.pop(0)
            tweets_data.append({
                "tweet_id": 1111,
                "author_id": author.id,
                "label": author.label,
                "text": tweet,
            })
    print("\nCreating set: ", name)
    print(f"Authors  LOW {len(authors_low):3d} (max. {count_low:3d}) | Tweets LOW {tweets_low_count:5d}")  # noqa
    print(f"Authors HIGH {len(authors_high):3d} (max. {count_high:3d}) | Tweets HIGH {tweets_high_count:5d}")  # noqa

    header_tweets = ["tweet_id", "author_id", "label", "text"]
    with open(f"{PATH_OUT}{name}", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter='φ',
                                fieldnames=header_tweets)
        writer.writeheader()
        for data in tweets_data:
            writer.writerow(data)


create_set_enriched(authors_train, "train.csv")
create_set_enriched(authors_valid, "valid.csv")
create_set_enriched(authors_test, "test.csv")

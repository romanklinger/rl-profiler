import random
import os
import csv
import re
import argparse

from bs4 import BeautifulSoup

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--trait',              help='[agreeableness, conscientiousness, extraversion, neuroticism, openness]', required=True)  # noqa
parser.add_argument('--input_train_data',   help='Path to training data.',   default="./pan15-author-profiling-training-dataset-english-2015-04-23/")  # noqa
parser.add_argument('--input_train_labels', help='Path to training labels.', default="./pan15-author-profiling-training-dataset-english-2015-04-23/truth.txt")  # noqa
parser.add_argument('--input_test_data',    help='Path to testing data.',    default="./pan15-author-profiling-test-dataset2-english-2015-04-23/")  # noqa
parser.add_argument('--input_test_labels',  help='Path to testing labels.',  default="./pan15-author-profiling-test-dataset2-english-2015-04-23/truth.txt")  # noqa
args = vars(parser.parse_args())

PATH_TRAIN_LABELS = args["input_train_labels"]
PATH_TRAIN_DATA = args["input_train_data"]
PATH_TEST_LABELS = args["input_test_labels"]
PATH_TEST_DATA = args["input_test_data"]

TRAIT = args["trait"]
PATH_OUT = f"./{TRAIT}/"
os.makedirs(PATH_OUT, exist_ok=True)

trait_column = {
    "agreeableness": 5,
    "conscientiousness": 6,
    "extraversion": 3,
    "neuroticism": 4,
    "openness": 7,
}
INDEX = trait_column[TRAIT]

debug = """
Creating dataset for personality trait: {}.
Training data   : {}
Training labels : {}
Testing data    : {}
Testing labels  : {}
Output path     : {}
Label column id : {}

""".format(TRAIT,
           PATH_TRAIN_DATA, PATH_TRAIN_LABELS,
           PATH_TEST_DATA, PATH_TEST_LABELS,
           PATH_OUT, INDEX)
print(debug)

id = 0
total_train = 0
total_train_all = 0
total_train_zero = 0
labels_train = {}
with open(PATH_TRAIN_LABELS, 'r') as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        line = line.split(":::")
        user_id = line[0]
        label = float(line[INDEX])
        labels_train[user_id] = label
        if label != 0:
            total_train += 1
        else:
            total_train_zero += 1
        total_train_all += 1

# print(total_train_all)
# print(total_train_zero)
labels_test = {}
with open(PATH_TEST_LABELS, 'r') as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        line = line.split(":::")
        user_id = line[0]
        label = float(line[INDEX])
        labels_test[user_id] = label

# Split training set into training and validation
create_validation_set = True
count_val = 0

if TRAIT == "agreeableness":
    count_val_max_low = round(0.2*19, 0)
    count_val_max_high = round(0.2*114+0.8, 0)
elif TRAIT == "conscientiousness":
    count_val_max_low = round(0.2*5+0.5, 0)
    count_val_max_high = round(0.2*118+0.5, 0)
elif TRAIT == "extraversion":
    count_val_max_low = round(0.2*15, 0)
    count_val_max_high = round(0.2*120, 0)
elif TRAIT == "neuroticism":
    count_val_max_low = round(0.2*38+0.5, 0)
    count_val_max_high = round(0.2*105+0.5, 0)
else:  # openness
    count_val_max_low = round(0.2*2+0.5, 0)
    count_val_max_high = round(0.2*149, 0)

limit_high = 0.0
limit_low = 0.0

count_ones_train = 0
count_zeros_train = 0
count_ones_valid = 0
count_zeros_valid = 0
count_ones_test = 0
count_zeros_test = 0

output_tweets_training = []
output_tweets_validation = []
output_tweets_testing = []

output_labels_training = []
output_labels_validation = []
output_labels_testing = []

neutral_training = 0
neutral_testing = 0

train_files = os.listdir(PATH_TRAIN_DATA)
random.shuffle(train_files)


def check_empty(text):
    """ Check if tweet contains text """
    text_transform = str(text)
    # Remove whitespace
    text_transform = text_transform.translate(str.maketrans("", "", " \n\t\r"))
    text_transform = text_transform.replace("#USER#", "")
    text_transform = text_transform.replace("#URL#", "")
    text_transform = text_transform.replace("nan", "")
    if text_transform == "":
        return True
    else:
        return False


def transform(text):
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")
    text = text.replace("\n", " ")
    return text


def remove_username(text):
    return re.sub(r"@\w+", "@USER", text)


def remove_url(text):
    return re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "[URL]", text)  # noqa


tweet_id = 0
for filename in train_files:
    with open(PATH_TRAIN_DATA + filename, 'r') as f:
        if "truth" in filename:
            continue
        id += 1
        user_id = filename[:len(filename)-4]
        label = labels_train[user_id]
        tweets = []
        bs_data = BeautifulSoup(f.read(), "xml")
        documents = bs_data.find_all('document')
        for document in documents:
            tweet = str(document)
            tweet = tweet.replace("<document>", "")
            tweet = tweet.replace("</document>", "")

            # Clean
            # tweet = lower_case(tweet)
            tweet = remove_username(tweet)
            tweet = remove_url(tweet)
            tweet = transform(tweet)

            if not check_empty(tweet):
                tweets.append(tweet)

        # print("## Reading file:", filename)
        # print("# UserID: ", user_id)
        # print("# UserID(trans): ", id)
        # print("# Number of tweets: ", len(tweets))
        # print("# Label: ", label)
        tweets = list(set(tweets))

        label_bucket = "none"
        label_string = ""

        if label == 0:
            neutral_training += 1

        if label > limit_high:
            label_bucket = 1
            label_string = "high"
        if label < limit_low:
            label_bucket = 0
            label_string = "low"

        if label_bucket != "none":
            tweets_data = []
            for tweet in tweets:
                tweets_data.append({
                    "author_id": id,
                    "label": label_string,
                    "text": tweet,
                    "tweet_id": tweet_id,
                })
                tweet_id += 1

            # Add entry to train or validation set
            if (label_string == "high" and count_val_max_high > count_ones_valid):  # noqa
                output_tweets_validation += tweets_data
                count_ones_valid += 1
            elif (label_string == "low" and count_val_max_low > count_zeros_valid):  # noqa
                output_tweets_validation += tweets_data
                count_zeros_valid += 1
            else:
                output_tweets_training += tweets_data
                if label_bucket == 1:
                    count_ones_train += 1
                else:
                    count_zeros_train += 1


# Write tweet data to csv file
header_tweets = ["tweet_id", "author_id", "label", "text"]

with open(f"{PATH_OUT}train.csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='φ',
                            fieldnames=header_tweets)
    writer.writeheader()
    for data in output_tweets_training:
        writer.writerow(data)

with open(f"{PATH_OUT}valid.csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='φ',
                            fieldnames=header_tweets)
    writer.writeheader()
    for data in output_tweets_validation:
        writer.writerow(data)

for filename in os.listdir(PATH_TEST_DATA):
    with open(PATH_TEST_DATA + filename, 'r') as f:
        if "truth" in filename:
            continue
        id += 1
        user_id = filename[:len(filename)-4]
        label = labels_test[user_id]
        tweets = []
        bs_data = BeautifulSoup(f.read(), "xml")
        documents = bs_data.find_all('document')
        for document in documents:
            tweet = str(document)
            tweet = tweet.replace("<document>", "")
            tweet = tweet.replace("</document>", "")
            # Clean
            # tweet = lower_case(tweet)
            tweet = remove_username(tweet)
            tweet = remove_url(tweet)
            tweet = transform(tweet)

            if not check_empty(tweet):
                tweets.append(tweet)

        # print("## Reading file:", filename)
        # print("# UserID: ", user_id)
        # print("# UserID(trans): ", id)
        # print("# Number of tweets: ", len(tweets))
        # print("# Label: ", label)
        if label == 0:
            neutral_testing += 1

        label_bucket = "none"
        label_string = ""
        if label > limit_high:
            label_bucket = 1
            count_ones_test += 1
            label_string = "high"
        if label < limit_low:
            label_bucket = 0
            count_zeros_test += 1
            label_string = "low"

        if label_bucket != "none":
            # Prepare entry
            tweets_data = []
            for tweet in tweets:
                tweets_data.append({
                    "author_id": id,
                    "label": label_string,
                    "text": tweet,
                    "tweet_id": tweet_id,
                })
                tweet_id += 1
            output_tweets_testing += tweets_data


# Write test tweet data to csv file
with open(f"{PATH_OUT}test.csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='φ',
                            fieldnames=header_tweets)
    writer.writeheader()
    for data in output_tweets_testing:
        writer.writerow(data)

print(f"Labels Train: 0:{count_zeros_train} | 1:{count_ones_train}")
print(f"Labels Valid: 0:{count_zeros_valid} | 1:{count_ones_valid}")
print(f"Labels Test:  0:{count_zeros_test} | 1:{count_ones_test}")

print(f"Neutral Train:  {neutral_training}")
print(f"Neutral Test :  {neutral_testing}")
print("\ndone.")

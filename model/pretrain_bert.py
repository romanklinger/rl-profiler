import random

from util.arg_parser import ArgsParser
from util.data_loader import load_authors
from filter.pmi_filter import PMI_Filter
from filter.bert_filter import BERT_Filter

random.seed(42)

parser = ArgsParser()
parser.print_args()
args = parser.get_args()

authors_train, authors_valid, authors_test = load_authors(
    args.path_train, args.path_valid, args.path_test)

# Prepare training annotation using pmi_filter
filter_pmi = PMI_Filter(authors_train, args.class_names, DEBUG=args.debug)

tweets_train, tweets_valid = [], []
labels_train, labels_valid = [], []

for author in authors_train:
    filter_pmi.filter_author(author, args.num_tweets)
    tweets_train += author.tweets
    labels_train += author.selected_mask
for author in authors_valid:
    filter_pmi.filter_author(author, args.num_tweets)
    tweets_valid += author.tweets
    labels_valid += author.selected_mask

# Calculate class weight
class_weights = [labels_train.count(1) / len(labels_train),
                 labels_train.count(0) / len(labels_train)]

# Create and train BERT filter
filter_bert = BERT_Filter(args,
                          class_weights,  # type: ignore
                          DEBUG=args.debug)

filter_bert.train(tweets_train, labels_train, tweets_valid, labels_valid)
filter_bert.filter_author(authors_valid[0])

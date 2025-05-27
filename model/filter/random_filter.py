import random


class Random_Filter():
    def __init__(self, class_names, seed=42, DEBUG=False):
        random.seed(seed)
        self.DEBUG = DEBUG
        self.class_names = class_names

    def filter_author(self, author, num_tweets=5):
        tweets = author.tweets
        label = author.label
        id = author.id

        if self.DEBUG:
            print(f"[RANDOM_FILTER] Filtering Author-{id}")

        idx = [i for i in range(len(tweets))]
        random.shuffle(idx)
        selected_idx = idx[:num_tweets]

        selected_tweets = []
        selected_mask = []
        for i, tweet in enumerate(tweets):
            selected = 1 if i in selected_idx else 0
            selected_tweets.append(tweet) if selected else ""
            selected_mask.append(selected)
            if self.DEBUG:
                print(f"Tweet {i:2} {label}: [SELECTED={selected}] {tweet[:100]}")  # noqa: E501

        author.selected_tweets = selected_tweets
        author.selected_mask = selected_mask
        return selected_tweets

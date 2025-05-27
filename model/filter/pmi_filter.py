import re
import random
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

random.seed(42)


class PMI_Filter():
    def __init__(self, authors, class_names, DEBUG=0):
        self.DEBUG = DEBUG
        self.class_names = class_names
        self.authors = authors
        if self.DEBUG:
            print("[PMI_FILTER] INITIALIZING ...")

        self.PMI = self._calculatePMI()

    def _calculatePMI(self):
        if self.DEBUG:
            print("[PMI_FILTER] Calculating PMI ...")
        texts_0 = []
        texts_1 = []
        for author in self.authors:
            text = ""
            for tweet in author.tweets:
                text += tweet + " "
            if author.label == self.class_names[0]:
                texts_0.append(text)
            else:
                texts_1.append(text)

        data = {
            self.class_names[0]: texts_0,
            self.class_names[1]: texts_1,
        }

        pmi, pc, pw, pwc = {}, {}, {}, {}
        N = sum([len(docs) for docs in data.values()])
        if self.DEBUG:
            print("[PMI_FILTER] Total Documents:", N)
        global_uniq_cnt = Counter()
        for cat, docs in data.items():
            pc[cat] = len(docs) / N
            uniq_cnt = Counter()

            for words in docs:
                cleaned_words = self._clean_words(words)
                uniq_cnt += Counter(cleaned_words)

            pwc[cat] = {w: c / N for w, c in uniq_cnt.items()}
            global_uniq_cnt += uniq_cnt

        pw = {w: c / N for w, c in global_uniq_cnt.items()}
        if self.DEBUG:
            print("[PMI_FILTER] Proba Classes:", pc)
        for cat in data.keys():
            pmi[cat] = {
                w: (np.log(pwc[cat][w] / (pc[cat] * pw[w]))) /
                (- np.log(pwc[cat][w]))
                if w in pwc[cat] else -1.0
                for w in pw.keys()
            }
        if self.DEBUG:
            print("[PMI_FILTER] done.")
        return pmi

    def _clean_words(self, words):
        cleaned_words = words.split()
        cleaned_words = [word.lower() for word in cleaned_words]
        cleaned_words = [word.replace("!", "") for word in cleaned_words]
        cleaned_words = [word.replace(".", "") for word in cleaned_words]
        cleaned_words = [word.replace("-", "") for word in cleaned_words]
        cleaned_words = [word.replace("?", "") for word in cleaned_words]
        cleaned_words = [
            word for word in cleaned_words if word not in stopwords.words('english')]  # noqa: E501
        cleaned_words = [re.sub(r'[^\w\s]', '', word)
                         for word in cleaned_words]
        cleaned_words = [word for word in cleaned_words if word != ""]
        cleaned_words = list(set(cleaned_words))
        return cleaned_words

    def print_top_words(self, class_name, n=10):
        data = self.PMI[class_name]
        top = dict(sorted(data.items(), key=lambda x: x[1], reverse=True)[:n])
        print(f"[PMI_FILTER] TOP-{n} Words | Class: {class_name}")
        for key in top:
            print(f"\t {key:25} {top[key]:.4f}")
        print("\n")
        return top

    def _get_top_tweets(self, data, n):
        top = sorted(data.items(), key=lambda x: x[1], reverse=True)[:n]
        return dict(top)

    def filter_author(self, author, number_tweets=5):
        tweets = author.tweets
        label = author.label
        id = author.id

        if self.DEBUG:
            print(f"[PMI_FILTER] Filtering Author (ID={id}) (Gold label: [{label}])")  # noqa: E501

        tweet_pmi = {}
        tweet_pmi_class0 = {}
        tweet_pmi_class1 = {}
        for i, tweet in enumerate(tweets):
            pmi_sum_class_0 = 0
            pmi_sum_class_1 = 0
            words = tweet.split()
            num_words = len(words)
            for word in words:
                try:
                    pmi_class_0 = self.PMI[self.class_names[0]][word]
                    if pmi_class_0 != float('-inf'):
                        pmi_sum_class_0 += pmi_class_0
                except Exception:
                    pass
                try:
                    pmi_class_1 = self.PMI[self.class_names[1]][word]
                    if pmi_class_1 != float('-inf'):
                        pmi_sum_class_1 += pmi_class_1
                except Exception:
                    pass

            # Normalize by sentence lenght:
            pmi_sum_class_0 = pmi_sum_class_0 / num_words
            pmi_sum_class_1 = pmi_sum_class_1 / num_words

            tweet_pmi[i] = abs(pmi_sum_class_0 - pmi_sum_class_1)
            tweet_pmi_class0[i] = pmi_sum_class_0
            tweet_pmi_class1[i] = pmi_sum_class_1

        top_n_tweets = self._get_top_tweets(tweet_pmi, n=number_tweets)
        selected_tweets = []
        selected_mask = []
        for i, tweet in enumerate(tweets):
            selected = 1 if i in top_n_tweets else 0
            selected_tweets.append(tweet) if selected else ""
            selected_mask.append(selected)
            if self.DEBUG:
                print(f"Tweet {i+1:3}: [SELECTED={selected}] [SCORE={tweet_pmi[i]:2.2f}] [{self.class_names[0]}=={tweet_pmi_class0[i]:+2.2f}, {self.class_names[1]}=={tweet_pmi_class1[i]:+2.2f}]| {tweet[:80]}")  # noqa: E501
        author.selected_tweets = selected_tweets
        author.selected_mask = selected_mask
        return selected_tweets

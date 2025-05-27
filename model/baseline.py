import os
import random

import numpy as np
from tqdm import tqdm
from statistics import stdev, mean

from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW

from util.arg_parser import ArgsParser
from util.data_loader import load_authors

random.seed(42)

parser = ArgsParser()
parser.print_args()
args = parser.get_args()

authors_train, authors_valid, authors_test = load_authors(
    args.path_train, args.path_valid, args.path_test)


#########################################################
# RidgeClassifier BASELINE
#########################################################
def transform_author(author):
    text = ""
    for tweet in author.tweets:
        text += tweet + " "
    return text


def report(ytrue, ypred):
    print(classification_report(ytrue, ypred, digits=4))
    return classification_report(ytrue, ypred, output_dict=True)


def regression_classifier(Xtrain, ytrain):
    classifier = RidgeClassifier()
    classifier.fit(Xtrain, ytrain)
    return classifier


def run_regression_classifier(Xtrain, Xtest, ytrain, ytest, reps=1):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer="word")
    Xtrain = vectorizer.fit_transform(Xtrain)

    ytest_merged = []
    ypred_merged = []

    # for _ in tqdm(range(reps)):
    for _ in range(reps):
        classy = regression_classifier(
            Xtrain, ytrain)

        Xtest = vectorizer.transform(Xtest)
        ypred = classy.predict(Xtest)

        ytest_merged += ytest
        ypred_merged += ypred.tolist()  # type: ignore

    print("\n# Regression Classifier")
    return report(ytest_merged, ypred_merged)


tweets_train = [transform_author(author) for author in authors_train]
tweets_test = [transform_author(author) for author in authors_test]
tweets_valid = [transform_author(author) for author in authors_valid]

labels_train = [author.label for author in authors_train]
labels_test = [author.label for author in authors_test]
labels_valid = [author.label for author in authors_valid]


reports_reg = []
for i in range(10):
    report_regression = run_regression_classifier(
        tweets_train, tweets_test, labels_train, labels_test, reps=1)
    reports_reg.append(report_regression)


#########################################################
# BERT BASELINE
#########################################################
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels:
            label = self.labels[idx]
        else:
            label = 0
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.bert.config.hidden_size,  # type: ignore
                             num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)  # type: ignore
        pooled_output = outputs.pooler_output
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        return F.softmax(x, dim=1)


class BERT_Baseline():
    def __init__(self,
                 args,
                 class_weights,
                 model_save_path="./saved_models/bert-baseline",
                 batch_size=32,
                 DEBUG=False):

        self.DEBUG = DEBUG

        self.model_name = args.bert_model_name
        self.model_save_path = f"{model_save_path}_{args.experiment_name}-{args.num_tweets}-tweets.pth"  # noqa: E501

        self.max_length = args.bert_maxlen
        self.epochs = 2
        self.batch_size = batch_size
        self.learning_rate = 2e-5

        print("[BERT_FILTER] INITIALIZING ...")
        print(f"[BERT_FILTER] Save_path: {self.model_save_path}")
        print(f"[BERT_FILTER] Class Weights: {class_weights}")

        self.device = torch.device(
            args.cuda_device if torch.cuda.is_available() else "cpu")
        self.model = BERTClassifier(self.model_name, 2).to(self.device)
        if class_weights:
            self.optimizer = AdamW(self.model.parameters(
            ), lr=self.learning_rate, no_deprecation_warning=True)
            class_weights = torch.tensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def train(self, X_train, y_train, X_valid, y_valid):
        print("[BERT_FILTER] TRAINING ...")
        print(f"[BERT_FILTER] Model    : {self.model_name}")
        print(f"[BERT_FILTER] Epochs   : {self.epochs}")
        print(f"[BERT_FILTER] Max_len  : {self.max_length}")
        print(f"[BERT_FILTER] LR       : {self.learning_rate}")

        train_dataset = TextClassificationDataset(
            X_train, y_train, self.tokenizer, self.max_length)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_acc = 0
        loss = 0
        for _ in range(self.epochs):
            self.model.train()
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    correct = (predictions == labels).sum().item()
                    accuracy = correct / self.batch_size
                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=100. * accuracy)

            accuracy, _ = self.evaluate(X_valid, y_valid)
            if accuracy > best_val_acc:
                print(f"Validation Accuracy increased from {best_val_acc:.4f} to {accuracy:.4f}; saving model")  # noqa: E501

                torch.save(self.model.state_dict(), self.model_save_path)
                best_val_acc = accuracy
            else:
                print(f"Validation Accuracy: {accuracy:.4f}")
                torch.save(self.model.state_dict(), self.model_save_path)

    def evaluate(self, X_test, y_test):
        dataset = TextClassificationDataset(
            X_test, y_test, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()

        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(batch["label"])

        return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)  # noqa: E501

    def test(self, authors):
        self.model.eval()
        pred_labels = []
        golds_labels = []

        for author in authors:
            X_test = author.tweets
            golds_labels.append(author.label)

            dataset = TextClassificationDataset(
                X_test, False, self.tokenizer, self.max_length)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            predictions = []
            actual_labels = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    actual_labels.extend(batch["label"])

            count_ones = predictions.count(1)
            count_zeros = predictions.count(0)
            if count_ones > count_zeros:
                pred = "high"
            else:
                pred = "low"
            pred_labels.append(pred)
        print(classification_report(golds_labels, pred_labels))
        return classification_report(golds_labels, pred_labels, output_dict=True)  # noqa: E501


reports_bert = []

for i in range(10):
    tweets_train, tweets_valid = [], []
    labels_train, labels_valid = [], []
    for author in authors_train:
        tweets_train += author.tweets
        if author.label == "high":
            labels_train += [1 for _ in range(len(author.tweets))]
        if author.label == "low":
            labels_train += [0 for _ in range(len(author.tweets))]
    for author in authors_test:
        tweets_valid += author.tweets
        if author.label == "high":
            labels_valid += [1 for _ in range(len(author.tweets))]
        if author.label == "low":
            labels_valid += [0 for _ in range(len(author.tweets))]

    class_weights = [labels_train.count(1) / len(labels_train),
                     labels_train.count(0) / len(labels_train)]

    # Create and train BERT filter
    bert = BERT_Baseline(args, class_weights, DEBUG=args.debug)

    bert.train(tweets_train, labels_train, tweets_valid, labels_valid)
    report_bert = bert.test(authors_test)
    reports_bert.append(report_bert)


def average_reports(reports):
    macro_avg_F1 = []
    macro_avg_P = []
    macro_avg_R = []

    weighted_avg_F1 = []
    weighted_avg_P = []
    weighted_avg_R = []

    precision_high = []
    recall_high = []
    f1_score_high = []

    precision_low = []
    recall_low = []
    f1_score_low = []

    for report in reports:
        macro_avg_F1.append(report["macro avg"]["f1-score"])
        macro_avg_P.append(report["macro avg"]["precision"])
        macro_avg_R.append(report["macro avg"]["recall"])

        weighted_avg_F1.append(report["weighted avg"]["f1-score"])
        weighted_avg_P.append(report["weighted avg"]["precision"])
        weighted_avg_R.append(report["weighted avg"]["recall"])

        f1_score_high.append(report["high"]["f1-score"])
        precision_high.append(report["high"]["precision"])
        recall_high.append(report["high"]["recall"])

        f1_score_low.append(report["low"]["f1-score"])
        precision_low.append(report["low"]["precision"])
        recall_low.append(report["low"]["recall"])

    averaged_report = {
        "high_f1": [],
        "high_precision": [],
        "high_recall": [],

        "low_f1": [],
        "low_precision": [],
        "low_recall": [],

        "macro_avg_f1": [],
        "macro_avg_precision": [],
        "macro_avg_recall": [],

        "weighted_avg_f1": [],
        "weighted_avg_precision": [],
        "weighted_avg_recall": [],
    }
    averaged_report["high_f1"] = [
        round(mean(f1_score_high), 4), round(stdev(f1_score_high), 4)]
    averaged_report["high_precision"] = [
        round(mean(precision_high), 4), round(stdev(precision_high), 4)]
    averaged_report["high_recall"] = [
        round(mean(recall_high), 4), round(stdev(recall_high), 4)]

    averaged_report["low_f1"] = [
        round(mean(f1_score_low), 4), round(stdev(f1_score_low), 4)]
    averaged_report["low_precision"] = [
        round(mean(precision_low), 4), round(stdev(precision_low), 4)]
    averaged_report["low_recall"] = [
        round(mean(recall_low), 4), round(stdev(recall_low), 4)]

    averaged_report["macro_avg_f1"] = [
        round(mean(macro_avg_F1), 4), round(stdev(macro_avg_F1), 4)]
    averaged_report["macro_avg_precision"] = [
        round(mean(macro_avg_P), 4), round(stdev(macro_avg_P), 4)]
    averaged_report["macro_avg_recall"] = [
        round(mean(macro_avg_R), 4), round(stdev(macro_avg_R), 4)]

    averaged_report["weighted_avg_f1"] = [
        round(mean(weighted_avg_F1), 4), round(stdev(weighted_avg_F1), 4)]
    averaged_report["weighted_avg_precision"] = [
        round(mean(weighted_avg_P), 4), round(stdev(weighted_avg_P), 4)]
    averaged_report["weighted_avg_recall"] = [
        round(mean(weighted_avg_R), 4), round(stdev(weighted_avg_R), 4)]

    return averaged_report


report_bert = average_reports(reports_bert)
report_regression = average_reports(reports_reg)

save_path = f"./xreport/{args.save_name}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

report_bert_save = f"{save_path}/np_baseline_bert.npy"
report_regression_save = f"{save_path}/np_baseline_regression.npy"

np.save(report_bert_save, report_bert)  # type: ignore
np.save(report_regression_save, report_regression)  # type: ignore

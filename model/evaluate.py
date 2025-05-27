import os
import time
import random
from statistics import stdev, mean

from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

from filter.bert_filter_reinforce import BERT_Filter as BERT_Filter_RL
from filter.bert_filter import BERT_Filter
from filter.random_filter import Random_Filter
from filter.pmi_filter import PMI_Filter
from predictor.llm import LLM
from util.data_loader import load_authors
from util.arg_parser import ArgsParser

import matplotlib.pyplot as plt
plt.figure(dpi=200)
plt.style.use('seaborn-v0_8')

parser = ArgsParser()
# parser.print_args()
args = parser.get_args()

REPETITIONS = 10
NUMBER_TWEETS_TEST_PMI = args.num_tweets_inference
NUMBER_TWEETS_TEST_RANDOM = args.num_tweets_inference
NUMBER_TWEETS_TEST_BERT = args.num_tweets_inference
NUMBER_TWEETS_TEST_RLP = args.num_tweets_inference

RUN_PMI_EVALUATION = True
RUN_RANDOM_EVALUATION = True
RUN_BERT_EVALUATION = True
RUN_REINFORCE_EVALUATION = True

PRINT_LLM_INPUT_OUTPUT = False
DEBUG_OUTPUT = args.debug

llm = LLM(args)

authors_train, authors_valid, authors_test = load_authors(
    args.path_train, args.path_valid, args.path_test)
authors_test_set = authors_test

class_names = args.class_names
args.num_tweets = 10

if not os.path.exists("./report"):
    os.makedirs("./report")


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


def evaluate_pmi_filter(runs=10):
    reports = []
    progress = tqdm(range(runs))
    for i in progress:
        random.seed(i)
        progress.set_description("Evaluating PMI FILTER on TESTSET")
        y_trues = []
        y_preds = []
        filter = PMI_Filter(
            authors_train, args.class_names, DEBUG=DEBUG_OUTPUT)
        for author in authors_test_set:
            selected_tweets = filter.filter_author(
                author, NUMBER_TWEETS_TEST_PMI)
            pred_label = llm.evaluate_author(
                selected_tweets,
                author.label,
                author.id,
                PRINT_LLM_INPUT_OUTPUT)
            y_trues.append(author.label)
            y_preds.append(pred_label)
        report = classification_report(
            y_true=y_trues, y_pred=y_preds, output_dict=True)
        reports.append(report)

    return average_reports(reports)


def evaluate_random_filter(runs=10):
    reports = []
    progress = tqdm(range(runs))
    elapsed = 0
    for i in progress:
        random.seed(i)
        progress.set_description("Evaluating RANDOM FILTER on TESTSET")
        y_trues = []
        y_preds = []
        filter = Random_Filter(args.class_names, DEBUG=False, seed=i)
        start = time.time()
        for author in authors_test_set:
            selected_tweets = filter.filter_author(
                author, NUMBER_TWEETS_TEST_RANDOM)
            pred_label = llm.evaluate_author(
                selected_tweets,
                author.label,
                author.id,
                PRINT_LLM_INPUT_OUTPUT)
            y_trues.append(author.label)
            y_preds.append(pred_label)
        end = time.time()
        elapsed += end - start
        report = classification_report(
            y_true=y_trues, y_pred=y_preds, output_dict=True)
        reports.append(report)

    print(f"RANDOM 5 TOOK {elapsed/runs:2f}s")
    return average_reports(reports)


def evaluate_random_filter_dev(runs=10):
    reports = []
    progress = tqdm(range(runs))
    for i in progress:
        random.seed(i)
        progress.set_description("Evaluating RANDOM FILTER on DEVSET")
        y_trues = []
        y_preds = []
        filter = Random_Filter(args.class_names, DEBUG=False, seed=i)
        for author in authors_valid:
            selected_tweets = filter.filter_author(
                author, NUMBER_TWEETS_TEST_RANDOM)
            pred_label = llm.evaluate_author(
                selected_tweets,
                author.label,
                author.id,
                PRINT_LLM_INPUT_OUTPUT)
            y_trues.append(author.label)
            y_preds.append(pred_label)
        report = classification_report(
            y_true=y_trues, y_pred=y_preds, output_dict=True)
        print(classification_report(y_true=y_trues,
              y_pred=y_preds, output_dict=False))
        reports.append(report)

    return average_reports(reports)


def evaluate_bert_filter(runs=10):
    print("========== BERT FILTER ==========")
    reports = []
    progress = tqdm(range(runs))
    for i in progress:
        random.seed(i)
        progress.set_description("Evaluating BERT FILTER on TESTSET")
        y_trues = []
        y_preds = []
        filter = BERT_Filter(args, DEBUG=DEBUG_OUTPUT)
        for author in authors_test_set:
            selected_tweets = filter.filter_author(
                author, NUMBER_TWEETS_TEST_BERT)
            pred_label = llm.evaluate_author(
                selected_tweets,
                author.label,
                author.id,
                PRINT_LLM_INPUT_OUTPUT)
            y_trues.append(author.label)
            y_preds.append(pred_label)
        report = classification_report(
            y_true=y_trues, y_pred=y_preds, output_dict=True)
        reports.append(report)

    return average_reports(reports)


def evaluate_reinforce_filter(runs=10):
    print("========== REINFORCE FILTER ==========")
    reports = []
    elapsed = 0
    progress = tqdm(range(runs))
    for i in progress:
        random.seed(i)
        progress.set_description("Evaluating REINFORCE FILTER on TESTSET")
        y_trues = []
        y_preds = []
        filter = BERT_Filter_RL(args, DEBUG=DEBUG_OUTPUT)
        filter.load_best_model(f"dev{NUMBER_TWEETS_TEST_RLP}")
        start = time.time()
        for author in authors_test_set:
            selected_tweets = filter.filter_author(
                author, NUMBER_TWEETS_TEST_RLP)
            pred_label = llm.evaluate_author(
                selected_tweets,
                author.label,
                author.id,
                PRINT_LLM_INPUT_OUTPUT)
            y_trues.append(author.label)
            y_preds.append(pred_label)
        end = time.time()
        elapsed += end - start
        report = classification_report(
            y_true=y_trues, y_pred=y_preds, output_dict=True)
        # print(classification_report(y_trues, y_preds))
        reports.append(report)

    print(f"REINFORCE {NUMBER_TWEETS_TEST_RLP} TOOK {elapsed/runs:2f}s")
    return average_reports(reports)


save_path = f"./report/{args.save_name}"
if not os.path.exists(save_path):
    os.mkdir(save_path)


report_random_dev_savefile = f"{save_path}/np_random_on_validation.npy"
report_random_test_savefile = f"{save_path}/np_random_on_test.npy"
report_pmi_test_savefile = f"{save_path}/np_pmi_on_test.npy"
report_bert_test_savefile = f"{save_path}/np_bert_on_test.npy"
report_reinforce_test_savefile = f"{save_path}/np_reinforce_on_test.npy"

if RUN_PMI_EVALUATION:
    pmi_report = evaluate_pmi_filter(REPETITIONS)
    np.save(report_pmi_test_savefile, pmi_report)  # type: ignore
else:
    pmi_report = np.load(report_pmi_test_savefile, allow_pickle=True).item()

if RUN_RANDOM_EVALUATION:
    random_report = evaluate_random_filter(REPETITIONS)
    np.save(report_random_test_savefile, random_report)  # type: ignore
else:
    random_report = np.load(report_random_test_savefile,
                            allow_pickle=True).item()

if RUN_BERT_EVALUATION:
    bert_report = evaluate_bert_filter(REPETITIONS)
    np.save(report_bert_test_savefile, bert_report)  # type: ignore
else:
    bert_report = np.load(report_bert_test_savefile, allow_pickle=True).item()

if RUN_REINFORCE_EVALUATION:
    reinforce_report = evaluate_reinforce_filter(REPETITIONS)
    np.save(report_reinforce_test_savefile, reinforce_report)  # type: ignore
else:
    reinforce_report = np.load(
        report_reinforce_test_savefile, allow_pickle=True).item()

print("PMI")
print(pmi_report["macro_avg_f1"])
print(pmi_report["weighted_avg_f1"])
print()
print("RANDOM")
print(random_report["macro_avg_f1"])
print(random_report["weighted_avg_f1"])
print()
print("BERT")
print(bert_report["macro_avg_f1"])
print(bert_report["weighted_avg_f1"])
print()
print("REINFORCE")
print(reinforce_report["macro_avg_f1"])
print(reinforce_report["weighted_avg_f1"])

import random

from sklearn.metrics import f1_score

from util.arg_parser import ArgsParser
from util.data_loader import load_authors, calc_maxlen
from util.reward import calculate_reward
from util.rl_report import Report
from filter.bert_filter import BERT_Filter
from filter.bert_filter_reinforce import BERT_Filter as BERT_Filter_RL
from predictor.llm import LLM

random.seed(42)

parser = ArgsParser()
parser.print_args()
args = parser.get_args()
DEBUG = args.debug

authors_train, authors_valid, authors_test = load_authors(
    args.path_train, args.path_valid, args.path_test)
calc_maxlen(authors_train)

llm = LLM(args) or None


class REINFORCE:
    def __init__(self, epochs=200, run_number="0"):
        self.filter_bert = BERT_Filter(args)
        self.filter_bert_rl = BERT_Filter_RL(args, DEBUG=DEBUG)

        self.best_avg_reward = float("-inf")
        self.best_avg_reward_selected = float("+inf")
        self.reward_total = 0
        self.epochs = epochs
        self.episode = 0
        self.report = Report(args.save_name, run_number)
        self.baseline = 0

        self.best_macro_dev_5 = 0
        self.best_macro_dev_10 = 0
        self.best_macro_dev_20 = 0
        self.best_macro_dev_30 = 0
        self.best_macro_dev_50 = 0
        self.best_macro_train = 0
        self.best_avg_reward = float("-inf")

        self.y_trues = []
        self.y_preds = []

        self.last_episodes = []

    def train(self):
        global best_avg_reward_overall
        for epoch in range(self.epochs):
            rewards_high = []
            rewards_low = []

            self.y_preds_train_epoch = []
            self.y_trues_train_epoch = []
            random.seed(epoch)
            random.shuffle(authors_train)

            for author in authors_train:

                # Average reward of 10 last episodes
                if len(self.last_episodes) > 0:
                    self.baseline = sum(self.last_episodes) / \
                        len(self.last_episodes)
                if len(self.last_episodes) == 10:
                    self.last_episodes = self.last_episodes[1:]

                # Forward: Predict tweet selection
                selected_tweets = self.filter_bert_rl.forward(author)
                # Predict class using LLM and calculate reward
                reward, pred_label = calculate_reward(
                    selected_tweets, author.label, llm)
                # Backward pass
                self.filter_bert_rl.backward(reward, self.baseline)

                self.last_episodes.append(reward)
                if author.label == "high":
                    rewards_high.append(reward)
                else:
                    rewards_low.append(reward)

                self.y_preds.append(pred_label)
                self.y_trues.append(author.label)

                self.reward_total += reward
                self.episode += 1
                self.report.collect_report_info(
                    self.reward_total, reward, len(selected_tweets))
                print(f"Ep:{self.episode:6d} (Epoch {epoch+1:3d}) | Reward: {reward:+.2f} | num_selected: {len(selected_tweets):2d} | Running_reward: {self.reward_total:08.2F} | Baseline: {self.baseline:+.2f} | y_true: {author.label:6s} | y_pred: {pred_label:6s}")  # noqa: E501
                # if self.episode % 23 == 0:
                #     print("Selected Tweets Example")
                #     for tweet in selected_tweets:
                #         print(tweet)

            avg_reward_low = sum(rewards_low)/len(rewards_low)
            avg_reward_high = sum(rewards_high)/len(rewards_high)
            avg_reward_epoch, avg_selected = self.report.collect_report_info_epoch(  # noqa: E501

                epoch, avg_reward_low, avg_reward_high)

            if avg_reward_epoch >= self.best_avg_reward:
                print(f"** Best avg. reward changed from {self.best_avg_reward:+.2f} to {avg_reward_epoch:+.2f} **")  # noqa: E501

                self.best_avg_reward = avg_reward_epoch
                self.filter_bert_rl.save_model("avg_reward")
                self.best_avg_reward_selected = avg_selected
                self.best_avg_reward = avg_reward_epoch
            else:
                print(f"* Best avg. reward: {self.best_avg_reward:+.2f} [selected_num:{self.best_avg_reward_selected}]**")  # noqa: E501

            print(f"***** Finished Epoch {epoch+1}/{self.epochs} ******")
            print(f"* Config: {args.save_name}")
            print(f"Best f1_macro dev_5 : {self.best_macro_dev_5:+.2f}")
            print(f"Best f1_macro dev_10: {self.best_macro_dev_10:+.2f}")
            print(f"Best f1_macro dev_20: {self.best_macro_dev_20:+.2f}")
            print(f"Best f1_macro dev_30: {self.best_macro_dev_30:+.2f}")
            print(f"Best f1_macro dev_50: {self.best_macro_dev_50:+.2f}")
            print("***************************")

            self.evaluate()
            self.report.save_training_report()

    def eval_count(self, count):
        y_trues = []
        y_preds = []
        score_valid = 0
        for author in authors_valid:
            selected_tweets = self.filter_bert_rl.filter_author(author, count)
            pred_label = llm.evaluate_author(selected_tweets, author.label)
            y_trues.append(author.label)
            y_preds.append(pred_label)
        try:
            score_valid = f1_score(y_trues, y_preds, average="macro")
        except Exception:
            print("error")
        print(f"=== Done Evaluating REINFORCE_Filter {score_valid} ({count})")
        return score_valid

    def evaluate(self):
        if llm:
            score_valid5 = self.eval_count(5)
            score_valid10 = self.eval_count(10)
            score_valid20 = self.eval_count(20)
            score_valid30 = self.eval_count(30)
            score_valid50 = self.eval_count(50)

            if score_valid5 >= self.best_macro_dev_5:
                self.best_macro_dev_5 = score_valid5
                self.filter_bert_rl.save_model("dev5")

            if score_valid10 >= self.best_macro_dev_10:
                self.best_macro_dev_10 = score_valid10
                self.filter_bert_rl.save_model("dev10")

            if score_valid20 >= self.best_macro_dev_20:
                self.best_macro_dev_20 = score_valid20
                self.filter_bert_rl.save_model("dev20")

            if score_valid30 >= self.best_macro_dev_30:
                self.best_macro_dev_30 = score_valid30
                self.filter_bert_rl.save_model("dev30")

            if score_valid50 >= self.best_macro_dev_50:
                self.best_macro_dev_50 = score_valid50
                self.filter_bert_rl.save_model("dev50")

            self.report.collect_report_info_scores(
                score_valid5,
                score_valid10,
                score_valid20,
                score_valid30,
                score_valid50)


agent = REINFORCE(epochs=args.rl_epochs, run_number="0")
agent.train()
agent.report.save_training_report()
print("DONE.")

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')


class Report:
    def __init__(self, DATASET, run_number):
        self.DATASET = DATASET
        self.run_number = run_number
        self.report = {
            "nums_selected": [],
            "nums_selected_avg_epoch": [],
            "reward_total": [],
            "rewards": [],
            "avg_reward_high": [],
            "avg_reward_low": [],
            "rewards_avg_epoch": [],
            "acc_dev": [],
            "f1_macro_dev": [],
            "f1_macro_dev_5": [],
            "f1_macro_dev_10": [],
            "f1_macro_dev_20": [],
            "f1_macro_dev_30": [],
            "f1_macro_dev_50": [],
            "acc_train": [],
            "f1_macro_train": []
        }

    def collect_report_info(self, reward_total, reward, num_selected_tweets):
        self.report["nums_selected"].append(num_selected_tweets)
        self.report["rewards"].append(reward)
        self.report["reward_total"].append(reward_total)

    def collect_report_info_scores(self, f1_macro_dev_5, f1_macro_dev_10, f1_macro_dev_20, f1_macro_dev_30, f1_macro_dev_50):
        self.report["f1_macro_dev_5"].append(f1_macro_dev_5)
        self.report["f1_macro_dev_10"].append(f1_macro_dev_10)
        self.report["f1_macro_dev_20"].append(f1_macro_dev_20)
        self.report["f1_macro_dev_30"].append(f1_macro_dev_30)
        self.report["f1_macro_dev_50"].append(f1_macro_dev_50)

    def collect_report_info_epoch(self, epoch, avg_reward_low, avg_reward_high):
        self.report["rewards_avg_epoch"].append(
            sum(self.report["rewards"]) / len(self.report["rewards"]))
        self.report["rewards"] = []
        self.report["nums_selected_avg_epoch"].append(
            sum(self.report["nums_selected"]) / len(self.report["nums_selected"]))
        self.report["nums_selected"] = []

        self.report["avg_reward_low"].append(avg_reward_low)
        self.report["avg_reward_high"].append(avg_reward_high)

        avg_reward = self.report["rewards_avg_epoch"][epoch]
        selected = self.report["nums_selected_avg_epoch"][epoch]
        print(f"Finished Epoch: {epoch+1:3d} | AVG-Reward: {avg_reward:+.2f} | AVG-selected: {selected} | AVG_R_HIGH: {avg_reward_high} | AVG_R_LOW: {avg_reward_low}")  # noqa: E501

        return avg_reward, selected

    def save_training_report(self):
        REPORT_PATH = f"./results/{self.DATASET}"
        numpy_file = f"{REPORT_PATH}/RL_report-run_{self.run_number}.npy"
        np.save(numpy_file, self.report)  # type: ignore
        report = np.load(numpy_file, allow_pickle=True).item()

        # Plot f1_dev per epoch
        plt.plot(report["f1_macro_dev_5"])
        plt.xlabel("Epoch")
        plt.ylabel("f1_macro_dev_5")
        plt.savefig(f"{REPORT_PATH}/f1_dev_5.png")
        plt.clf()
        # Plot f1_dev per epoch
        plt.plot(report["f1_macro_dev_10"])
        plt.xlabel("Epoch")
        plt.ylabel("f1_macro_dev_10")
        plt.savefig(f"{REPORT_PATH}/f1_dev_10.png")
        plt.clf()
        # Plot f1_dev per epoch
        plt.plot(report["f1_macro_dev_20"])
        plt.xlabel("Epoch")
        plt.ylabel("f1_macro_dev_20")
        plt.savefig(f"{REPORT_PATH}/f1_dev_20.png")
        plt.clf()
        # Plot f1_dev per epoch
        plt.plot(report["f1_macro_dev_30"])
        plt.xlabel("Epoch")
        plt.ylabel("f1_macro_dev_30")
        plt.savefig(f"{REPORT_PATH}/f1_dev_30.png")
        plt.clf()
        # Plot f1_dev per epoch
        plt.plot(report["f1_macro_dev_50"])
        plt.xlabel("Epoch")
        plt.ylabel("f1_macro_dev_50")
        plt.savefig(f"{REPORT_PATH}/f1_dev_50.png")
        plt.clf()

        # Plot #selected instances per epoch
        plt.plot(report["nums_selected_avg_epoch"])
        plt.xlabel("Epoch")
        plt.ylabel("Mean Selected Tweets")
        plt.savefig(f"{REPORT_PATH}/avg_epoch_selected_{self.run_number}.png")
        plt.clf()

        # Plot total reward along episodes
        plt.plot(report["reward_total"])
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f"{REPORT_PATH}/total_reward_{self.run_number}.png")
        plt.clf()

        # Plot total reward for authors labeled with class high
        plt.plot(report["avg_reward_high"])
        plt.xlabel("Epoch")
        plt.ylabel("mean reward high")
        plt.savefig(f"{REPORT_PATH}/mean_reward_high_{self.run_number}.png")
        plt.clf()

        # Plot total reward for authors labeled with class low
        plt.plot(report["avg_reward_low"])
        plt.xlabel("Epoch")
        plt.ylabel("mean reward low")
        plt.savefig(f"{REPORT_PATH}/mean_reward_low_{self.run_number}.png")
        plt.clf()

        # Plot average reward per epoch
        plt.plot(report["rewards_avg_epoch"])
        plt.xlabel("Epoch")
        plt.ylabel("Mean Reward")
        plt.savefig(f"{REPORT_PATH}/avg_epoch_reward_{self.run_number}.png")
        plt.clf()

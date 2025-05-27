import os
import argparse


class ArgsParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--experiment_name", type=str, default='test')
        self.parser.add_argument("--save_name", type=str, default='test')
        self.parser.add_argument("--path_train", type=str, required=True)
        self.parser.add_argument("--path_valid", type=str, required=True)
        self.parser.add_argument("--path_test", type=str, required=True)

        self.parser.add_argument("--class_name1", type=str, required=True)
        self.parser.add_argument("--class_name2", type=str, required=True)
        self.parser.add_argument("--num_tweets", type=int, required=True)

        self.parser.add_argument("--learning_rate_rl", type=float, required=True)  # noqa: E501

        self.parser.add_argument("--inference_model_name", type=str, required=True)  # noqa: E501

        self.parser.add_argument("--instruction", type=str, required=True)
        self.parser.add_argument("--prompt_template", type=str, required=True)

        self.parser.add_argument("--bert_model_name", type=str, required=False)
        self.parser.add_argument("--bert_epochs", type=int, required=False)
        self.parser.add_argument("--rl_epochs", type=int, required=False)
        self.parser.add_argument("--bert_maxlen", type=int, required=False)
        self.parser.add_argument("--bert_learning_rate", type=float, required=False)  # noqa: E501

        self.parser.add_argument("--concept", type=str, required=False)
        self.parser.add_argument("--debug", type=int, default=0)
        self.parser.add_argument("--num_tweets_inference", type=int, default=5)

        self.parser.add_argument("--cuda_device", type=str, required=False)
        self.args = self.parser.parse_args()

        self.args.class_names = [self.args.class_name1, self.args.class_name2]

        outdir = "./results/" + self.args.save_name
        if not os.path.exists(outdir):
            print(f"Creating output_directory for experiment: {outdir}")
            os.makedirs(outdir)
        if not os.path.exists("./saved_models"):
            print("Creating output_directory for saved models:")
            os.makedirs("./saved_models")

    def print_args(self):
        print("*** Config ***")
        for arg in vars(self.args):
            print(f"\t{arg}", getattr(self.args, arg))
        print("***")

    def get_args(self):
        return self.args

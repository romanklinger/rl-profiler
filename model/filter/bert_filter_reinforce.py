import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from torch.distributions import Categorical

random.seed(42)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        text_orig = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                "text": text_orig,
                "text_model_input": text}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.bert.config.hidden_size,  # type: ignore
                             num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,            # type: ignore
                            attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        return F.softmax(x, dim=1)


class BERT_Filter():
    def __init__(self,
                 args,
                 load_pretrained=True,
                 model_save_path="./saved_models/bert_rl",
                 pre_train_save_path="./saved_models/bert",
                 epochs=1,
                 batch_size=8,
                 DEBUG=0):

        self.DEBUG = DEBUG
        self.tweets = []

        self.model_name = args.bert_model_name
        self.model_save_path = f"{model_save_path}_{args.save_name}.pth"
        self.pre_train_save_path = f"{pre_train_save_path}_{args.experiment_name}.pth"                # noqa: E501
        self.best_model_save_path_avg_reward = f"{model_save_path}_{args.save_name}_best_reward.pth"  # noqa: E501
        self.best_model_save_path_dev = f"{model_save_path}_{args.save_name}_best_dev.pth"            # noqa: E501
        self.best_model_save_path_dev_5 = f"{model_save_path}_{args.save_name}_best_dev_5.pth"        # noqa: E501
        self.best_model_save_path_dev_10 = f"{model_save_path}_{args.save_name}_best_dev_10.pth"      # noqa: E501
        self.best_model_save_path_dev_20 = f"{model_save_path}_{args.save_name}_best_dev_20.pth"      # noqa: E501
        self.best_model_save_path_dev_30 = f"{model_save_path}_{args.save_name}_best_dev_30.pth"      # noqa: E501
        self.best_model_save_path_dev_50 = f"{model_save_path}_{args.save_name}_best_dev_50.pth"      # noqa: E501
        self.best_model_save_path_test = f"{model_save_path}_{args.save_name}_best_test.pth"          # noqa: E501

        self.max_length = args.bert_maxlen
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = args.learning_rate_rl

        self.device = torch.device(args.cuda_device)
        self.model = BERTClassifier(self.model_name, 2).to(self.device)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.learning_rate,
                               no_deprecation_warning=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        self.actions = []
        self.actions_proba = []
        self.selected_tweets = []

        print("[BERT_FILTER_REINFORCE] INITIALIZING ...")
        if load_pretrained:
            print(
                f"[BERT_FILTER_REINFORCE] Loading pre-train model from: {self.pre_train_save_path}")  # noqa: E501

            self.model.load_state_dict(torch.load(self.pre_train_save_path))

    def load_best_model(self, id="none"):
        if id == "avg_reward":
            path = self.best_model_save_path_avg_reward
        elif id == "dev5":
            print("loading dev 5 model")
            path = self.best_model_save_path_dev_5
        elif id == "dev10":
            print("loading dev 10 model")
            path = self.best_model_save_path_dev_10
        elif id == "dev20":
            print("loading dev 20 model")
            path = self.best_model_save_path_dev_20
        elif id == "dev30":
            print("loading dev 30 model")
            path = self.best_model_save_path_dev_30
        elif id == "dev50":
            print("loading dev 50 model")
            path = self.best_model_save_path_dev_50
        else:
            print("EERRRROR Loading", id)
            import sys
            sys.exit(0)

        print(
            f"[BERT_FILTER_REINFORCE] Loading best model checkpoint from: {path}")  # noqa: E501

        self.model.load_state_dict(torch.load(
            path, map_location=torch.device(self.device)))

    def save_model(self, id="none"):
        print(f"[BERT_FILTER_REINFORCE] Saving Model Checkpoint {id}")
        if id == "avg_reward":
            torch.save(self.model.state_dict(),
                       self.best_model_save_path_avg_reward)
        elif id == "dev5":
            torch.save(self.model.state_dict(),
                       self.best_model_save_path_dev_5)
        elif id == "dev10":
            torch.save(self.model.state_dict(),
                       self.best_model_save_path_dev_10)
        elif id == "dev20":
            torch.save(self.model.state_dict(),
                       self.best_model_save_path_dev_20)
        elif id == "dev30":
            torch.save(self.model.state_dict(),
                       self.best_model_save_path_dev_30)
        elif id == "dev50":
            torch.save(self.model.state_dict(),
                       self.best_model_save_path_dev_50)
        elif id == "test":
            torch.save(self.model.state_dict(), self.best_model_save_path_test)
        else:
            torch.save(self.model.state_dict(), self.model_save_path)

    def backward(self, reward, baseline):
        loss = 0
        for log_prob in self.actions_proba:
            loss += - (reward-baseline) * log_prob

        loss.backward()  # type: ignore
        self.optimizer.step()

    def forward(self, author):
        self.tweets = author.tweets
        dataset = TextClassificationDataset(
            author.tweets, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        pred_actions = []
        self.actions = []
        self.actions_proba = []
        selected_tweets = []
        self.model.train()
        self.optimizer.zero_grad()
        for _, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pred_actions = self.model(
                input_ids=input_ids, attention_mask=attention_mask)
            for k, pred_action in enumerate(pred_actions):
                # Select an action from probability distribution
                m = Categorical(pred_action)
                action = m.sample()
                self.actions_proba.append(m.log_prob(action))
                self.actions.append(action)
                if action == 1:
                    selected_tweets.append(batch["text"][k])

        # i = 0
        # for action in self.actions:
        #     print(action)
        #     print(tweets[i])
        #     i += 1
        return selected_tweets

    def filter_author(self, author, num_tweets=10):
        if self.DEBUG:
            print(f"[BERT_FILTER_REINFORCE] Filtering Author-{author.id}")
        # self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        dataset = TextClassificationDataset(
            author.tweets, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)
        predictions = []
        tweets = []

        with torch.no_grad():
            for batch in dataloader:
                tweets.extend(batch["text"])
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                predictions.extend(outputs.cpu().tolist())

        preds_selected = []
        # print(predictions)
        for pred in predictions:
            preds_selected.append(pred[1])

        # Get idx of top-N instances
        selected_idx = sorted(range(len(preds_selected)),
                              key=lambda i: preds_selected[i])[-num_tweets:]

        selected_tweets = []
        selected_mask = []
        for i, tweet in enumerate(tweets):
            selected = 1 if i in selected_idx else 0
            selected_tweets.append(tweet) if selected else ""
            selected_mask.append(selected)
            if self.DEBUG:
                print(f"Tweet {i:2} : [SELECTED={selected}] [proba={preds_selected[i]:.4f}] | {tweet[:80]}")  # noqa: E501

        author.selected_tweets = selected_tweets
        author.selected_mask = selected_mask
        return selected_tweets

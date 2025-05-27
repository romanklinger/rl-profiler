import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

random.seed(42)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        text_orig = self.texts[idx]
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
                'label': torch.tensor(label),
                "text": text_orig,
                "text_model_input": text}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.bert.config.hidden_size,  # type: ignore
                             num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,            # type:ignore
                            attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        return F.softmax(x, dim=1)


class BERT_Filter():
    def __init__(self,
                 args,
                 class_weights=False,
                 model_save_path="./saved_models/bert",
                 batch_size=32,
                 DEBUG=0):

        self.DEBUG = DEBUG

        self.model_name = args.bert_model_name
        self.model_save_path = f"{model_save_path}_{args.experiment_name}.pth"  # noqa: E501

        self.max_length = args.bert_maxlen
        self.epochs = args.bert_epochs
        self.batch_size = batch_size
        self.learning_rate = args.bert_learning_rate

        print("[BERT_FILTER] INITIALIZING ...")
        print(f"[BERT_FILTER] Save_path: {self.model_save_path}")
        print(f"[BERT_FILTER] Class Weights: {class_weights}")

        self.device = torch.device(
            args.cuda_device if torch.cuda.is_available() else "cpu")
        self.model = BERTClassifier(self.model_name, 2).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        if class_weights:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=self.learning_rate,
                                   no_deprecation_warning=True)
            class_weights = torch.tensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

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
                        input_ids=input_ids,
                        attention_mask=attention_mask)
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

        acc = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions)
        return acc, report

    def test(self, X_test, y_test):
        if self.DEBUG:
            print("[BERT_FILTER] Testing BERT-Filter Selection")

        self.model.load_state_dict(torch.load(self.model_save_path,
                                              weights_only=True))
        self.model.eval()

        dataset = TextClassificationDataset(
            X_test, y_test, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
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

        acc = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions)
        return acc, report

    def filter_author(self, author, num_tweets=5):
        if self.DEBUG:
            print(f"[BERT_FILTER] Filtering Author-{author.id}")

        self.model.load_state_dict(torch.load(self.model_save_path,
                                              weights_only=True))
        self.model.eval()

        predictions = []
        tweets = []
        dataset = TextClassificationDataset(
            author.tweets, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.no_grad():
            for batch in dataloader:
                tweets.extend(batch["text"])
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                predictions.extend(outputs.cpu().tolist())

        preds_selected = []
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

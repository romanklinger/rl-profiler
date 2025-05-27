# RL-Profiler

This repository contains the code accompanying the paper **"Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering"**.

(Under review for the 31st International Conference on Computational Linguistics (COLING 2025))

## Environment Setup
To reproduce the results, we recommend using `conda` for environment management.
```bash
# Create a new conda environment
conda create --name rlp
# Activate the environment
conda activate rlp
# Install Python and dependencies
conda install -y python==3.11.5
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers accelerate pandas nltk scikit-learn auto-gptq optimum matplotlib bs4 lxml
```

## Dataset Preparation
To prepare the "PAN15 Author Profiling" dataset for the experiments follow the steps below:
1. Download the dataset from [https://zenodo.org/records/3745945](https://zenodo.org/records/3745945).
2. Place unpacked dataset in the `./data/PAN15/` directory.
3. Run the `create_datasets.sh` script in the `./data/PAN15/` directory.
You may need to adapt the following variables in this script:
```bash
# Path to training profiles and corresponding labels
TRAIN_DATA="./pan15-author-profiling-training-dataset-english-2015-04-23/"
TRAIN_LABELS="./pan15-author-profiling-training-dataset-english-2015-04-23/truth.txt"

# Path to testing profiles and corresponding labels
TEST_DATA="./pan15-author-profiling-test-dataset2-english-2015-04-23/"
TEST_LABELS="./pan15-author-profiling-test-dataset2-english-2015-04-23/truth.txt"
```
This will create training, validation and testing splits for each of the Big Five personality traits.

We also provide a mockup dataset in `./data/example_dataset/extraversion/` consisting of generated posts only that does not require downloading the "PAN15 Author Profiling" dataset.


## Adding BFI-Items
We use "The Big Five Inventory" (BFI-44) to enrich context in prompts. 
Items from this inventory need to be added to the prompt templates before running our experiments.
We exemplarily show how to add these items to prompts for the *extraversion* trait:
1. Download the BFI-44-Inventory from [https://www.ocf.berkeley.edu/~johnlab/bfi.htm](https://www.ocf.berkeley.edu/~johnlab/bfi.htm)
2. Add BFI-Item 1, 11, 16, 26, 36 to `TRAIT_DESCRIPTION_HIGH` in `./run.sh`.
3. Add BFI-Item 6R, 21R, 31R to `TRAIT_DESCRIPTION_LOW` in `./run.sh`.
```bash
TRAIT="extraversion"
TRAIT_DESCRIPTION_HIGH="is talkative, ..." # BFI-Item 1, 11, 16, 26, 36
TRAIT_DESCRIPTION_LOW="is reserved, ..."   # BFI-Item 6R, 21R, 31R
```
Similarly, items for the other traits can be added in `./run.sh`.

## Reproducing Specific Experiments
To reproduce experiments on specific dataset splits and traits uncomment
one of the following settings:
```bash
### Run on example dataset.
# PATH_TRAIN="./data/example_dataset/$TRAIT/train.csv"
# PATH_VALID="./data/example_dataset/$TRAIT/valid.csv"
# PATH_TEST="./data/example_dataset/$TRAIT/test.csv"

### Run on PAN15 data.
# PATH_TRAIN="./data/PAN15/$TRAIT/train.csv"
# PATH_VALID="./data/PAN15/$TRAIT/valid.csv"
# PATH_TEST="./data/PAN15/$TRAIT/test.csv"

### Run on PAN15 data enriched with artificial tweets.
# PATH_TRAIN="./data/PAN15/${TRAIT}_enriched/train.csv"
# PATH_VALID="./data/PAN15/${TRAIT}_enriched/valid.csv"
# PATH_TEST="./data/PAN15/${TRAIT}_enriched/test.csv"
```

## Training and Evaluating RL-Profiler
The general workflow of RL-Profiler is divided into the following steps:
1. Finetune BERT (pretrain RL agent) on NPMI-Annotations:
```bash
./run.sh pretrain
```
2. Train using Reinforcement Learning:
```bash
./run.sh train_reinforce
```
3. Validate Trained Model on Validation Data:
```bash
./run.sh validate
```
4. Evaluate Trained Model on Testing Data:
```bash
./run.sh evaluate
```
5. Train/Validate/Test Baseline Models:
```bash
./run.sh baseline
```

## Repository Structure
```
├── data
│   ├── example_dataset
│   │   └── extraversion            # Mockup dataset for predicting levels of Extraversion
│   └── generated_tweets            # Artificial posts generated using Llama 2
├── model
│   ├── baseline.py                 # Code for supervised-learning based baseline models
│   ├── evaluate.py
│   ├── filter                     # Selection Network (SelNet)
│   │   ├── bert_filter.py             # Code for BERT model learning to replicate NPMI-Annotations
│   │   ├── bert_filter_reinforce.py   # Code for RL agent learning to select relevant instances
│   │   ├── pmi_filter.py              # Code for choosing instances using NPMI-Scores
│   │   └── random_filter.py           # Code for choosing instances at random
│   ├── predictor                 # Classification Network (CNet)
│   │   ├── llm.py                    # LLM (Llama 2) configuration
│   ├── pretrain_bert.py            # Code for fine-tuning BERT on NPMI-Annotations
│   ├── train_reinforce.py          # Code for training loop using reinforcement learning
│   └── util
│       ├── reward.py               # Reward calculcation
```

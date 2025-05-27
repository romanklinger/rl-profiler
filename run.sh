###############################################################################
# Extraversion
###############################################################################
TRAIT="extraversion"
TRAIT_DESCRIPTION_HIGH="is talkative, ..."
TRAIT_DESCRIPTION_LOW="is reserved, ..."

###############################################################################
# Agreeableness
###############################################################################
# TRAIT="agreeableness"
# TRAIT_DESCRIPTION_HIGH="is helpful and unselfish with others, ..."
# TRAIT_DESCRIPTION_LOW="tends to find fault with others, ..."

###############################################################################
# Conscientiousness
###############################################################################
# TRAIT="conscientiousness"
# TRAIT_DESCRIPTION_HIGH="does a thorough job, ..."
# TRAIT_DESCRIPTION_LOW="can be somewhat careless, ..."

###############################################################################
# Neuroticism
###############################################################################
# TRAIT="neuroticism"
# TRAIT_DESCRIPTION_HIGH="is depressed or blue, ..."
# TRAIT_DESCRIPTION_LOW="is relaxed or handles stress well, ..."

###############################################################################
# Openness
###############################################################################
# TRAIT="openness"
# TRAIT_DESCRIPTION_HIGH="is original and comes up with new ideas, ..."
# TRAIT_DESCRIPTION_LOW="prefers work that is routine, ..."

###############################################################################
# Dataset Settings
###############################################################################
### Run on example dataset
PATH_TRAIN="./data/example_dataset/$TRAIT/train.csv"
PATH_VALID="./data/example_dataset/$TRAIT/valid.csv"
PATH_TEST="./data/example_dataset/$TRAIT/test.csv"

### Run on PAN15 data
# PATH_TRAIN="./data/PAN15/$TRAIT/train.csv"
# PATH_VALID="./data/PAN15/$TRAIT/valid.csv"
# PATH_TEST="./data/PAN15/$TRAIT/test.csv"

### Run on PAN15 data enriched with artificial tweets
# PATH_TRAIN="./data/PAN15/${TRAIT}_enriched/train.csv"
# PATH_VALID="./data/PAN15/${TRAIT}_enriched/valid.csv"
# PATH_TEST="./data/PAN15/${TRAIT}_enriched/test.csv"
###############################################################################

#
###############################################################################
DEBUG=0
CUDA_DEVICE="cuda"

CLASS_NAME1="low"
CLASS_NAME2="high"

BERT_MODEL_NAME="bert-base-uncased"
BERT_EPOCHS=2
BERT_LEARNING_RATE=1e-06
BERT_MAXLEN=64
NUM_TWEETS_PMI_PRETRAIN=10

INFERENCE_MODEL_NAME="TheBloke/Llama-2-13B-chat-GPTQ"

RL_EPOCHS=200
RL_LEARNING_RATE=1e-6
NUM_TWEETS_INFERENCE=10

PROMPT_TEMPLATE="<s>[INST] <<SYS>>
one word response
<</SYS>>

Recall the personality trait $TRAIT.
A person with a high level of $TRAIT may see themselves as someone who $TRAIT_DESCRIPTION_HIGH.
A person with a low level of $TRAIT may see themselves as someone who $TRAIT_DESCRIPTION_LOW.

Consider the following tweets written by the same person:
{tweets}
{instruction} [/INST]"
INSTRUCTION="Does this person show a low or high level of $TRAIT? Do not give an explanation."
###############################################################################
#

EXPERIMENTNAME="$TRAIT"
SAVE_NAME="$EXPERIMENTNAME-$RL_LEARNING_RATE"

if [ "$1" == "pretrain" ]; then
  runner=pretrain_bert.py
elif [ "$1" == "train_reinforce" ]; then
  runner=train_reinforce.py
elif [ "$1" == "validate" ]; then
  PATH_TEST=$PATH_VALID
  runner=evaluate.py
elif [ "$1" == "evaluate" ]; then
  runner=evaluate.py
elif [ "$1" == "baseline" ]; then
  runner=baseline.py
else
  echo "No task specified. [Possible tasks: pretrain, train_reinforce, validate, evaluate, baseline]"
  exit -1
fi

python model/$runner \
  --experiment_name "$EXPERIMENTNAME" \
  --save_name "$SAVE_NAME" \
  --path_train "$PATH_TRAIN" \
  --path_valid "$PATH_VALID" \
  --path_test "$PATH_TEST" \
  --class_name1 "$CLASS_NAME1" \
  --class_name2 "$CLASS_NAME2" \
  --num_tweets "$NUM_TWEETS_PMI_PRETRAIN" \
  --num_tweets_inference "$NUM_TWEETS_INFERENCE" \
  --learning_rate_rl "$RL_LEARNING_RATE" \
  --rl_epochs "$RL_EPOCHS" \
  --inference_model_name "$INFERENCE_MODEL_NAME" \
  --instruction "$INSTRUCTION" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --cuda_device "$CUDA_DEVICE" \
  --bert_model_name "$BERT_MODEL_NAME" \
  --bert_maxlen "$BERT_MAXLEN" \
  --bert_learning_rate "$BERT_LEARNING_RATE" \
  --bert_epochs "$BERT_EPOCHS" \
  --concept "$TRAIT" \
  --debug "$DEBUG"

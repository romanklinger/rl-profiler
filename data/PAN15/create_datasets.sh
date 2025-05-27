# Path to training profiles and ground-truth labels
TRAIN_DATA="./pan15-author-profiling-training-dataset-english-2015-04-23/"
TRAIN_LABELS="./pan15-author-profiling-training-dataset-english-2015-04-23/truth.txt"

# Path to testing profiles and ground-truth labels
TEST_DATA="./pan15-author-profiling-test-dataset2-english-2015-04-23/"
TEST_LABELS="./pan15-author-profiling-test-dataset2-english-2015-04-23/truth.txt"

declare -a traits=("openness" "conscientiousness" "extraversion" "agreeableness" "neuroticism")
for trait in "${traits[@]}"; do
  echo "Creating Dataset Splits for Trait: $trait"
  python collect_pan15_dataset.py \
    --trait $trait \
    --input_train_data $TRAIN_DATA \
    --input_train_labels $TRAIN_LABELS \
    --input_test_data $TEST_DATA \
    --input_test_labels $TEST_LABELS
  echo "------------------------------------------"
  echo "Creating --Enriched-- Dataset Splits for Trait: $trait"
  python collect_pan15_dataset_enriched.py \
    --trait $trait
  echo "------------------------------------------"
done

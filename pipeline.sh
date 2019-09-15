#!/usr/bin/env bash

python3 1_clean_train_subset.py
python3 2_clean_test_subset.py
python3 3_create_tokenizer_spanish.py
python3 4_create_tokenizer_portuguese.py
python3 5_create_trainset_spanish.py
python3 6_create_trainset_portuguese.py


#!/usr/bin/env bash

# nohup bash ./test_models.sh > test_models_log.out &

source /home/juan_francisco_chango/envs/tf-gpu/bin/activate

python3 14_test_model_rich_cnn_lstm_portuguese.py
python3 13_test_model_rich_cnn_lstm_spanish.py


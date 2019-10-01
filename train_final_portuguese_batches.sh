#!/usr/bin/env bash

# nohup bash ./train_portuguese_batches.sh > portuguese_dense_last_log.out &

source /home/juan_francisco_chango/envs/tf-gpu/bin/activate

python3 12_continue_train_model_rich_cnn_lstm_portuguese.py 1 0.0001
python3 12_continue_train_model_rich_cnn_lstm_portuguese.py 2 0.0001
#python3 12_continue_train_model_rich_cnn_lstm_portuguese.py 3 0.0001
#python3 12_continue_train_model_rich_cnn_lstm_portuguese.py 4 0.0001
#python3 12_continue_train_model_rich_cnn_lstm_portuguese.py 5 0.0001

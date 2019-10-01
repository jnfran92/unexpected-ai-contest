#!/usr/bin/env bash

# nohup bash ./train_spanish_batches.sh > spanish_dense_last_log.out &

source /home/juan_francisco_chango/envs/tf-gpu/bin/activate

python3 9_train_model_rich_cnn_lstm_spanish.py
python3 10_continue_train_model_rich_cnn_lstm_spanish.py 1 0.0001
python3 10_continue_train_model_rich_cnn_lstm_spanish.py 2 0.0001
#python3 10_continue_train_model_rich_cnn_lstm_spanish.py 3 0.0001
#python3 10_continue_train_model_rich_cnn_lstm_spanish.py 4 0.0001
#python3 10_continue_train_model_rich_cnn_lstm_spanish.py 5 0.0001

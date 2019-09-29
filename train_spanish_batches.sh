#!/usr/bin/env bash

#nohup bash ./train_spanish_batches.sh > spanish_dense_log.out &

source /home/juan_francisco_chango/envs/tf-gpu/bin/activate
python3 10_continue_train_model_rich_cnn_lstm_spanish.py 1 0.0001
python3 10_continue_train_model_rich_cnn_lstm_spanish.py 2 0.00001
python3 10_continue_train_model_rich_cnn_lstm_spanish.py 3 0.000001
python3 10_continue_train_model_rich_cnn_lstm_spanish.py 4 0.0000001
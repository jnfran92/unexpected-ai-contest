#!/usr/bin/env bash

# nohup bash ./final_final_final.sh > final_final_final_log.out &

source /home/juan_francisco_chango/envs/tf-gpu/bin/activate

python3 12_1_continue_train_model_rich_cnn_lstm_portuguese.py 0 6 7 0.0001
python3 10_1_continue_train_model_rich_cnn_lstm_spanish.py 0 8 9 0.0001


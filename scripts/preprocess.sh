#!/bin/bash
DATA_DIR=$1
python preprocess.py -train_src $DATA_DIR/src-train.txt -train_tgt $DATA_DIR/tgt-train.txt -valid_src $DATA_DIR/src-val.txt -valid_tgt $DATA_DIR/tgt-val.txt -save_data $DATA_DIR/demo

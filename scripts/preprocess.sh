#!/bin/bash
DATA_DIR=$1
if [ "$DATA_DIR" = "" ] ; then
    echo "Specify data dir. (1st arg)"
    exit 1
fi
if [ ! -d "$DATA_DIR" ] ; then
    echo "Not a directory (1st arg)"
    exit 1
fi
TRG_PREFIX=$2
if [ "$TRG_PREFIX" = "" ] ; then
    TRG_PREFIX="procData"
fi


TGT_VOC_SIZE=100000

./preprocess.py -tgt_vocab_size $TGT_VOC_SIZE \
                -train_src $DATA_DIR/src-train.txt \
                -train_tgt $DATA_DIR/tgt-train.txt \
                -valid_src $DATA_DIR/src-val.txt \
                -valid_tgt $DATA_DIR/tgt-val.txt \
                -tgt_vocab_size $TGT_VOC_SIZE \
                -save_data $DATA_DIR/$TRG_PREFIX

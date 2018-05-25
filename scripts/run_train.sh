#!/usr/bin/env bash
DATA_DIR=$1
if [ "$DATA_DIR" = "" ] ; then
    echo "Specify data dir. (1st arg)"
    exit 1
fi
if [ ! -d "$DATA_DIR" ] ; then
    echo "Not a directory (1st arg)"
    exit 1
fi
MODEL_DIR=$2
if [ "$MODEL_DIR" = "" ] ; then
    echo "Specify DATA model dir. (1d arg)"
    exit 1
fi
if [ ! -d "$MODEL_DIR" ] ; then
    echo "Not a directory (2d arg)"
    exit 1
fi
COMP_OPT=$3
if [ "$COMP_OPT" = "" ] ; then
    echo "Specify a composition option (3rd arg)"
    exit 1
fi
EPOCH_QTY=$4
if [ "$EPOCH_QTY" = "" ] ; then
    echo "Specify the number of epoch (4th arg)"
    exit 1
fi
PREVIOUS_MODEL=$5
if [ "PREVIOUS_MODEL" != "" ] ; then
    if [ ! -f "$PREVIOUS_MODEL" ] ; then
        echo "Not a file (5th arg): $PREVIOUS_MODEL"
    fi
    TRAIN_FROM="-train_from $PREVIOUS_MODEL"
    echo "Reusing the model: $PREVIOUS_MODEL"
fi

GPU_ID=$6
GPU_OPT=""
if [ "$GPU_ID" != "" ] ; then
    echo "Using GPU id $GPU_ID"
    GPU_OPT=" -gpuid $GPU_ID"
fi
TRG_PREFIX="procData"

START_LR=0.1
NESTEROV="-nesterov"
MOMENTUM="-momentum 0.9"
DROPOUT="0.3"

# All combinations will have the same total number of dimensions 600
EMBED_SIZE=`scripts/comp_embed_size.py $COMP_OPT 600` # one method dim=600 (total)

echo "Using $EMBED_SIZE for each component model for composition option $COMP_OPT"

CHAN_QTY=100
RNN_LAYER_QTY=2

python -u train.py $GPU_OPT \
        $TRAIN_FROM \
        -data $DATA_DIR/$TRG_PREFIX \
        -save_model $MODEL_DIR/$COMP_OPT \
        -optim sgd -learning_rate $START_LR $MOMENTUM $NESTEROV -dropout $DROPOUT \
        -epochs $EPOCH_QTY \
        -tgt_word_vec_size $EMBED_SIZE \
        -char_comp_cnn_chan_qty $CHAN_QTY \
        -char_comp_rnn_layer $RNN_LAYER_QTY \
        -char_compos_type $COMP_OPT \

if [ "$?" != "0" ] ; then
    echo "Failed"
    exit 1
fi

echo "Success!"
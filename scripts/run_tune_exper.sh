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
    echo "Specify model dir. (1d arg)"
    exit 1
fi

TGT_VOC_SIZE=100000
START_LR=0.1
NESTEROV="-nesterov"
MOMENTUM="-momentum 0.9"
DROPOUT="0.3"

# All combinations will have the same total number of dimensions (see EMBED_SIZE2 and EMBED_SIZE3 below)
EMBED_SIZE1=600 # one method dim=600 (total)

CHAN_QTY=100
RNN_LAYER_QTY=2
EPOCH_QTY=2

for comp_opt in none brnn cnn ; do
    python -u train.py \
            -data $DATA_DIR/demo
            -save_model $MODEL_DIR/$comp_opt \
            -optim sgd -learning_rate $LR $MOMENTUM $NESTEROV -dropout $DROPOUT \
            -epochs $EPOCH_QTY \
            -tgt_vocab_size $TGT_VOC_SIZE \
            -tgt_word_vec_size $EMBED_SIZE1 \
            -char_comp_cnn_chan_qty $CHAN_QTY \
            -char_comp_rnn_layer $RNN_LAYER_QTY \
            -char_compos_type $comp_opt \

    if [ "$?" != "0" ] ; then
        echo "Failed"
        exit 1
    fi
done

EMBED_SIZE2=300 # two methods 300*2=600 (total)

for comp_opt in brnn,cnn brnn,wembed ; do
    python -u train.py \
            -data $DATA_DIR/demo
            -save_model $MODEL_DIR/$comp_opt \
            -optim sgd -learning_rate $LR $MOMENTUM $NESTEROV -dropout $DROPOUT \
            -epochs $EPOCH_QTY \
            -tgt_vocab_size $TGT_VOC_SIZE \
            -tgt_word_vec_size $EMBED_SIZE2 \
            -char_comp_cnn_chan_qty $CHAN_QTY \
            -char_comp_rnn_layer $RNN_LAYER_QTY \
            -char_compos_type $comp_opt \

    if [ "$?" != "0" ] ; then
        echo "Failed"
        exit 1
    fi
done

EMBED_SIZE3=200 # three methods 200*3=600 (total)

for comp_opt in brnn,cnn,wembed ; do
    python -u train.py \
            -data $DATA_DIR/demo
            -save_model $MODEL_DIR/$comp_opt \
            -optim sgd -learning_rate $LR $MOMENTUM $NESTEROV -dropout $DROPOUT \
            -epochs $EPOCH_QTY \
            -tgt_vocab_size $TGT_VOC_SIZE \
            -tgt_word_vec_size $EMBED_SIZE3 \
            -char_comp_cnn_chan_qty $CHAN_QTY \
            -char_comp_rnn_layer $RNN_LAYER_QTY \
            -char_compos_type $comp_opt \

    if [ "$?" != "0" ] ; then
        echo "Failed"
        exit 1
    fi
done
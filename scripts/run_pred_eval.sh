#!/usr/bin/env bash
DATA_DIR=$1
if [ "$DATA_DIR" = "" ] ; then
    echo "Specify data dir. (1st arg)"
    exit 1
fi
if [ ! -d "$DATA_DIR" ] ; then
    echo "Not a directory: $DATA_DIR (1st arg)"
    exit 1
fi
MODEL_DIR=$2
if [ "$MODEL_DIR" = "" ] ; then
    echo "Specify DATA model dir. (2d arg)"
    exit 1
fi
if [ ! -d "$MODEL_DIR" ] ; then
    echo "Not a directory: $MODEL_DIR (2d arg)"
    exit 1
fi
PRED_DIR=$3
if [ "$PRED_DIR" = "" ] ; then
    echo "Specify prediction model dir. (3rd arg)"
    exit 1
fi
if [ ! -d "$PRED_DIR" ] ; then
    echo "Not a directory: $PRED_DIR (3rd arg)"
    exit 1
fi
COMP_OPT=$4
if [ "$COMP_OPT" = "" ] ; then
    echo "Specify a composition option (4th arg)"
    exit 1
fi
EPOCH_QTY=$5
if [ "$EPOCH_QTY" = "" ] ; then
    echo "Specify the number of epoch (5th arg)"
    exit 1
fi

TRG_PREFIX="procData"


MODEL_FULL_PATH=`ls ${MODEL_DIR}/${COMP_OPT}_*_e${EPOCH_QTY}.pt`
if [ "$?" != "0" ] ; then
    echo "Error finding the model or the model is not found!"
    exit 1
fi

PRED_FILE=$PRED_DIR/${COMP_OPT}.pred
./translate.py -model "$MODEL_FULL_PATH" -src "$DATA_DIR/src-val.txt" -output "$PRED_FILE" -replace_unk -verbose
tools/multi-bleu-detok.perl "$DATA_DIR/tgt-val.txt" < "$PRED_FILE"
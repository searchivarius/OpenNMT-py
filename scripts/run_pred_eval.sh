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

TRG_PREFIX="procData"


MODEL_FULL_PATH=`ls ${MODEL_DIR}/${COMP_OPT}_*_e${EPOCH_QTY}.pt`
if [ "$?" != "0" ] ; then
    echo "Error finding the model or the model is not found!"
    exit 1
fi

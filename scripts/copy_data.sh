#!/bin/bash
SRC_DIR=$1
TRG_DIR=$2
LANG1=$3
LANG2=$4

for part1 in train dev test1 ; do
  if [ "$part1" = "dev" ] ; then
    part2="val"
  elif [ "$part1" = "test1" ] ; then
    part2="test"
  else
    part2=$part1
  fi
  if [ "$part1" = "train" ] ; then
    add="clean."
  else
    add=""
  fi
  cp $SRC_DIR/ted_${part1}_${LANG1}-${LANG2}.tok.${add}$LANG1 $TRG_DIR/src-${part2}.txt
  cp $SRC_DIR/ted_${part1}_${LANG1}-${LANG2}.tok.${add}$LANG2 $TRG_DIR/tgt-${part2}.txt
done

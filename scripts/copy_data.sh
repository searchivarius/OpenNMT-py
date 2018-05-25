#!/bin/bash
# Copy data from TED-talk folders to
# our folders and rename to make standard names
SRC_DIR=$1
if [ "SRC_DIR" = "" ] ; then
    echo "Specify src data dir. (1st arg)"
    exit 1
fi
if [ ! -d "SRC_DIR" ] ; then
    echo "Not a directory (1st arg)"
    exit 1
fi
TRG_DIR=$2
if [ "TRG_DIR" = "" ] ; then
    echo "Specify dst data dir. (2d arg)"
    exit 1
fi
if [ ! -d "TRG_DIR" ] ; then
    echo "Not a directory (2d arg)"
    exit 1
fi
LANG1=$3
if [ "$LANG1" = "" ] ; then
    echo "Specify src lang (3rd arg)"
    exit 1
fi
LANG2=$4
if [ "$LANG2" = "" ] ; then
    echo "Specify trg lang (4th arg)"
    exit 1
fi

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

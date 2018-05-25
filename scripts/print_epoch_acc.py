#!/usr/bin/env python

import sys

acc = []
perpl = []


with open(sys.argv[1]) as f:
  ln=0
  for line in f:
    line = line.strip()
    ln += 1
    if line.startswith('Validation perplexity:'):
      if len(acc) != len(perpl):
        raise Exception('Bad file format, unbalanced val. perpl. msg in line %d' % ln)

      perpl.append(line.split(':')[1].strip())
    if line.startswith('Validation accuracy:'):
      if len(acc) != len(perpl) - 1:
        raise Exception('Bad file format, unbalanced val. acc. msg in line %d' % ln)
      acc.append(line.split(':')[1].strip())


epochs = len(acc)
print('\t'.join([str(i+1) for i in range(epochs)]))
print('\t'.join(acc))
print('\t'.join(perpl))
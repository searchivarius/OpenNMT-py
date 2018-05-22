import torch
import numpy as np

SPECIAL_TOKEN_LIST = ['<unk>', '<blank>', '<s>', '</s>']
SPECIAL_TOKENS = set(SPECIAL_TOKEN_LIST)
ONE_WORD_TOKENS = set(['&quot;', '...', '&apos;'])

def wordToChars(s):
  if s in SPECIAL_TOKENS or s in ONE_WORD_TOKENS:
    return [s]
  else:
    return [c for c in s]

def getVocabSpell(voc, padIdx = 0):
  char2idx = {t : i for i, t in enumerate(SPECIAL_TOKEN_LIST)}

  maxLen = 0

  for word in voc.itos:
    maxLen = max(len(wordToChars(word)), maxLen)

  wordQty = len(voc.itos)

  word2chars = np.full((wordQty, maxLen + 1), padIdx)

  for wi, word in enumerate(voc.itos):
    charList = wordToChars(word)
    word2chars[wi, -1] = len(charList)
    assert(word2chars[wi, -1] <= word2chars.shape[1] - 1)
    for ci, char in enumerate(charList):
      if char not in char2idx:
        char2idx[char] = len(char2idx)
      word2chars[wi, ci] = char2idx[char]

  return torch.LongTensor(word2chars), len(char2idx)

def initWeights(m):
  """
  Simple initializer
  """
  if hasattr(m, 'weight'):
    torch.nn.init.xavier_uniform(m.weight.data)
  if hasattr(m, 'bias'):
    m.bias.data.zero_()


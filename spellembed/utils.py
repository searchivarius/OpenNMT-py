import torch
import numpy as np

SPECIAL_TOKEN_LIST = ['<unk>', '<blank>', '<s>', '</s>']
SPECIAL_TOKENS = set(SPECIAL_TOKEN_LIST)
ONE_WORD_TOKENS = set(['&quot;', '...', '&apos;', '-lrb-', '-rrb-'])

def wordToChars(s):
  if s in SPECIAL_TOKENS or s in ONE_WORD_TOKENS:
    return [s]
  else:
    return [c for c in s]

def deviceMap(x, isCuda):
  return x.cuda() if isCuda else x.cpu()

def getVocabSpell(voc, isCuda, padIdx = 0):
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

  return deviceMap(torch.LongTensor(word2chars), isCuda), \
         len(char2idx)

def initWeights(m):
  """
  Simple initializer
  """
  if hasattr(m, 'weight'):
    torch.nn.init.xavier_uniform(m.weight.data)
  if hasattr(m, 'bias'):
    m.bias.data.zero_()


def tensorSort(v):
  assert len(v.shape) == 1
  sorted_v, ind_v = torch.sort(v, 0, descending=True)
  return sorted_v, ind_v

def tensorUnsort(sorted_v, sort_idx):
  unsorted_v = torch.zeros_like(sorted_v)
  unsorted_v[sort_idx] = sorted_v
  return unsorted_v

def tensorToList(v):
  if v.is_cuda:
    v_list = v.cpu().numpy().tolist()
  else:
    v_list = v.numpy().tolist()
  return v_list


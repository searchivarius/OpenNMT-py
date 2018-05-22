import torch

SPECIAL_TOKEN_LIST = ['<unk>', '<blank>', '<s>', '</s>']
SPECIAL_TOKENS = set(SPECIAL_TOKEN_LIST)
ONE_WORD_TOKENS = set(['&quot;', '...', '&apos;'])

def wordToChars(s):
  if s in SPECIAL_TOKENS or s in ONE_WORD_TOKENS:
    return s
  else:
    return [c for c in s]

def getVocabSpell(voc, padIdx):
  char2idx = {t : i for i, t in enumerate(SPECIAL_TOKEN_LIST)}

  maxLen = 0

  for word in voc.itos:
    maxLen = max(wordToChars(word), maxLen)

  wordQty = len(voc.itos)

  word2chars = torch.LongTensor(wordQty, maxLen + 1).fill_(padIdx)

  for wi, word in enumerate(voc.itos):
    charList = wordToChars(word)
    word2chars[wi, -1] = len(charList)
    assert(word2chars[wi, -1] <= word2chars.size()[1] - 1)
    for ci, char in enumerate():
      if char not in char2idx:
        char2idx[char] = len(char2idx)
      word2chars[wi, ci] = char2idx[char]

  return word2chars


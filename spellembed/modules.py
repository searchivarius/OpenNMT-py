import torch

from torch.nn import Module, Dropout, LSTM, Embedding

# This is modelled after the following code:
# https://github.com/arendu/pytorch-sgns/blob/master/model.py

from spellembed.utils import *

class Char2VecRNN(Module):
  def __init__(self,
               vocabSpell,
               wordEmbedSize=300,
               charEmbedSize=20,
               padIdx=0,
               dropout=0.3,
               numLayers=1,
               bidir=False):
    super(Char2VecRNN, self).__init__()
    self.word2chars, charVocabSize = vocabSpell
    self.dropout = Dropout(dropout)
    self.charEmbed = Embedding(charVocabSize, charEmbedSize, padding_idx=padIdx)
    initWeights(self.charEmbed)
    self.charRnn = LSTM(charEmbedSize, wordEmbedSize, num_layers=numLayers,
                        dropout=dropout, bidirectional=bidir)
    initWeights(self.charRnn)

  def forward(self, data):
    isCuda = self.is_cuda
    data = deviceMap(data, isCuda)
    dataIdxs = deviceMap(torch.arange(data.size(0)).long(), isCuda)
    spellData = self.word2chars[dataIdxs]
    chars = spellData[:, :-1]
    lens = spellData[:, -1]
    
    return self.batchRnn(chars, lens)








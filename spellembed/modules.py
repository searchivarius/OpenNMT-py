import torch

from torch.nn import Module, Dropout, LSTM, Embedding
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack

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
               isBidir=False):
    super(Char2VecRNN, self).__init__()
    self.word2chars, charVocabSize = vocabSpell
    self.dropout = Dropout(dropout)
    self.charEmbed = Embedding(charVocabSize, charEmbedSize, padding_idx=padIdx)
    initWeights(self.charEmbed)
    self.charRnn = LSTM(charEmbedSize, wordEmbedSize, num_layers=numLayers,
                        dropout=dropout, bidirectional=isBidir)
    initWeights(self.charRnn)

  def batchRnn(self, charTensor, lenTensor):
    sortedLens, sortedLenIdx = tensorSort(lenTensor)
    sortedData = Variable(charTensor[sortedLenIdx])
    sortedEmbeds = self.charEmbed(sortedData)
    sortedEmbedsPacked = pack(sortedEmbeds,
                              tensorToList(sortedLens),
                              batch_first=True)
    output, (ht, ct) = self.charRnn(sortedEmbedsPacked, None)
    del output, ct

    if self.charRnn.bidirectional:
      # concat the last ht from fwd RNN and first ht from bwd RNN
      ht = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1)
    else:
      ht = ht.squeeze(dim=1)

    htUnsorted = tensorUnsort(ht, sortedLenIdx)

    del sortedData, sortedLens, sortedLenIdx, sortedEmbeds, sortedEmbedsPacked, ht

    return htUnsorted

  def forward(self, data):
    isCuda = self.is_cuda
    data = deviceMap(data, isCuda)
    dataIdxs = deviceMap(torch.arange(data.size(0)).long(), isCuda)
    spellData = self.word2chars[dataIdxs]
    charTensor = spellData[:, :-1]
    lenTensor = spellData[:, -1]

    return self.batchRnn(charTensor, lenTensor)








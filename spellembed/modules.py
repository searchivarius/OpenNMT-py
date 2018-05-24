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
               wordEmbedSize,
               charEmbedSize=20,
               padIdx=0,
               dropout=0.3,
               numLayers=1,
               isBidir=False):
    super(Char2VecRNN, self).__init__()
    self.word2chars, charVocabSize = vocabSpell
    self.charEmbedSize = charEmbedSize
    self.dropout = Dropout(dropout)
    self.charEmbed = Embedding(charVocabSize, charEmbedSize, padding_idx=padIdx)
    initWeights(self.charEmbed)
    self.embedding_size = wordEmbedSize
    self.charRnn = LSTM(charEmbedSize, wordEmbedSize, num_layers=numLayers,
                        dropout=dropout, bidirectional=isBidir)
    for w in self.charRnn.all_weights:
      initWeights(w)

  def batchRnn(self, charTensor, lenTensor):
    sentLen, batchQty, maxWordLen = charTensor.size()
    charVar = Variable(charTensor.contiguous().view(-1))
    charEmbeds = self.charEmbed(charVar).data # sentLen x B x wordLen x charEmbedDim
    charEmbeds = charEmbeds.view(sentLen * batchQty, maxWordLen, -1)

    lenTensor = lenTensor.contiguous().view(-1)
    assert(lenTensor.size()[0] == charEmbeds.size()[0])
    lenTensorSorted, sortIdx = tensorSort(lenTensor)

    assert(len(lenTensorSorted) == charEmbeds.size()[0])

    charEmbedsSorted = Variable(charEmbeds[sortIdx].transpose(0, 1))
    embedsPackedSorted = pack(charEmbedsSorted,
                        tensorToList(lenTensorSorted),
                        batch_first=False)

    output, (ht, ct) = self.charRnn(embedsPackedSorted, None)

    if self.charRnn.bidirectional:
      ht = ht[0, :, :] + ht[1, :, :]
    else:
      ht = ht.squeeze(dim=0)

    ht = tensorUnsort(ht, sortIdx)
    ht = ht.view(sentLen, batchQty, -1)

    return ht

  def forward(self, inp):
    isCuda = inp.data.is_cuda
    sentLen, batchQty, dim = inp.data.size()
    assert(dim == 1)
    spellData = self.word2chars[inp.data.view(-1)]
    spellData = spellData.view(sentLen, batchQty, -1) # sentLen x B x (charLen + 1)
    charTensor = spellData[:, : , : -1]
    lenTensor = spellData[:, :, -1]

    return self.batchRnn(charTensor, lenTensor)







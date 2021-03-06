import torch

from torch.nn import Module, Dropout, LSTM, Embedding, Linear, Conv1d
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

from onmt.modules.Embeddings import Embeddings

# This is modelled after the following code:
# https://github.com/arendu/pytorch-sgns/blob/master/model.py

from spellembed.utils import *

class Char2VecBase(Module):
  def __init__(self,
               vocabSpell,
               wordEmbedSize,
               charEmbedSize=20,
               padIdx=0,
               dropout=0.3):
    super(Char2VecBase, self).__init__()

    self.word2chars, charVocabSize = vocabSpell
    self.charEmbedSize = charEmbedSize
    self.maxWordLen = self.word2chars.size(1) - 1
    self.dropout = Dropout(dropout)
    self.charEmbed = Embedding(charVocabSize, charEmbedSize, padding_idx=padIdx)
    initWeights(self.charEmbed)
    self.embedding_size = wordEmbedSize


  def embedReshapeChars(self, charTensor, lenTensor):
    sentLen, batchQty, maxWordLen = charTensor.size()
    charVar = Variable(charTensor.contiguous().view(-1))
    # After coming through the dropout layer charEmbeds become a variable
    charEmbeds = self.dropout(self.charEmbed(charVar).data) # sentLen x B x wordLen x charEmbedDim
    charEmbeds = charEmbeds.view(sentLen * batchQty, maxWordLen, -1)
    lenTensor = lenTensor.contiguous().view(-1)
    assert(lenTensor.size()[0] == charEmbeds.size()[0])

    return charEmbeds, lenTensor, sentLen, batchQty, maxWordLen

  def batch(charTensor, lenTensor):
    return NotImplementedError

  def forward(self, inp):
    isCuda = inp.data.is_cuda
    sentLen, batchQty, dim = inp.data.size()
    assert(dim == 1)
    spellData = self.word2chars[inp.data.view(-1)]
    spellData = spellData.view(sentLen, batchQty, -1) # sentLen x B x (charLen + 1)
    charTensor = spellData[:, : , : -1]
    lenTensor = spellData[:, :, -1]

    return self.batch(charTensor, lenTensor)

class Char2VecCNN(Char2VecBase):
  def __init__(self,
               vocabSpell,
               wordEmbedSize,
               charEmbedSize=20,
               chanQty = 50,
               padIdx=0,
               dropout=0.3
               ):
    super(Char2VecCNN, self).__init__(vocabSpell,
                                       wordEmbedSize,
                                       charEmbedSize,
                                       padIdx,
                                       dropout)
    print('Maximum word length %d' % self.maxWordLen)
    lIn = self.maxWordLen
    outLens = [Char2VecCNN.compLout(lIn, kernelSize) for kernelSize in [2, 3, 4, 5]]
    #print('Expected conv dims', outLens)
    self.lOutTot = chanQty * sum(outLens)
    self.conv2g = Conv1d(charEmbedSize, chanQty, 2)
    self.conv3g = Conv1d(charEmbedSize, chanQty, 3)
    self.conv4g = Conv1d(charEmbedSize, chanQty, 4)
    self.conv5g = Conv1d(charEmbedSize, chanQty, 5)

    self.convFC = Linear(self.lOutTot, wordEmbedSize)
    initWeights(self.convFC)
    initWeights(self.conv2g)
    initWeights(self.conv3g)
    initWeights(self.conv4g)
    initWeights(self.conv5g)

  def compLout(lIn, kernelSize, padding = 0, stride=1, dilation = 1):
    return (lIn + 2*padding - dilation*(kernelSize-1) - 1)//stride + 1

  def batch(self, charTensor, lenTensor):
    charEmbeds, lenTensor, sentLen, batchQty, maxWordLenBatch = self.embedReshapeChars(charTensor, lenTensor)

    megaBatchQty = sentLen * batchQty

    assert(lenTensor.size(0) == megaBatchQty)
    assert(charEmbeds.size(0) == megaBatchQty)

    mask = ((np.tile(np.arange(maxWordLenBatch), megaBatchQty).reshape(megaBatchQty, -1) <
                    lenTensor.view(-1,1)) * 1).type('torch.FloatTensor')
    isCuda = charTensor.is_cuda
    mask = deviceMap(mask.unsqueeze(dim=2), charTensor.is_cuda)

    assert(mask.size(0) == charEmbeds.size(0))

    charEmbeds = (charEmbeds * Variable(mask)).view(megaBatchQty, maxWordLenBatch, -1).transpose(1, 2) # Embeddings dimension become in-channels

    assert(maxWordLenBatch <= self.maxWordLen)
    if maxWordLenBatch < self.maxWordLen:
      padZeros = deviceMap(torch.zeros(megaBatchQty,
                                       self.charEmbedSize,
                                       self.maxWordLen - maxWordLenBatch), isCuda)
      charEmbeds = torch.cat(charEmbeds, padZeros, dim=2)

    v2 = self.conv2g(charEmbeds)
    v3 = self.conv3g(charEmbeds)
    v4 = self.conv4g(charEmbeds)
    v5 = self.conv5g(charEmbeds)

    #print('conv dims: ', v2.size(), v3.size(), v4.size(), v5.size())

    v = torch.cat([v2, v3, v4, v5], dim=2).view(megaBatchQty, -1)

    assert(v.size(1) == self.lOutTot)

    res = self.convFC(v).view(sentLen, batchQty, -1)
    assert(res.size(2) == self.embedding_size)

    return res

class Char2VecRNN(Char2VecBase):
  def __init__(self,
               vocabSpell,
               wordEmbedSize,
               charEmbedSize=20,
               padIdx=0,
               dropout=0.3,
               numLayers=2,
               isBiDir=True):
    super(Char2VecRNN, self).__init__(vocabSpell,
                                       wordEmbedSize,
                                       charEmbedSize,
                                       padIdx,
                                       dropout)

    self.isBiDir = isBiDir
    self.charRnn = LSTM(charEmbedSize, wordEmbedSize, num_layers=numLayers,
                        dropout=0, bidirectional=isBiDir,
                        batch_first=False)
    for w in self.charRnn.all_weights:
      initWeights(w)

    self.rnnFC = Linear(wordEmbedSize, wordEmbedSize)
    initWeights(self.rnnFC)

  def batch(self, charTensor, lenTensor):
    charEmbeds, lenTensor, sentLen, batchQty, maxWordLenBatch = self.embedReshapeChars(charTensor, lenTensor)

    lenTensorSorted, sortIdx = tensorSort(lenTensor)

    assert(len(lenTensorSorted) == charEmbeds.size()[0])

    charEmbedsSorted = Variable(charEmbeds.data[sortIdx].transpose(0, 1))
    embedsPackedSorted = pack(charEmbedsSorted,
                        tensorToList(lenTensorSorted),
                        batch_first=False)

    outPacked, (ht, ct) = self.charRnn(embedsPackedSorted, None)

    if self.charRnn.bidirectional:
      ht = ht[-1, :, :] + ht[-2, :, :]
    else:
      ht = ht[-1, :, :]

    ht = tensorUnsort(ht, sortIdx)
    ht = self.rnnFC(ht)
    ht = ht.view(sentLen, batchQty, -1)

    return ht

class Char2VecComposite(Module):
  def __init__(self,
               vocabSpell,
               configs,
               charEmbedSize=20,
               padIdx=0,
               dropout=0.3):
    super(Char2VecComposite, self).__init__()

    totWordEmbedSize = 0

    word2chars, _ = vocabSpell

    self.modules = []
    self.wordQty = word2chars.size(0)
    print('Number of target lang. words: %d' % self.wordQty)

    for modName, modConfig in configs.items():

      embedSize = modConfig['embedSize']

      if modName == 'rnn' or modName == 'brnn':
        isBiDir = modName == 'brnn'

        oneMod = Char2VecRNN(vocabSpell,
                             wordEmbedSize=embedSize,
                             charEmbedSize=charEmbedSize,
                             dropout=dropout,
                             isBiDir=isBiDir,
                             numLayers=modConfig['numLayers'])

      elif modName == 'cnn':

        oneMod = Char2VecCNN(vocabSpell,
                             wordEmbedSize=embedSize,
                             charEmbedSize=charEmbedSize,
                             chanQty=modConfig['chanQty'],
                             padIdx=padIdx,
                             dropout=dropout)

      elif modName == 'wembed':

        oneMod = Embeddings(word_vec_size = embedSize,
                            word_vocab_size = self.wordQty,
                            word_padding_idx = padIdx,
                            dropout=dropout)

      self.add_module(modName, oneMod)
      self.modules.append(oneMod)
      totWordEmbedSize += embedSize

    self.embedding_size = totWordEmbedSize
    print('Composite/hybrid embedding size: %d' % self.embedding_size)

  def forward(self, inp):
    res = []
    for oneMod in self.modules:
      res.append(oneMod(inp))

    return torch.cat(res, dim =2)






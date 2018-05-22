from torch.nn import Module, Dropout, LSTM, Embedding

from spellembed.utils import initWeights

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




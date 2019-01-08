import caller
import nltk
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1




def buildVocab(load, annList, threshold):
    counter = Counter()
    i = 0
    for idList in annList:
        label = load.getLabel(idList)
        for ann in label:
            tokens = nltk.tokenize.word_tokenize(ann.lower())
            counter.update(tokens)

        print('[{}/{}] Tokenized captions: '.format(i+1, len(annList)))
        i+=1

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    print('1',vocab)
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    print('1',vocab.__call__('end'))
    for word in words:
        vocab.add_word(word)
    return vocab

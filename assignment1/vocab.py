from __future__ import print_function
from collections import Counter


class Vocab(object):
    def __init__(self, pad=False, unk=False, max_size=None):
        self.word2id = dict()
        self.id2word = dict()
        self.pad_id = self.unk_id = None
        if pad:
            self.add('<pad>')
            self.pad_id = self.word2id['<pad>']
        if unk:
            self.add('<unk>')
            self.unk_id = self.word2id['<unk>']
        self.max_size = max_size

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def build(self, texts):
        """
        Build vocabulary from list of sentences
        Args:
            texts: list(list(str)), list of tokenized sentences
        """
        word_freq = Counter()
        for words in texts:
            for w in words:
                word_freq[w] += 1
        
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)
        for word in top_k_words:
            if self.max_size:
                if len(self.word2id) < self.max_size:
                    self.add(word)
            else:
                self.add(word)

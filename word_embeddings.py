#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import joblib

class GloveVec:
    def __init__(self, vocabulary, embed_dim=200):
        self.vocabulary = vocabulary
        vocab_size = len(self.vocabulary) + 1
        self.embed_mat = np.zeros((vocab_size, embed_dim))
        self.embed_idx = {}

        self._gen_word_idx_dict()
        self._pretrained_vec()
        self._gen_embed_mat()

    def _gen_word_idx_dict(self):
        self.idx2word = {}
        self.word2idx = {}
        idx = 1 # 0 is reserved
        for word in self.vocabulary:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

    def _pretrained_vec(self):
        with open('glove/glove.6B.200d.txt', 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embed_idx[word] = coefs
        assert len(self.embed_idx) == 400000

    def _gen_embed_mat(self):
        for word, idx in self.word2idx.items():
            embed_vec = self.embed_idx.get(word)
            if embed_vec is not None:
                self.embed_mat[idx] = embed_vec
        assert self.embed_mat.shape == (1652, 200)

if __name__ == "__main__":
    # Load vocabulary
    vocab = np.loadtxt('flickr8k/vocabulary.txt', dtype=str)

    glove_vec = GloveVec(vocab)
    print('embedding_matrix:', glove_vec.embed_mat.shape)

    with open('flickr8k/word2idx.pkl', 'wb') as f:
        joblib.dump(glove_vec.word2idx, f, compress=3)

    with open('flickr8k/idx2word.pkl', 'wb') as f:
        joblib.dump(glove_vec.idx2word, f, compress=3)

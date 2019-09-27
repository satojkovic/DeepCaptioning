#!/usr/bin/env python
# -*- coding=utf-8 -*-

from keras import Input
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from gen_train_captions import EOS_TOKEN, SOS_TOKEN
import numpy as np

class ImgCapModel:
    def __init__(self, img_embed_dim=2048, word_embed_dim=200, vocab_size=1652, max_length=34):
        self.vocab_size = vocab_size
        self.max_length = max_length
        img_embed = Input(shape=(img_embed_dim,))
        hidden_img = Dropout(0.5)(img_embed)
        hidden_img = Dense(256, activation='relu')(hidden_img)
        word_seq = Input(shape=(max_length,))
        word_embed = Embedding(self.vocab_size, word_embed_dim, mask_zero=True)(word_seq)
        word_embed = Dropout(0.5)(word_embed)
        hidden_word = LSTM(256)(word_embed)
        decoder = add([hidden_img, hidden_word])
        decoder = Dense(256, activation='relu')(decoder)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)
        self.model = Model(inputs=[img_embed, word_seq], outputs=outputs)

    def summary(self):
        self.model.summary()

    def greedy_search(self, img_feat, word2idx, idx2word):
        in_text = SOS_TOKEN
        for i in range(self.max_length):
            seq = [word2idx[w] for w in in_text.split() if w in word2idx]
            seq = pad_sequences([seq], maxlen=self.max_length)
            yhat = self.model.predict([img_feat, seq], verbose=False)
            yhat = np.argmax(yhat)
            word = idx2word[yhat]
            in_text += ' ' + word
            if word == EOS_TOKEN:
                break
        result_text = in_text.split()
        result_text = result_text[1:-1]
        return ' '.join(result_text)

if __name__ == "__main__":
    img_cap_model = ImgCapModel()
    img_cap_model.summary()
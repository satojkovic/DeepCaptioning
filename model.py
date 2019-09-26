#!/usr/bin/env python
# -*- coding=utf-8 -*-

from keras.models import Model
from keras.layers import Dropout, Dense, Embedding, LSTM
from keras.layers.merge import add
from keras import Input

class ImgCapModel:
    def __init__(self, img_embed_dim=2048, word_embed_dim=200, vocab_size=1652, max_length=34):
        self.vocab_size = vocab_size
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

if __name__ == "__main__":
    img_cap_model = ImgCapModel()
    img_cap_model.summary()
#!/usr/bin/env python
# -*- coding=utf-8 -*-

import joblib
import numpy as np
from keras.engine.training_generator import fit_generator

from data_generator import data_generator
from model import ImgCapModel
from word_embeddings import GloveVec


def max_length(captions):
    lines = []
    for k in captions.keys():
        for cap in captions[k]:
            lines.append(cap)
    return max([len(d.split()) for d in lines])

def get_train_captions(path):
    with open(path, 'rb') as f:
        train_captions = joblib.load(f)
    return train_captions

if __name__ == "__main__":
    # Load data
    vocab = np.loadtxt('flickr8k/vocabulary.txt', dtype=str)

    train_captions = get_train_captions('flickr8k/train_captions.pkl')
    max_length = max_length(train_captions)
    print('max_length:', max_length)

    with open('flickr8k/train_image_feats.pkl', 'rb') as f:
        img_feats = joblib.load(f)

    # Model definition
    img_cap_model = ImgCapModel(vocab_size=len(vocab)+1, max_length=max_length)

    # Use Glove as embedding layer
    glove_vec = GloveVec(vocab)
    img_cap_model.model.layers[2].set_weights([glove_vec.embed_mat])
    img_cap_model.model.layers[2].trainable = False

    # Compile
    img_cap_model.model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Training
    num_epochs = 30
    batch_size = 3
    steps = len(train_captions) // batch_size
    for i in range(num_epochs):
        generator = data_generator(train_captions, img_feats, glove_vec.word2idx, max_length, batch_size, img_cap_model.vocab_size)
        img_cap_model.model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=True)
        img_cap_model.model.save('model_weights/model_' + str(i) + '.h5')
#!/usr/bin/env python
# -*- coding=utf-8 -*-

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

def data_generator(captions, img_feats, word2idx, max_length, batch_size, vocab_size):
    X1, X2, y = [], [], []
    num_pairs = 0
    while True:
        for key, caps in captions.items():
            num_pairs += 1
            img_feat = img_feats[key]
            for cap in caps:
                # Encode words
                seq = [word2idx[w] for w in cap.split(' ') if w in word2idx]
                # Create X, y pairs
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]
                    # Pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # Encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # X, y pair
                    X1.append(img_feat)
                    X2.append(in_seq)
                    y.append(out_seq)
            if num_pairs == batch_size:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = [], [], []
                num_pairs = 0
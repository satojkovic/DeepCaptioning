#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.sequence import pad_sequences

from gen_train_captions import EOS_TOKEN, SOS_TOKEN
from train import get_train_captions, max_length
from model import ImgCapModel

if __name__ == "__main__":
    with open('flickr8k/test_image_feats.pkl', 'rb') as f:
        test_img_feats = joblib.load(f)
    test_imgs = list(test_img_feats.keys())
    target_img = np.random.choice(test_imgs, 1)[0]
    target_img_feat = test_img_feats[target_img].reshape(1, 2048)

    train_captions = get_train_captions('flickr8k/train_captions.pkl')
    max_length = max_length(train_captions)
    with open('flickr8k/word2idx.pkl', 'rb') as f:
        word2idx = joblib.load(f)
    with open('flickr8k/idx2word.pkl', 'rb') as f:
        idx2word = joblib.load(f)

    # Load trained model
    img_cap_model = ImgCapModel()
    img_cap_model.model.load_weights('model_weights/model_9.h5')
    result_text = img_cap_model.greedy_search(target_img_feat, word2idx, idx2word)

    x = plt.imread(os.path.join('flickr8k', 'Flicker8k_Dataset', target_img))
    plt.imshow(x)
    plt.title(result_text)
    plt.show()
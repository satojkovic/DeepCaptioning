#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.sequence import pad_sequences

from gen_train_captions import EOS_TOKEN, SOS_TOKEN
from train import get_train_captions, max_length
from model import ImgCapModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root directory (flickr8k or COCO)')
    parser.add_argument('--img_root', type=str, required=True, help='Path to image root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model.h5')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.dataset_root, 'test_image_feats.pkl'), 'rb') as f:
        test_img_feats = joblib.load(f)
    test_imgs = list(test_img_feats.keys())
    target_img = np.random.choice(test_imgs, 1)[0]
    target_img_feat = test_img_feats[target_img].reshape(1, 2048)

    with open(os.path.join(args.dataset_root, 'word2idx.pkl'), 'rb') as f:
        word2idx = joblib.load(f)
    with open(os.path.join(args.dataset_root, 'idx2word.pkl'), 'rb') as f:
        idx2word = joblib.load(f)

    # Load trained model
    img_cap_model = ImgCapModel(vocab_size=len(word2idx.keys()) + 1)
    img_cap_model.model.load_weights(args.model_path)
    result_text = img_cap_model.greedy_search(target_img_feat, word2idx, idx2word)

    if args.dataset_root == 'COCO':
        target_img = 'COCO_val2014_' + '%012d.jpg' % (target_img)
    x = plt.imread(os.path.join(args.img_root, target_img))
    plt.imshow(x)
    plt.title(result_text)
    plt.show()

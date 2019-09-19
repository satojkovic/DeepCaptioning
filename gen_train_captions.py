#!/usr/bin/env python
# -*- coding=utf-8 -*-

import joblib

SOS_TOKEN = 'zsosz'
EOS_TOKEN = 'zeosz'

def to_vocabulary(captions):
    all_caps = set()
    for k in captions.keys():
        [all_caps.update(c.split()) for c in captions[k]]
    return all_caps

def add_sos_eos(captions):
    for k in captions.keys():
        captions[k] = list(map(lambda x: SOS_TOKEN + ' ' + x + ' ' + EOS_TOKEN, captions[k]))
    return captions

if __name__ == "__main__":
    with open('flickr8k/cleaned_captions.pkl', 'rb') as f:
        cleaned_captions = joblib.load(f)

    # Add start and end token
    cleaned_captions = add_sos_eos(cleaned_captions)

    train_images = []
    with open('flickr8k/Flickr_8k.trainImages.txt', 'r') as f:
        for line in f:
            train_images.append(line.strip())

    train_captions = {}
    for train_image in train_images:
        if train_image in cleaned_captions:
            train_captions[train_image] = cleaned_captions[train_image]

    # Save train captions
    with open('flickr8k/train_captions.pkl', 'wb') as f:
        joblib.dump(train_captions, f, compress=3)

    # Save vocabulary
    vocabulary = to_vocabulary(train_captions)
    with open('flickr8k/vocabulary.txt', 'w') as f:
        for vocab in vocabulary:
            f.writelines(vocab)
            f.writelines('\n')

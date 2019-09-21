#!/usr/bin/env python
# -*- coding=utf-8 -*-

import joblib

SOS_TOKEN = 'zsosz'
EOS_TOKEN = 'zeosz'

TH_WORD_COUNT = 10

def to_vocabulary(words):
    vocab = set()
    [vocab.add(word) for word in words]
    return vocab

def add_sos_eos(captions):
    for k in captions.keys():
        captions[k] = list(map(lambda x: SOS_TOKEN + ' ' + x + ' ' + EOS_TOKEN, captions[k]))
    return captions

def filter_by_count(captions):
    word_counts = {}
    for k in captions.keys():
        for v in captions[k]:
            for w in v.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1
    return [w for w in word_counts if word_counts[w] >= TH_WORD_COUNT]

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
    train_words = filter_by_count(train_captions)
    vocabulary = to_vocabulary(train_words)
    with open('flickr8k/vocabulary.txt', 'w') as f:
        for vocab in vocabulary:
            f.writelines(vocab)
            f.writelines('\n')
    print('Vocabulary size:', len(vocabulary))
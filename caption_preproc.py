#!/usr/bin/env python
# -*- coding=utf-8 -*-

import string
import joblib

SOS_TOKEN = 'zsosz'
EOS_TOKEN = 'zeosz'

def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for _, cap_list in captions.items():
        for i in range(len(cap_list)):
            cap = cap_list[i]
            # tokenize
            cap = cap.split()
            # convert to lower case
            cap = [w.lower() for w in cap]
            # remove punctuation
            cap = [w.translate(table) for w in cap]
            # remove tokens with numbers in them
            cap = [w for w in cap if w.isalpha()]
            # store as string
            cap_list[i] = ' '.join(cap)
    return captions

def add_sos_eos(captions):
    for k in captions.keys():
        captions[k] = list(map(lambda x: SOS_TOKEN + ' ' + x + ' ' + EOS_TOKEN, captions[k]))
    return captions

if __name__ == "__main__":
    captions = {}
    with open('flickr8k/Flickr8k.token.txt', 'r') as f:
        for line in f:
            line = line.strip()
            elems = line.split('\t')
            fn = elems[0][:-2]
            if fn in captions:
                captions[fn].append(elems[1])
            else:
                captions[fn] = [elems[1]]
    print('Num. of data:', len(captions))

    # Clean captions
    cleaned_captions = clean_captions(captions)

    # Add start and end token
    cleaned_captions = add_sos_eos(cleaned_captions)

    # Save captions
    with open('flickr8k/cleaned_captions.pkl', 'wb') as f:
        joblib.dump(cleaned_captions, f, compress=3)
#!/usr/bin/env python
# -*- coding=utf-8 -*-

import string
import joblib

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
            # remove hanging 's' and 'a'
            cap = [w for w in cap if len(w) > 1]
            # store as string
            cap_list[i] = ' '.join(cap)
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

    # Save captions
    with open('flickr8k/cleaned_captions.pkl', 'wb') as f:
        joblib.dump(cleaned_captions, f, compress=3)
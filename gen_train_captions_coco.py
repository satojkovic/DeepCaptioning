#!/usr/bin/env python

import json
import argparse
import os
from sklearn.utils import shuffle
from gen_train_captions import add_sos_eos, to_vocabulary
import joblib
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='COCO', help='Path to COCO dataset directory')
    parser.add_argument('--num_examples', type=int, default=30000, help='The size of the training set')
    parser.add_argument('--top_k', type=int, default=5000, help='Choose the top_k words from the vocabulary')
    return parser.parse_args()

def filter_by_count_coco(captions, top_k):
    all_captions = []
    for k in captions.keys():
        [all_captions.extend([cap for caption in captions[k] for cap in caption.split()])]
    c = Counter(all_captions)
    words, counts = zip(*c.most_common(top_k))
    return words

if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.coco_path, 'cleaned_captions.pkl'), 'rb') as f:
        cleaned_captions = joblib.load(f)
    cleaned_captions = add_sos_eos(cleaned_captions)

    image_ids = [image_id for image_id in cleaned_captions.keys()]
    image_ids = shuffle(image_ids)
    train_image_ids = image_ids[:args.num_examples]

    train_captions = {}
    for train_image_id in train_image_ids:
        train_captions[train_image_id] = cleaned_captions[train_image_id]
    with open(os.path.join(args.coco_path, 'train_captions.pkl'), 'wb') as f:
        joblib.dump(train_captions, f, compress=3)

    train_words = filter_by_count_coco(train_captions, args.top_k)
    vocabulary = to_vocabulary(train_words)
    with open(os.path.join(args.coco_path, 'vocabulary.txt'), 'w') as f:
        for vocab in sorted(vocabulary):
            f.writelines(vocab)
            f.writelines('\n')
    print('Vocabulary size:', len(vocabulary))


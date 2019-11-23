#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import os
import json
from caption_preproc import clean_captions
import joblib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='COCO', help='Path to COCO dataset directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(os.path.join(args.coco_path, 'annotations', 'captions_train2014.json'), 'r') as f:
        data = json.load(f)

    captions = {}
    for annot in data['annotations']:
        if annot['image_id'] in captions:
            captions[annot['image_id']].append(annot['caption'])
        else:
            captions[annot['image_id']] = [annot['caption']]
    print('Num. of images:', len(captions))

    cleaned_captions = clean_captions(captions)

    with open(os.path.join(args.coco_path, 'cleaned_captions.pkl'), 'wb') as f:
        joblib.dump(cleaned_captions, f, compress=3)
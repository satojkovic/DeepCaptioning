#!/usr/bin/env python

import json
import argparse
import os
from sklearn.utils import shuffle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='COCO', help='Path to COCO dataset directory')
    parser.add_argument('--num_examples', type=int, default=30000, help='The size of the training set')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.coco_path, 'annotations/captions_train2014.json'), 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_name_vector = []
    for annot in annotations['annotations']:
        caption = '<start>' + annot['caption'] + '<end>'
        image_id = annot['image_id']
        full_coco_image_path = args.coco_path + '/images/train2014/COCO_train2014_' + '%012d.jpg' % image_id
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector)
    train_captions = train_captions[:args.num_examples]
    img_name_vector = img_name_vector[:args.num_examples]
    print('Num. of examples: train_captions {}, img_name_vector {}'.format(len(train_captions), len(img_name_vector)))
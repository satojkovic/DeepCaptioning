#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import os
import joblib
from gen_image_feats import ImageEncoder, save_feats
from tqdm import tqdm
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='COCO', help='Path to COCO dataset directory')
    return parser.parse_args()

def gen_feats(coco_path, img_ids, img_encoder):
    img_feats = {}
    for img_id in tqdm(img_ids):
        full_coco_img_path = coco_path + '/images/train2014/COCO_train2014_' + '%012d.jpg' % (img_id)
        img_feats[img_id] = img_encoder.encode(full_coco_img_path)
    return img_feats

if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.coco_path, 'train_captions.pkl'), 'rb') as f:
        data = joblib.load(f)
    train_img_ids = [img_id for img_id in data.keys()]

    img_encoder = ImageEncoder()
    train_feats = gen_feats(args.coco_path, train_img_ids, img_encoder)
    save_feats(train_feats, os.path.join(args.coco_path, 'train_image_feats.pkl'))
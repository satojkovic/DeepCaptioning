#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import os
import joblib
from gen_image_feats import ImageEncoder, save_feats
from tqdm import tqdm
from sklearn.utils import shuffle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_val', type=str, required=True, help='train or val')
    parser.add_argument('--coco_path', type=str, default='COCO', help='Path to COCO dataset directory')
    return parser.parse_args()

def gen_feats(coco_path, prefix, img_ids, img_encoder):
    img_feats = {}
    for img_id in tqdm(img_ids):
        full_coco_img_path = coco_path + prefix + '%012d.jpg' % (img_id)
        img_feats[img_id] = img_encoder.encode(full_coco_img_path)
    return img_feats

if __name__ == "__main__":
    args = parse_args()

    img_encoder = ImageEncoder()
    if args.train_or_val.lower() == 'train':
        with open(os.path.join(args.coco_path, 'train_captions.pkl'), 'rb') as f:
            data = joblib.load(f)
        train_img_ids = [img_id for img_id in data.keys()]
        train_feats = gen_feats(args.coco_path, '/images/train2014/COCO_train2014_', train_img_ids, img_encoder)
        save_feats(train_feats, os.path.join(args.coco_path, 'train_image_feats.pkl'))
    else:
        val_img_ids = [int(img_id.split('.')[0].split('_')[-1])
                        for img_id in os.listdir(os.path.join(args.coco_path, 'images', 'val2014'))]
        n_val_img_ids = len(val_img_ids)
        val_img_ids = shuffle(val_img_ids)
        part_val_img_ids = val_img_ids[:int(n_val_img_ids*0.1)]
        val_feats = gen_feats(args.coco_path, '/images/val2014/COCO_val2014_', part_val_img_ids, img_encoder)
        save_feats(val_feats, os.path.join(args.coco_path, 'val_image_feats.pkl'))
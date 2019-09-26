#!/usr/bin/env python
# -*- coding=utf-8 -*-

from tqdm import tqdm
import os
import joblib
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image

class ImageEncoder:
    def __init__(self):
        base_model = InceptionV3(weights='imagenet')
        self.image_model = Model(base_model.input, base_model.layers[-2].output)

    def encode(self, img):
        img = self.preprocess_an_image(img)
        feat = self.image_model.predict(img)
        feat = np.reshape(feat, feat.shape[1]) # (1, 2048) -> (2048,)
        return feat

    def preprocess_an_image(self, img_path):
        # RGB format
        img = image.load_img(img_path, target_size=(299, 299, 3))
        # Convert numpy array to PIL image
        img_array = image.img_to_array(img)
        # Add one more dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the image using inception_v3 module
        img_array = preprocess_input(img_array)
        return img_array

def read_images_file(image_file):
    imgs = []
    with open(image_file, 'r') as f:
        for line in f:
            line = line.strip()
            imgs.append(os.path.join('flickr8k', 'Flicker8k_Dataset', line))
    return imgs

def gen_feats(imgs):
    img_feats = {}
    for img in tqdm(imgs):
        img_feats[os.path.basename(img)] = img_encoder.encode(img)
    return img_feats

def save_feats(feats, save_fname):
    with open(save_fname, 'wb') as f:
        joblib.dump(feats, f, compress=3)

if __name__ == "__main__":
    img_encoder = ImageEncoder()

    train_imgs = read_images_file('flickr8k/Flickr_8k.trainImages.txt')
    train_feats = gen_feats(train_imgs)
    save_feats(train_feats, 'flickr8k/train_image_feats.pkl')

    test_imgs = read_images_file('flickr8k/Flickr_8k.testImages.txt')
    test_feats = gen_feats(test_imgs)
    save_feats(test_feats, 'flickr8k/test_image_feats.pkl')
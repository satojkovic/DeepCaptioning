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

if __name__ == "__main__":
    img_encoder = ImageEncoder()

    train_imgs = []
    with open('flickr8k/Flickr_8k.trainImages.txt', 'r') as f:
        for line in f:
            line = line.strip()
            train_imgs.append(os.path.join('flickr8k', 'Flicker8k_Dataset', line))

    img_feats = {}
    for train_img in tqdm(train_imgs):
        img_feats[os.path.basename(train_img)] = img_encoder.encode(train_img)

    with open('flickr8k/train_image_feats.pkl', 'wb') as f:
        joblib.dump(img_feats, f, compress=3)
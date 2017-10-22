import os
import os.path
import keras.applications
import keras.applications
from keras.preprocessing import image
import numpy as np
import bcolz

def get_files_in_dir(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            yield os.path.join(root, name)

def run_extractor_on_folder(feature_extractor, images_dir, bottleneck_output_dir):
    for file_path in get_files_in_dir(images_dir):
        label = os.path.basename(os.path.dirname(file_path))

        bottleneck_dir = os.path.join(
            bottleneck_output_dir,
            label,
        )
        os.makedirs(bottleneck_dir, exist_ok=True)

        bottleneck_path = os.path.join(
            bottleneck_dir,
            os.path.basename(file_path) + '.txt'
        )

        if os.path.isfile(bottleneck_path):
            print('skipping ', bottleneck_path)
            continue
        
        print('processing ', bottleneck_path)
        bottleneck_values = feature_extractor.get_features(file_path)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)


def save_array(fname, arr): c = bcolz.carray(
    arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

class VGG16LastConvFeatureExtractor:
    def __init__(self, *args):
        self.model = keras.applications.vgg16.VGG16(
            weights='imagenet', include_top=False
        )
    
    def get_features(self, img_path):
        width = 224
        height = 224
        img = image.load_img(img_path, target_size=(width, height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.vgg16.preprocess_input(x)

        features = self.model.predict(x)

        return features.flatten()
        

class InceptionV3FeatureExtractor:
    def __init__(self, *args):
        self.model = keras.applications.inception_v3.InceptionV3(
            weights='imagenet', include_top=True,
        )
    
    def get_features(self, img_path):
        width = 299
        height = 299
        img = image.load_img(img_path, target_size=(width, height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.inception_v3.preprocess_input(x)

        features = self.model.predict(x)

        return features.flatten()
        

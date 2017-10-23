import os
import os.path
import keras.applications
from keras.preprocessing import image
import numpy as np
from keras.layers import Flatten, Input
from keras.models import Model

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

def combine(files, dest_path):
    
    if len(files) <= 4:
        num_tile = 2
    else:
        num_tile = 3
    command = 'montage -quality 100 -tile X{num_tile} -geometry +10+10 {files} miff:- | convert miff:- -resize 1024x1024 {dest_path}'.format(
        files=' '.join(files),
        dest_path=dest_path,
        num_tile=num_tile,
    )
    os.system(command)
    
class VGG16LastConvFeatureExtractor:
    def __init__(self):
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
    width, height = 299, 299
    def __init__(self):
        base_model = keras.applications.inception_v3.InceptionV3(
            weights='imagenet', include_top=True,
        )

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output,
        )
    def get_features(self, img_path):
        img = image.load_img(img_path, target_size=(self.width, self.height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.inception_v3.preprocess_input(x)

        features = self.model.predict(x)

        return features.flatten()

class Resnet50LastConvFeatureExtractor:
    width, height = 224, 224
    def __init__(self):
        base_model = keras.applications.resnet50.ResNet50(
            weights='imagenet', pooling=max, include_top=False
        )
        input = Input(shape=(self.width, self.height, 3), name='image_input')
        x = base_model(input)
        x = Flatten()(x)
        self.model = Model(inputs=input, outputs=x)
        

    def get_features(self, img_path):
        img = image.load_img(img_path, target_size=(self.width, self.height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.inception_v3.preprocess_input(x)

        features = self.model.predict(x)

        return features.flatten()


class Resnet50FeatureExtractor:
    width, height = 224, 224

    def __init__(self):
        base_model = keras.applications.resnet50.ResNet50(
            weights='imagenet', pooling=max, include_top=True,
        )
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output,
        )

    def get_features(self, img_path):
        img = image.load_img(img_path, target_size=(self.width, self.height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.resnet50.preprocess_input(x)

        features = self.model.predict(x)

        return features.flatten()

import keras.applications
from keras.preprocessing import image
import numpy as np
import sys
from utils import *

width = 224
height = 224
def get_features(model, img_path):
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)

    features = model.predict(x)

    return features.flatten()

def main():
    images_dir = sys.argv[1]#'data/per_label_joined'
    bottleneck_output_dir = sys.argv[2] # 'data/vgg_last_conv_bottleneck'

    model = keras.applications.vgg16.VGG16(
        weights='imagenet', include_top=False
    )
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
        bottleneck_values = get_features(model, file_path)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

if __name__ == '__main__':
    main()

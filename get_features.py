import numpy as np
import sys
from utils import *

def main():
    model_name = sys.argv[1]
    images_dir = sys.argv[2]
    bottleneck_output_dir = sys.argv[3]
    
    print(model_name)
    if model_name == 'vgg16_last_conv':
        feature_extractor = VGG16LastConvFeatureExtractor()
    elif model_name == 'inception_v3':
        feature_extractor = InceptionV3FeatureExtractor()
    elif model_name == 'resnet50_last_conv':
        feature_extractor = Resnet50LastConvFeatureExtractor()
    elif model_name == 'resnet50':
        feature_extractor = Resnet50FeatureExtractor()
    else:
        raise Exception('invalid model name')
    run_extractor_on_folder(feature_extractor, images_dir, bottleneck_output_dir)

if __name__ == '__main__':
    main()

import numpy as np
import sys
from utils import *

def main():
    images_dir = sys.argv[1]
    bottleneck_output_dir = sys.argv[2]

    feature_extractor = VGG16LastConvFeatureExtractor()
    run_extractor_on_folder(feature_extractor, images_dir, bottleneck_output_dir)

if __name__ == '__main__':
    main()

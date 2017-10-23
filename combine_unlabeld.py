# Combine multiple images into one.
#
# To install the Pillow module on Mac OS X:
#
# $ xcode-select --install
# $ brew install libtiff libjpeg webp little-cms2
# $ pip install Pillow
#

from __future__ import print_function
import os

from PIL import Image
import collections
from utils import *


test_dataset_path = 'data/test_dataset.txt'
data_dir = 'data/'
out_dir = data_dir + 'unlabled_test_joined/'

lines = open(test_dataset_path).readlines()[1:]
ad_id_to_path_list = collections.defaultdict(list)
for line in lines:
    ad_id, path = line.strip().split()

    ad_id_to_path_list[ad_id].append(
            path,
    )

os.makedirs(out_dir, exist_ok=True)
for ad_id in ad_id_to_path_list:
    files =[]
    for path in ad_id_to_path_list[ad_id]:
        files.append(data_dir + '/' + path)
    
    dest_file = out_dir + '/' + ad_id + '.jpg'

    if os.path.isfile(dest_file):
        continue
    print(files)
    print(dest_file)
    combine(files, dest_file)

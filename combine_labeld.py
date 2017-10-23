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


lines = open('data/train_dataset.txt').readlines()[1:]
data_dir = 'data/'
out_dir = data_dir + 'per_label_joined/'

ad_id_to_path_label_list = collections.defaultdict(list)
for line in lines:
    ad_id, path, label = line.strip().split()

    ad_id_to_path_label_list[ad_id].append(
        (
            path,
            label,
        )
    )

for ad_id in ad_id_to_path_label_list:
    files =[]
    for path, label in ad_id_to_path_label_list[ad_id]:
        files.append(data_dir + '/' + path)
    
    dest_dir = out_dir + label 
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = dest_dir + '/' + ad_id + '.jpg'

    if os.path.isfile(dest_file):
        continue
    print(files)
    print(dest_file)
    combine(files, dest_file)

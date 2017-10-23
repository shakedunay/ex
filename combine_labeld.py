from __future__ import print_function
import os
from utils import *

import collections


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

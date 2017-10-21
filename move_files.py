import shutil
import os
import os

lines = open('data/train_dataset.txt').readlines()[1:]

for line in lines:
    ad_id, path, label = line.strip().split()
    path = 'data/' + path
    dest = 'data/per_label/' + label + '/'
    # print(dest)
    os.makedirs(dest, exist_ok=True)
    command = 'cp {} {}'.format(path, dest)
    # print(command)
    os.system(command)
    # shutil.copy(path, '')

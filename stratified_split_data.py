from utils import *
import shutil
import sys
from sklearn.model_selection import StratifiedShuffleSplit


files = []
labels = []

images_dir = sys.argv[1]

out_dir = images_dir + '_stratified'

for file_path in get_files_in_dir(images_dir):
    label = os.path.basename(os.path.dirname(file_path))
    files.append(file_path)
    labels.append(label)

sss = StratifiedShuffleSplit(
    test_size=0.2,
    random_state=1337,
)

train_index, test_index = next(
    sss.split(np.zeros_like(labels), labels)
)

train_files = [files[i] for i in train_index]
test_files = [files[i] for i in test_index]
train_labels = [labels[i] for i in train_index]
test_labels = [labels[i] for i in test_index]


for file_path, label in zip(train_files,train_labels):
    dest_dir = os.path.join(
        out_dir,
        'train',
        label,
    )

    os.makedirs(dest_dir, exist_ok=True)
    
    shutil.copy(file_path, dest_dir)

for file_path, label in zip(test_files,test_labels):
    dest_dir = os.path.join(
        out_dir,
        'test',
        label,
    )

    os.makedirs(dest_dir, exist_ok=True)
    print(file_path, dest_dir)
    shutil.copy(file_path, dest_dir)

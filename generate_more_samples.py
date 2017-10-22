import os
from utils import *
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random

ad_id_to_num_of_samples = {
    1521: 512,
    1703: 282,
    1707: 255,
    1729: 425,
    1751: 204,
    1755: 1532,
    1837: 54,
    2037: 54,
    2525: 44,
    2526: 17,
}

max_images = 1532
images_base_dir = 'data/per_label_joined/'
dest_base_dir = 'data/per_label_joined_oversampling/'
for ad_id in ad_id_to_num_of_samples:
    # os.system(
    #     'rm -r {dest_base_dir}; cp -R {images_base_dir} {dest_base_dir}'.format(
    #         images_base_dir=images_base_dir,
    #         dest_base_dir=dest_base_dir,
    #     )
    # )

    ad_id_dir = os.path.join(
        images_base_dir,
        str(ad_id),
    )

    dest_ad_id_dir = os.path.join(
        dest_base_dir,
        str(ad_id),
    )

    os.makedirs(dest_ad_id_dir, exist_ok=True)

    file_path_list = list(get_files_in_dir(ad_id_dir))
    
    while ad_id_to_num_of_samples[ad_id] <= max_images:
        file_path = random.choice(file_path_list)
        gen = image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            # width_zoom_range=0.05,
            zoom_range=0.05,
            channel_shift_range=10,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=True,
        )

        # gen = ImageDataGenerator(
        #     rotation_range=40,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.2,
        #     zoom_range=0.2,
        #     horizontal_flip=True,
        #     fill_mode='nearest',
        # )

        img = load_img(file_path)
        
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        batches = gen.flow(
            x,
            batch_size=1,
            save_to_dir=dest_ad_id_dir,
            save_prefix='augmented',
            save_format='jpeg',
        )
        next(batches)
        ad_id_to_num_of_samples[ad_id] += 1
        
    # images_base_dir

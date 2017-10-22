import numpy
numpy.random.seed(1337)

import keras
import argparse
import argparse
import os
import os.path
import json

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3

def fine_tune(
    train_image_dir,
    validation_image_dir,
    batch_size,
    epochs,
    nb_classes,
):
    img_height, img_width, model = get_model(
        nb_classes=nb_classes,
    )
    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
    )
    train_generator = train_data_generator.flow_from_directory(
        train_image_dir,
        target_size=(
            img_height,
            img_width,
        ),
        batch_size=batch_size,
        class_mode='categorical',
    )

    classes = train_generator.class_indices

    results_dir = 'results'
    os.makedirs(
        results_dir,
        exist_ok=True,
    )
    classes_output_file_path = os.path.join(
        results_dir,
        'classes.json',
    )

    with open(classes_output_file_path, 'w') as classes_output_file:
        json.dump(
            obj=classes,
            fp=classes_output_file,
            sort_keys=True,
            indent=4,
            separators=(
                ',',
                ': ',
            )
        )
    validation_data_generator = ImageDataGenerator(
        rescale=1. / 255,
    )
    validation_generator = validation_data_generator.flow_from_directory(
        validation_image_dir,
        target_size=(
            img_height,
            img_width,
        ),
        batch_size=batch_size,
        class_mode='categorical',
    )

    nb_train_samples = train_generator.n
    nb_validation_samples = validation_generator.n

    log_dir = os.path.join(
        results_dir,
        'log',
    )
    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    checkpoints_dir = os.path.join(
        results_dir,
        'checkpoints'
    )
    os.makedirs(
        checkpoints_dir,
        exist_ok=True,
    )
    callbacks = [
        keras.callbacks.CSVLogger(
            os.path.join(
                log_dir,
                'training.log',
            ),
        ),
        # keras.callbacks.ModelCheckpoint(
        #     os.path.join(
        #         checkpoints_dir,
        #         'weights.epoch.{epoch:02d}-val_loss.{val_loss:.2f}.hdf5',
        #     ),
        #     verbose=1,
        # ),
    ] 

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n//validation_generator.batch_size,
        callbacks=callbacks,
    )

    model.save(
        os.path.join(
            checkpoints_dir,
            'weights.hdf5',
        )
    )

def get_model(
    nb_classes,
):
    def setup_to_transfer_learn(base_model):
        num_of_layers = len(base_model.layers)
        for layer in base_model.layers:
            layer.trainable = False

    def add_new_last_layer(base_model, nb_classes):
        FC_SIZE = 1024

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
        predictions = Dense(
            nb_classes,
            activation='softmax',
        )(x)
        model = keras.models.Model(
            input=base_model.input,
            output=predictions,
        )
        return model

    results_dir = 'results'

    checkpoints_dir = os.path.join(
        results_dir,
        'checkpoints'
    )

    weights_path = os.path.join(
        checkpoints_dir,
        'weights.hdf5',
    )
    if os.path.isfile(weights_path):
        print('loading from file...')
        model = keras.models.load_model(weights_path)
    else:
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
        ) #include_top=False excludes final FC layer
        model = add_new_last_layer(
            base_model,
            nb_classes,
        )

        setup_to_transfer_learn(
            base_model,
        )
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    img_height, img_width = 299, 299

    return img_height, img_width, model

def handle_args():
    parser = argparse.ArgumentParser(
        description='fine tune using inception',
    )
    parser.add_argument(
        '--train_image_dir',
        type=str,
        help='path of files to train from',
        required=True,
    )
    parser.add_argument(
        '--validation_image_dir',
        type=str,
        help='path of files to train from',
        required=True,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--nb_classes',
        type=int,
        help='number of claesses in new model',
        required=True,
    )

    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = handle_args()
    fine_tune(
        train_image_dir=args.train_image_dir,
        validation_image_dir=args.validation_image_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        nb_classes=args.nb_classes,
    )

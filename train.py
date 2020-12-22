"""
A keras model to perform multi-task classification. Images have two classes: indoor vs outdoor, and dog vs cat.

This is intended as an example/toy model only, so I didn't bother splitting off a validation set. you can if you want.
"""
from tensorflow.python.keras import backend as K
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)
K.set_session(session)
from keras.models import Input, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.losses import binary_crossentropy
from pickle import load
import numpy as np
import random

def multitask_model(split_fc=False):
    model_input = Input(shape=(224, 224, 3), name='input')

    # throw in some conv->relu->maxpool->dropout layers
    x = Conv2D(64, (7,7), activation='relu', name='conv2d_1', input_shape=(224, 224, 3))(model_input)
    x = MaxPooling2D(pool_size=(3,3), name='maxpool_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)

    x = Conv2D(64, (5,5), activation='relu', name='conv2d_2')(x)
    x = MaxPooling2D(pool_size=(3,3), name='maxpool_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)

    x = Conv2D(64, (3, 3), activation='relu', name='conv2d_3')(x)
    x = MaxPooling2D(pool_size=(3, 3), name='maxpool_3')(x)
    x = Dropout(0.5, name='dropout_3')(x)

    # add some FC layers and then our output predictions. split_fc to split off two branches of fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='dense_4')(x)
    x = Dropout(0.5, name='dropout_4')(x)
    if split_fc:
        x_io = Dense(256, activation='relu', name='dense_5_io')(x)
        x_io = Dropout(0.5, name='dropout_5_io')(x_io)
        x_a = Dense(256, activation='relu', name='dense_5_a')(x)
        x_a = Dropout(0.5, name='dropout_5_a')(x_a)
        output_animal = Dense(1, activation='sigmoid', name='output_animal')(x_a)
        output_indoor_outdoor = Dense(1, activation='sigmoid', name='output_indoor_outdoor')(x_io)
    else:
        x = Dense(256, activation='relu', name='dense_5')(x)
        x = Dropout(0.5, name='dropout_5')(x)
        output_animal = Dense(1, activation='sigmoid', name='output_animal')(x)
        output_indoor_outdoor = Dense(1, activation='sigmoid', name='output_indoor_outdoor')(x)

    model = Model(model_input, [output_indoor_outdoor, output_animal])
    
    return model

def train_gen(training_dict, batch_size=10):
    """
    Our image generator. This should load a batch of images of size batch_size using our training dict, resize them
    all to 224x224, and then stack them together into a (batch_size, 224, 224, 3) tensor, or a stack of (224, 224, 3)
    images

    Target is a stack of [indoor_outdoor_target, animal_target]

    Should return [image batch, target]
    """
    training_list = list(training_dict.items())
    training_len = len(training_list)
    print('training_len = %d' % training_len)
    random.shuffle(training_list)
    list_index = 0
    current_batch_size = 0

    # yield loop
    while 1:
        images = []
        targets_indoor_outdoor = []
        targets_animal = []
        while current_batch_size < batch_size:
            images.append( image.img_to_array( image.load_img( training_list[list_index][0], target_size=(224,224) ) ) / 255.0 )
            targets_indoor_outdoor.append(training_list[list_index][1][0])
            targets_animal.append(training_list[list_index][1][1])
            list_index+=1
            current_batch_size += 1
            #print("list_index = %d, current_batch_size = %d" %(list_index, current_batch_size))
            if list_index >= training_len: list_index = 0
        current_batch_size = 0
        targets = [np.array(targets_indoor_outdoor), np.array(targets_animal)]

        yield [np.stack(images, axis=0)], targets

if __name__ == '__main__':
    batch_size = 10
    epochs = 200

    # load training dict and initialize generator
    training_dict = load(open('training_dict.pkl', 'rb'))
    gen = train_gen(training_dict, batch_size)
    steps = len(training_dict) // batch_size

    # initialize and compile model
    model = multitask_model(True)
    model.compile(loss={'output_indoor_outdoor': 'binary_crossentropy', 'output_animal': 'binary_crossentropy'}, loss_weights={'output_indoor_outdoor': 0.5, 'output_animal': 0.5}, optimizer=SGD(lr=0.001, momentum=0.9), metrics={'output_indoor_outdoor': 'accuracy', 'output_animal': 'accuracy'})

    # fit model
    model.fit_generator(generator=gen, steps_per_epoch=steps, epochs=epochs)

    model.save('trained_model.h5')

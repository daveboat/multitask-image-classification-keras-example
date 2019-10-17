"""
Run prediction using trained_model.h5 on cat.jpeg
"""
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
K.set_session(session)
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# load test image and preprocess
testimg = np.expand_dims(image.img_to_array(image.load_img('dog.jpeg', target_size=(224, 224)))/255, axis=0)

model = load_model('trained_model.h5')

output = model.predict(testimg)
print(output)
# Truth values:
#            Dog     Cat
# Indoor    (0,0)   (0,1)
# Outdoor   (1,0)   (1,1)

if output[1] > 0.5:
    animal = 'cat'
else:
    animal = 'dog'
if output[0] > 0.5:
    location = 'outdoor'
else:
    location = 'indoor'

print('Detected an ' + location + ' image of a ' + animal)

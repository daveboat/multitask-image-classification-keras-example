## Example of multitask image classification in Keras

This is an example/toy model which demonstrates multi-output classification using keras. I use google_images_download (https://github.com/hardikvasa/google-images-download) to scrape images from Google images.

This example performs two classification tasks on the same image: whether the image is of a cat or a dog, and whether the image is indoor or outdoor. Since it's only a toy model, I wasn't concerned with accuracy or even having a validation set. You can improve on it if you would like. I got about 85% training accuracy for indoor/outdoor and about 65% training accuracy for cat/dog after 200 epochs.

The accuracy could be vastly improved by using a pre-trained model (with weights frozen or trainable) as the feature encoder, such as VGG16 or ResNet50, but that wasn't the point of this particular exercise.

Note that when two losses are passed to model.compile, keras adds them together by default, weighted by loss_weights

# Requirements

Only requrires keras and tensorflow (or tensorflow-gpu for CUDA-accelerated training)

google_images_download requires Google Chrome to be installed (https://www.google.com/chrome/) and you need chromedriver (https://chromedriver.chromium.org/downloads)

# Running

1. Run scrape-images.py to create training folders in data/train (Cat Indoors - thumbnail, Dog Indoors - thumbnail, and so forth)
2. Run preprocess-training-set to create training-dict.pkl, a pickled dictionary of training image locations and truth values
3. Run train.py to train and save trained_model.h5
4. Run predict.py to run prediction on the provided image sample

# Other notes

The code block

```
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
K.set_session(session)
```

in train.py and predict.py is needed on my machine due to GPU memory issues with the tensorflow backend. Remove it if you are't running on GPU or if your GPU has more memory than mine.
"""
A quick script to read the scraped images in data/train/ and create a dictionary with filepaths and classifications
"""

import glob
from pickle import dump

# Truth values:
#            Dog     Cat
# Indoor    (0,0)   (0,1)
# Outdoor   (1,0)   (1,1)

dog_indoor_folder = 'data/train/Dog Indoors - thumbnail/'
dog_outdoor_folder = 'data/train/Dog Outdoors - thumbnail/'
cat_indoor_folder = 'data/train/Cat Indoors - thumbnail/'
cat_outdoor_folder = 'data/train/Cat Outdoors - thumbnail/'

training_dict = {}

for file in glob.glob(dog_indoor_folder + '*'):
    training_dict[file] = (0, 0)
for file in glob.glob(dog_outdoor_folder + '*'):
    training_dict[file] = (1, 0)
for file in glob.glob(cat_indoor_folder + '*'):
    training_dict[file] = (0, 1)
for file in glob.glob(cat_outdoor_folder + '*'):
    training_dict[file] = (1, 1)

# save training dict to a file
with open('training_dict.pkl', 'wb') as savefile:
    dump(training_dict, savefile)
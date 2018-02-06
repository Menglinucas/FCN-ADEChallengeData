# convert MITSceneParsingData to trainning_records and validation records
# refer to "https://github.com/shekkizh/FCN.tensorflow/blob/master/read_MITSceneParsingData.py"
import os
import glob
import random
import numpy as np
from six.moves import cPickle as pickle

# create image lists: 
#		{"training":[{"image":..,"annotation":..,"filename":..}],
#			"validation":[{"image":..,"annotation":..,"filename":..}]}
def create_image_lists(image_dir):
    
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        for f in file_list:
            filename = os.path.splitext(f.split("\\")[-1])[0]
            annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
            if os.path.exists(annotation_file):
                record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                image_list[directory].append(record)
            else:
                print("Annotation file not found for %s - Skipping" % filename)
        
        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))
    
    training_records = image_list['training']
    validation_records = image_list['validation']

    return training_records, validation_records
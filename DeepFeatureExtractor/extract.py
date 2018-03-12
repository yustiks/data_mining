# Extract ResNet features using Keras. 
# Images are read directly from SequenceFiles 
# produced by Hadoop (which originally downloaded 
# the images and was later used to resize and crop them)

import sys
import io
import keras.backend as K
from hadoop.io import SequenceFile
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
from keras.preprocessing.image import img_to_array

# Config tensorflow not to take over all resouces
import tensorflow as tf
config = tf.ConfigProto() 
config.gpu_options.allow_growth=True 
sess = tf.Session(config=config)

# laod models
Fmodel = ResNet50(include_top=False, weights='imagenet', pooling='avg')
Pmodel = ResNet50(include_top=True, weights='imagenet')

# format a ndarray as a string
def vector_to_csv(input):
     fmt = ["%.5e", ] * input.shape[1]
     format = ','.join(fmt)
     return format%tuple(input[0])

# output files
featFile = open("ResNet50-features-avg"+sys.argv[1]+".txt","w") 
predFile = open("ResNet50-predictions"+sys.argv[1]+".txt","w") 
clasFile = open("ResNet50-labels"+sys.argv[1]+".txt","w") 

# iterate over all inout SequenceFiles containing images
for i in range(2,len(sys.argv)):
    print ("Processing " + sys.argv[i])
    reader = SequenceFile.Reader(sys.argv[i])

    key_class = reader.getKeyClass()
    value_class = reader.getValueClass()

    key = key_class()
    value = value_class()

    # iterate over all images (key is the flickrid; value is the jpeg bytes)
    while reader.next(key, value):
        skey = key.toString()

        # load the image
        image = Image.open(io.BytesIO(value.toString()))
        image = img_to_array(image, K.image_data_format())
        image = preprocess_input(image)
        image = image.reshape(1,224,224,3)

        # extract from model
        features = Fmodel.predict(image)
        predictions = Pmodel.predict(image)

        # write the values
        featFile.write(skey + "," + vector_to_csv(features) + "\n")
        predFile.write(skey + "," + vector_to_csv(predictions) + "\n")
        clasFile.write(skey + "," + str(decode_predictions(predictions, top=10)[0]) + "\n")

featFile.close()
predFile.close()
clasFile.close()

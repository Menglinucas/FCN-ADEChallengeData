import training_validation_records as getrecords
import BatchDatsetReader as dataset
import numpy as np
import tensorflow as tf
from sys import getsizeof
import scipy.io
import TensorflowUtils as utils
import vgg_inference
import datetime

# 1. load data
# read datasets
image_dir = "E:\pythontest\FCN\ADEChallengeData2016"
image_options = {'resize': True, 'resize_size': 224}
train_records, valid_records = getrecords.create_image_lists(image_dir)

# get batches
train_dataset_reader = dataset.BatchDatset(train_records, image_options)	# class BatchDatset
valid_dataset_reader = dataset.BatchDatset(valid_records, image_options)
#train_images, train_annotations = train_dataset_reader.next_batch(50)	# get ndarray data of batches

## read VGG16 model
#model_data = scipy.io.loadmat("imagenet-vgg-verydeep-19")

# 2. define basical parameters
IMAGE_SIZE = 224
NUM_OF_CLASSES = 151
MAX_ITERATION = 50
learning_rate = 1.0e-4
batch_size = 10

# 3. Construct networks
keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

pred_annotation, logits = vgg_inference.inference(image, keep_probability, NUM_OF_CLASSES)

loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy")))

trainable_var = tf.trainable_variables()

optimizer = tf.train.AdamOptimizer(learning_rate)
grads= optimizer.compute_gradients(loss,trainable_var)
train_op = optimizer.apply_gradients(grads)

# 4. training
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for itr in range(MAX_ITERATION):
		train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
		feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
		
		sess.run(train_op, feed_dict=feed_dict)

		if itr % 1 == 0:
			train_loss = sess.run(loss, feed_dict=feed_dict)
			print("Step: %d, Train_loss:%g" % (itr, train_loss))

		if itr % 500 == 0:
			valid_images, valid_annotations = valid_dataset_reader.next_batch(batch_size)
			valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 1.0})
			print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
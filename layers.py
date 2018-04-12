from keras.layers import Lambda
from util import spatial_transform
import keras.backend as K
import tensorflow as tf


def spatial_transformation(inputs, sign, name):
    def output_shape(input_shape):

        return input_shape[0]

    return Lambda(lambda x: spatial_transform(x[0], sign*x[1]), output_shape=output_shape, name=name)(inputs)


def expand_dims(inputs, dimension, name):
    def output_shape(input_shape):
        shape = list(input_shape)

        shape[3] = 1

        return tuple(shape)

    return Lambda(lambda x: K.expand_dims(inputs[:, :, :, dimension], 3), output_shape=output_shape, name=name)(inputs)

def depth_to_disparity(inputs, baseline, focal_length, width, name):
    def output_shape(input_shape):
        return input_shape
    _num_batch    = inputs.shape[0]
    _height       = inputs.shape[1]
    _width        = inputs.shape[2]
    f = tf.tile(tf.reshape(focal_length,(_num_batch,1,1,1)), tf.stack([1,_height,_width,1]))
    b = tf.tile(tf.reshape(baseline,(_num_batch,1,1,1)), tf.stack([1,_height,_width,1]))
    return Lambda(lambda x: x[2] * x[1] /(x[0] * width) , output_shape=output_shape, name=name)([inputs, f, b])

#def depth_to_disparity(inputs, baseline, focal_length, width, name):
#    def output_shape(input_shape):
#        return input_shape

#    return Lambda(lambda x: baseline * focal_length / x, output_shape=output_shape, name=name)(inputs)

def disparity_to_depth(inputs, baseline, focal_length, width, name):
    def output_shape(input_shape):
        return input_shape
    _num_batch    = inputs.shape[0]
    _height       = inputs.shape[1]
    _width        = inputs.shape[2]
    f = tf.tile(tf.reshape(focal_length,(_num_batch,1,1,1)), tf.stack([1,_height,_width,1]))
    b = tf.tile(tf.reshape(baseline,(_num_batch,1,1,1)), tf.stack([1,_height,_width,1]))
    return Lambda(lambda x: x[2] * x[1] /(x[0] * width) , output_shape=output_shape, name=name)([inputs, f, b])

#def disparity_to_depth(inputs, baseline, focal_length, width, name):
#    def output_shape(input_shape):
#        return input_shape

#    return Lambda(lambda x: baseline * focal_length /(x * width) , output_shape=output_shape, name=name)(inputs)

def disparity_difference(disparities, name):
    def output_shape(input_shape):
        return input_shape

    return Lambda(lambda x: x[0] - x[1], output_shape=output_shape, name=name)(disparities)


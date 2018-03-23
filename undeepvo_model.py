
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Cropping2D, Dense, Flatten
from layers import depth_to_disparity, disparity_difference, expand_dims, spatial_transformation

from bilinear_sampler import *

undeepvo_parameters = namedtuple('parameters',
                        'height, width, '
                        'resize_ratio, '
                        'baseline, focal_length, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')

class UndeepvoModel(object):
    """undeepvo model"""

    def __init__(self, params, mode, left, right, left_next, right_next, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.left_next = left_next
        self.right_next = right_next
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()     


    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 1, 3, 1, activation='sigmoid')
        return disp

    @staticmethod
    def conv(input, channels, kernel_size, strides, activation='elu'):

        return Conv2D(channels, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(input)

    @staticmethod
    def deconv(input, channels, kernel_size, scale):

        output =  Conv2DTranspose(channels, kernel_size=kernel_size, strides=scale, padding='same')(input)
        output_shape = output._keras_shape
        output.set_shape(output_shape)
        return output
    @staticmethod
    def maxpool(input, kernel_size):
        
        return MaxPooling2D(pool_size=kernel_size, strides=2, padding='same', data_format=None)(input)

    def conv_block(self, input, channels, kernel_size):
        conv1 = self.conv(input, channels, kernel_size, 1)

        conv2 = self.conv(conv1, channels, kernel_size, 2)

        return conv2

    def deconv_block(self, input, channels, kernel_size, skip):

        deconv1 = self.deconv(input, channels, kernel_size, 2)
        if skip is not None:
            s = skip.shape
            if  s[1] % 2 != 0:
                deconv1 = deconv1[:,:-1,:,:]
            if  s[2] % 2 != 0:
                deconv1 = deconv1[:,:,:-1,:]
            concat1 = concatenate([deconv1, skip], 3)
        else:
            concat1 = deconv1

        iconv1 = self.conv(concat1, channels, kernel_size, 1)

        return iconv1

    def build_pose_architecture(self,in_image,in_image_next):
#        in_image_resized  = tf.image.resize_images(in_image,  [384, 1280], tf.image.ResizeMethod.AREA)
#        in_image_next_resized  = tf.image.resize_images(in_image_next,  [384, 1280], tf.image.ResizeMethod.AREA)
#        input = concatenate([in_image_resized, in_image_next_resized], axis=3)
        input = concatenate([in_image, in_image_next], axis=3)

        conv1 = self.conv(input, 16, 7, 2, activation='relu')

        conv2 = self.conv(conv1, 32, 5, 2, activation='relu')

        conv3 = self.conv(conv2, 64, 3, 2, activation='relu')

        conv4 = self.conv(conv3, 128, 3, 2, activation='relu')

        conv5 = self.conv(conv4, 256, 3, 2, activation='relu')

        conv6 = self.conv(conv5, 256, 3, 2, activation='relu')

        conv7 = self.conv(conv6, 512, 3, 2, activation='relu')

#        flat1 = Flatten()(conv7)
        dim = np.prod(conv7.shape[1:])
        flat1 = tf.reshape(conv7, [-1, dim])

        # translation

        fc1_tran = Dense(512, input_shape=(dim,))(flat1)

        fc2_tran = Dense(512, input_shape=(512,))(fc1_tran)

        fc3_tran = Dense(3, input_shape=(512,))(fc2_tran)

        # rotation

        fc1_rot = Dense(512, input_shape=(dim,))(flat1)

        fc2_rot = Dense(512, input_shape=(512,))(fc1_rot)

        fc3_rot = Dense(3, input_shape=(512,))(fc2_rot)


        return fc3_tran, fc3_rot

    def build_depth_architecture(self,in_image):
        # encoder
        conv1 = self.conv_block(in_image, 32, 7)

        conv2 = self.conv_block(conv1, 64, 5)

        conv3 = self.conv_block(conv2, 128, 3)

        conv4 = self.conv_block(conv3, 256, 3)

        conv5 = self.conv_block(conv4, 512, 3)

        conv6 = self.conv_block(conv5, 512, 3)

        conv7 = self.conv_block(conv6, 512, 3)

        # skips
        skip1 = conv1

        skip2 = conv2

        skip3 = conv3

        skip4 = conv4

        skip5 = conv5

        skip6 = conv6

        deconv7 = self.deconv_block(conv7, 512, 3, skip6)

        deconv6 = self.deconv_block(deconv7, 512, 3, skip5)

        deconv5 = self.deconv_block(deconv6, 256, 3, skip4)

        deconv4 = self.deconv_block(deconv5, 128, 3, skip3)

        deconv3 = self.deconv_block(deconv4, 64, 3, skip2)

        deconv2 = self.deconv_block(deconv3, 32, 3, skip1)

        deconv1 = self.deconv_block(deconv2, 1, 3, None)

        s = in_image.shape
        if  s[1] % 2 != 0:
            deconv1 = deconv1[:,:-1,:,:]
        if  s[2] % 2 != 0:
            deconv1 = deconv1[:,:,:-1,:]
#        return deconv1
        disp = self.get_disp(deconv1)
        return disp

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.depthmap_left = self.build_depth_architecture(self.left)
                self.depthmap_right = self.build_depth_architecture(self.right)
                self.translation_left, self.rotation_left = self.build_pose_architecture(self.left,self.left_next)
                self.translation_right, self.rotation_right = self.build_pose_architecture(self.right,self.right_next)

    def build_outputs(self):
        if self.mode == 'test':
            return

        # generate disparities
        self.disparity_left = depth_to_disparity(self.depthmap_left, self.params.baseline, self.params.focal_length*self.params.resize_ratio, 1, 'disparity_left')
        self.disparity_right = depth_to_disparity(self.depthmap_right, self.params.baseline, self.params.focal_length*self.params.resize_ratio, 1, 'disparity_right')

        # generate estimates of left and right images
        self.left_est  = self.generate_image_left(self.right, self.disparity_left)
        self.right_est = self.generate_image_right(self.left, self.disparity_right)
        
        # generate left - right consistency
        self.right_to_left_disparity = self.generate_image_left(self.disparity_right, self.disparity_left)
        self.left_to_right_disparity = self.generate_image_right(self.disparity_left, self.disparity_right)


    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = tf.reduce_mean(tf.abs( self.left_est - self.left))
            self.l1_right = tf.reduce_mean(tf.abs(self.right_est - self.right))
            # SSIM
            self.ssim_left = tf.reduce_mean(self.SSIM( self.left_est,  self.left))
            self.ssim_right = tf.reduce_mean(self.SSIM(self.right_est, self.right))

            # PHOTOMETRIC CONSISTENCY
            self.image_loss_left  = self.params.alpha_image_loss * self.ssim_left  + (1 - self.params.alpha_image_loss) * self.l1_left
            self.image_loss_right = self.params.alpha_image_loss * self.ssim_right + (1 - self.params.alpha_image_loss) * self.l1_right        
            self.image_loss = self.image_loss_left + self.image_loss_right

            # DISPARITY
            self.l1_disp_left = tf.reduce_mean(tf.abs( self.disparity_left - self.right_to_left_disparity))
            self.l1_disp_right = tf.reduce_mean(tf.abs( self.disparity_right - self.left_to_right_disparity))

            # DISPARITY CONSISTENCY
            self.disp_loss = self.l1_disp_left + self.l1_disp_right

            # POSE CONSISTENCY
            self.l1_translation = tf.reduce_mean(tf.abs( self.translation_left - self.translation_right))
            self.l1_rotation = tf.reduce_mean(tf.abs( self.rotation_left - self.rotation_right))
            self.pose_loss = self.l1_translation + self.l1_rotation

            # PHOTOMETRIC REGISTRATION

            # GEOMETRIC REGISTRATION

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.disp_loss+ self.pose_loss
#            self.total_loss = self.image_loss + self.disp_loss + self.pose_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
#            tf.summary.scalar('image_loss', self.image_loss, collections=self.model_collection)
#            tf.summary.scalar('disp_loss', self.disp_loss, collections=self.model_collection)
#            tf.summary.scalar('pose_loss', self.pose_loss, collections=self.model_collection)
            

            if self.params.full_summary:
                tf.summary.image('left_est', self.left_est, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right_est', self.right_est, max_outputs=4, collections=self.model_collection)
                tf.summary.image('ssim_left', self.ssim_left,  max_outputs=4, collections=self.model_collection)
                tf.summary.image('ssim_right', self.ssim_right, max_outputs=4, collections=self.model_collection)
                tf.summary.image('l1_left', self.l1_left,  max_outputs=4, collections=self.model_collection)
                tf.summary.image('l1_right', self.l1_right, max_outputs=4, collections=self.model_collection)
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)


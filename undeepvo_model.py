
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Cropping2D, Dense, Flatten

from bilinear_sampler import *

undeepvo_parameters = namedtuple('parameters',
                        'height, width, '
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

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

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

    @staticmethod
    def conv(input, channels, kernel_size, strides, activation='elu'):

        return Conv2D(channels, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(input)

    @staticmethod
    def deconv(input, channels, kernel_size, scale):

        return Conv2DTranspose(channels, kernel_size=kernel_size, strides=scale, padding='same')(input)
    
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
            concat1 = concatenate([deconv1, skip], 3)
        else:
            concat1 = deconv1

        iconv1 = self.conv(concat1, channels, kernel_size, 1)

        return iconv1

    def build_pose_architecture(self,in_image,in_image_next):
        in_image_resized  = tf.image.resize_images(in_image,  [384, 1280], tf.image.ResizeMethod.AREA)
        in_image_next_resized  = tf.image.resize_images(in_image_next,  [384, 1280], tf.image.ResizeMethod.AREA)
        input = concatenate([in_image_resized, in_image_next_resized], axis=3)

        conv1 = self.conv(input, 16, 7, 2, activation='relu')

        conv2 = self.conv(conv1, 32, 5, 2, activation='relu')

        conv3 = self.conv(conv2, 64, 3, 2, activation='relu')

        conv4 = self.conv(conv3, 128, 3, 2, activation='relu')

        conv5 = self.conv(conv4, 256, 3, 2, activation='relu')

        conv6 = self.conv(conv5, 256, 3, 2, activation='relu')

        conv7 = self.conv(conv6, 512, 3, 2, activation='relu')

        flat1 = Flatten()(conv6)

        # translation

        fc1_tran = Dense(512, input_shape=(15360,))(flat1)

        fc2_tran = Dense(512, input_shape=(512,))(fc1_tran)

        fc3_tran = Dense(3, input_shape=(512,))(fc2_tran)

#        self.translation = fc3_tran

        # rotation

        fc1_rot = Dense(512, input_shape=(15360,))(flat1)

        fc2_rot = Dense(512, input_shape=(512,))(fc1_rot)

        fc3_rot = Dense(3, input_shape=(512,))(fc2_rot)

#        self.rotation = fc3_rot

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

        deconv1 = self.deconv_block(deconv2, 16, 3, None)

#        self.depthmap = self.get_depth(deconv1)
        return deconv1

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
        self.disparity_left = depth_to_disparity(self.depthmap_left, self.baseline, self.focal_length, 1,
                                                 'disparity_left')
        self.disparity_right = depth_to_disparity(self.depthmap_right, self.baseline, self.focal_length, 1,
                                                  'disparity_right')

        # generate estimates of left and right images
        self.left_est = spatial_transformation([self.right, self.disparity_right], -1, 'left_est')
        self.right_est = spatial_transformation([self.left, self.disparity_left], 1, 'right_est')

        # generate left - right consistency

        self.right_to_left_disparity = spatial_transformation([self.disparity_right, self.disparity_right], -1,
                                                              'r2l_disparity')
        self.left_to_right_disparity = spatial_transformation([self.disparity_left, self.disparity_left], 1,
                                                              'l2r_disparity')
        self.disparity_diff_left = disparity_difference([self.disparity_left, self.right_to_left_disparity],
                                                        'disp_diff_left')
        self.disparity_diff_right = disparity_difference([self.disparity_right, self.left_to_right_disparity],
                                                         'disp_diff_right')

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = tf.abs( self.left_est - self.left)
            self.l1_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = tf.abs(self.right_est - self.right)
            self.l1_loss_right  = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = self.SSIM( self.left_est,  self.left)
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left] 
            self.ssim_right = self.SSIM(self.right_est, self.right)
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # PHOTOMETRIC CONSISTENCY 
            self.image_loss_left  = self.params.alpha_image_loss * self.ssim_loss_left  + (1 - self.params.alpha_image_loss) * self.l1_loss_left
            self.image_loss_right = self.params.alpha_image_loss * self.ssim_loss_right + (1 - self.params.alpha_image_loss) * self.l1_loss_right         
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY CONSISTENCY
            self.disp_loss = tf.add_n(self.disparity_diff_left + self.disparity_diff_right)

            # POSE CONSISTENCY
            self.l1_translation_loss = tf.abs( self.translation_left, self.translation_right)
            self.l1_rotation_loss = tf.abs( self.rotation_left, self.rotation_right)
            self.pose_loss = tf.add_n(self.l1_translation_loss+self.l1_rotation_loss)

            # PHOTOMETRIC REGISTRATION

            # GEOMETRIC REGISTRATION

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.disp_loss + self.pose_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('image_loss', self.image_loss, collections=self.model_collection)
            tf.summary.scalar('disp_loss', self.disp_loss, collections=self.model_collection)
            tf.summary.scalar('disp_loss', self.pose_loss, collections=self.model_collection)
            

            if self.params.full_summary:
                tf.summary.image('left_est', self.left_est, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right_est', self.right_est, max_outputs=4, collections=self.model_collection)
                tf.summary.image('ssim_left', self.ssim_left,  max_outputs=4, collections=self.model_collection)
                tf.summary.image('ssim_right', self.ssim_right, max_outputs=4, collections=self.model_collection)
                tf.summary.image('l1_left', self.l1_left,  max_outputs=4, collections=self.model_collection)
                tf.summary.image('l1_right', self.l1_right, max_outputs=4, collections=self.model_collection)
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)


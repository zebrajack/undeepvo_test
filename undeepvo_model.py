
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Cropping2D, Dense, Flatten, Input, Reshape
from keras.models import Model
from layers import depth_to_disparity, disparity_difference, expand_dims, spatial_transformation, disparity_to_depth

from bilinear_sampler import *
from projective_transformer import *
from projective_transformer_inv import *

undeepvo_parameters = namedtuple('parameters',
                        'height, width, '
                        'baseline, ' 
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'alpha_image_loss, '
                        'image_loss_weight, disp_loss_weight, pose_loss_weight, gradient_loss_weight, temporal_loss_weight, '
                        'full_summary')

class UndeepvoModel(object):
    """undeepvo model"""

    def __init__(self, params, mode, left, right, left_next, right_next, cam_params, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.left_next = left_next
        self.right_next = right_next
        self.model_collection = ['model_' + str(model_index)]

        self.focal_length1_left = cam_params[:,0]*cam_params[:,8]
        self.focal_length2_left = cam_params[:,0]*cam_params[:,7]
        self.c0_left = cam_params[:,1]*cam_params[:,8]
        self.c1_left = cam_params[:,2]*cam_params[:,7]
        self.focal_length1_right = cam_params[:,3]*cam_params[:,8]
        self.focal_length2_right = cam_params[:,3]*cam_params[:,7]
        self.c0_right = cam_params[:,4]*cam_params[:,8]
        self.c1_right = cam_params[:,5]*cam_params[:,7]
        self.baseline = cam_params[:,6]

        self.reuse_variables = reuse_variables

        
        self.build_depth_architecture()
        self.build_pose_architecture()

        self.depthmap_left, self.depthmap_right, self.depthmap_left_next, self.depthmap_right_next, self.translation_left, self.rotation_left, self.translation_right, self.rotation_right = self.build_model(self.left, self.right, self.left_next, self.right_next)
#        self.disparity_left, self.disparity_right, self.disparity_left_next, self.disparity_right_next, self.translation_left, self.rotation_left, self.translation_right, self.rotation_right = self.build_model(self.left, self.right, self.left_next, self.right_next)
        

        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        gx = tf.pad(gx, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        gy = tf.pad(gy, [[0, 0], [0, 1], [0, 0], [0, 0]], "CONSTANT")
        return gy

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def get_disparity_smoothness_single(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True)) 
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return smoothness_x + smoothness_y

#    def get_disparity_smoothness(self, disp, pyramid):
#        disp_gradients_x = [self.gradient_x(d) for d in disp]
#        disp_gradients_y = [self.gradient_y(d) for d in disp]

#        image_gradients_x = [self.gradient_x(img) for img in pyramid]
#        image_gradients_y = [self.gradient_y(img) for img in pyramid]

#        weights_x = [tf.exp(-1.0/(tf.reduce_mean(tf.abs(g), 3, keepdims=True)+0.01)) for g in image_gradients_x]
#        weights_y = [tf.exp(-1.0/(tf.reduce_mean(tf.abs(g), 3, keepdims=True)+0.01)) for g in image_gradients_y]

#        smoothness_x = [1.0/(disp_gradients_x[i]+1.0) * weights_x[i] for i in range(4)]
#        smoothness_y = [1.0/(disp_gradients_y[i]+1.0) * weights_y[i] for i in range(4)]
#        return smoothness_x + smoothness_y

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keepdims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keepdims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

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

    def upsample_nn(self, input, ratio):
        s = input.get_shape().as_list()
        h = s[1]
        w = s[2]
        nh = h * ratio
        nw = w * ratio
        output = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, [nh, nw]))(input)
        output.set_shape(output_shape)
        return output

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]    
        s = img.get_shape().as_list()
        h = s[1]
        w = s[2]
        tmp_h = h
        tmp_w = w
        nh = h
        nw = w
        for i in range(num_scales - 1):
            nh = nh // 2
            nw = nw // 2
            if  tmp_h % 2 != 0:
                nh = nh +1
            if  tmp_w % 2 != 0:
                nw = nw +1
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            scaled_imgs[i+1].set_shape([s[0],nh,nw,s[3]])
            tmp_h = nh
            tmp_w = nw
        return scaled_imgs

    def get_disp(self, x):
        disp = self.conv(x, 1, 3, 1, activation='sigmoid')
        disp = Lambda(lambda x: 0.3 * x)(disp)
        return disp

    def get_depth(self, x):
        depth = self.conv(x, 1, 3, 1, activation='sigmoid')
        depth = Lambda(lambda x: 100.0 * x)(depth)
        return depth

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
#            print(s)
#            print(deconv1.shape)
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            concat1 = concatenate([deconv1, skip], 3)
        else:
            concat1 = deconv1

        iconv1 = self.conv(concat1, channels, kernel_size, 1)
#        print(iconv1.shape)
        return iconv1


    def build_pose_architecture(self):

        input_shape = concatenate([self.left,self.left_next], axis=3)
        input = Input(batch_shape=input_shape.get_shape().as_list())

        conv1 = self.conv(input, 16, 7, 2, activation='relu')

        conv2 = self.conv(conv1, 32, 5, 2, activation='relu')

        conv3 = self.conv(conv2, 64, 3, 2, activation='relu')

        conv4 = self.conv(conv3, 128, 3, 2, activation='relu')

        conv5 = self.conv(conv4, 256, 3, 2, activation='relu')

        conv6 = self.conv(conv5, 256, 3, 2, activation='relu')

        conv7 = self.conv(conv6, 512, 3, 2, activation='relu')

        dim = np.prod(conv7.shape[1:])

        flat1 = Lambda(lambda x: tf.reshape(x, [-1, dim]))(conv7)

        # translation

        fc1_tran = Dense(512, input_shape=(dim,))(flat1)

        fc2_tran = Dense(512, input_shape=(512,))(fc1_tran)

        fc3_tran = Dense(3, input_shape=(512,))(fc2_tran)

        # rotation

        fc1_rot = Dense(512, input_shape=(dim,))(flat1)

        fc2_rot = Dense(512, input_shape=(512,))(fc1_rot)

        fc3_rot = Dense(3, input_shape=(512,))(fc2_rot)

#        # uncertainty

#        fc1_uncertainty = Dense(512, input_shape=(dim,))(flat1)

#        fc2_uncertainty = Dense(512, input_shape=(512,))(fc1_uncertainty)

#        fc3_uncertainty = Dense(21, input_shape=(512,))(fc2_uncertainty)

        self.pose_model = Model(input, [fc3_tran, fc3_rot])     
#        self.pose_model = Model(input, [fc3_tran, fc3_rot, fc3_uncertainty])

    def build_depth_architecture(self):

        input = Input(batch_shape=self.left.get_shape().as_list())

        # encoder
        conv1 = self.conv_block(input, 32, 7)

        conv2 = self.conv_block(conv1, 64, 5)

        conv3 = self.conv_block(conv2, 128, 3)


        conv4 = self.conv_block(conv3, 256, 3)

        conv5 = self.conv_block(conv4, 512, 3)

        conv6 = self.conv_block(conv5, 512, 3)

        conv7 = self.conv_block(conv6, 512, 3)

#        conv8 = self.conv_block(conv7, 512, 3)

        # skips
        skip1 = conv1

        skip2 = conv2

        skip3 = conv3

        skip4 = conv4

        skip5 = conv5

        skip6 = conv6

#        skip7 = conv7

        # decoder1
#        deconv8 = self.deconv_block(conv8,1024,3,skip7)

        deconv7 = self.deconv_block(conv7, 512, 3, skip6)

        deconv6 = self.deconv_block(deconv7, 512, 3, skip5)

        deconv5 = self.deconv_block(deconv6, 256, 3, skip4)
        
        deconv4 = self.deconv_block(deconv5, 128, 3, skip3)
        disp4 = self.get_depth(deconv4)

        deconv3 = self.deconv_block(deconv4, 64, 3, skip2)
        disp3 = self.get_depth(deconv3)

        deconv2 = self.deconv_block(deconv3, 32, 3, skip1)
        disp2 = self.get_depth(deconv2)

        deconv1 = self.deconv_block(deconv2, 16, 3, None)

        s = self.left.shape
        if  s[1] % 2 != 0:
            deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
        if  s[2] % 2 != 0:
            deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

        disp1 = self.get_depth(deconv1)

        disp_est  = [disp1, disp2, disp3, disp4]

#        # decoder2
#        deconv8_2 = self.deconv_block(conv8,1024,3,skip7)

#        deconv7_2 = self.deconv_block(deconv8_2, 1024, 3, skip6)

#        deconv6_2 = self.deconv_block(deconv7_2, 512, 3, skip5)

#        deconv5_2 = self.deconv_block(deconv6_2, 256, 3, skip4)
#        
#        deconv4_2 = self.deconv_block(deconv5_2, 128, 3, skip3)
#        disp4_2 = self.get_depth(deconv4_2)

#        deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)
#        disp3_2 = self.get_depth(deconv3_2)

#        deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)
#        disp2_2 = self.get_depth(deconv2_2)

#        deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

#        s = self.left.shape
#        if  s[1] % 2 != 0:
#            deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
#        if  s[2] % 2 != 0:
#            deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

#        disp1_2 = self.get_depth(deconv1_2)

#        disp_est  = [disp1, disp2, disp3, disp4, disp1_2, disp2_2, disp3_2, disp4_2]

        self.depth_model = Model(input, disp_est)

#        return depth

    def build_model(self,img_left,img_right,img_left_next,img_right_next):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model',reuse=self.reuse_variables):

                self.left_pyramid  = self.scale_pyramid(img_left,  4)
                self.right_pyramid  = self.scale_pyramid(img_right,  4)

                disp_left = self.depth_model(img_left) 
                [disp_left[i].set_shape(disp_left[i]._keras_shape) for i in range(4)]
#                [uncertainty_left[i].set_shape(uncertainty_left[i]._keras_shape) for i in range(4)]
                disp_right = self.depth_model(img_right)
                [disp_right[i].set_shape(disp_right[i]._keras_shape) for i in range(4)]
#                [uncertainty_right[i].set_shape(uncertainty_right[i]._keras_shape) for i in range(4)]

                disp_left_next = self.depth_model(img_left_next) 
                [disp_left_next[i].set_shape(disp_left_next[i]._keras_shape) for i in range(4)] 
#                [uncertainty_left_next[i].set_shape(uncertainty_left_next[i]._keras_shape) for i in range(4)]
                disp_right_next = self.depth_model(img_right_next)
                [disp_right_next[i].set_shape(disp_right_next[i]._keras_shape) for i in range(4)]
#                [uncertainty_right_next[i].set_shape(uncertainty_right_next[i]._keras_shape) for i in range(4)]

                [left_trans, left_rot] = self.pose_model(concatenate([img_left,img_left_next], axis=3))                
                [right_trans, right_rot] = self.pose_model(concatenate([img_right,img_right_next], axis=3))

        return disp_left,disp_right, disp_left_next, disp_right_next, left_trans, left_rot, right_trans, right_rot
#        return disp_left,disp_right, left_trans, left_rot, right_trans, right_rot


    def build_outputs(self):
        if self.mode == 'test':
            return
    
        # generate depth maps
        self.disparity_left = [disparity_to_depth(self.depthmap_left[i], self.baseline, self.focal_length1_left, self.params.width, 'depthmap_left') for i in range(4)]
        self.disparity_right = [disparity_to_depth(self.depthmap_right[i], self.baseline, self.focal_length1_right, self.params.width, 'depthmap_right') for i in range(4)]
        self.disparity_left_next = [disparity_to_depth(self.depthmap_left_next[i], self.baseline, self.focal_length1_left, self.params.width, 'depthmap_left_next') for i in range(4)]
        self.disparity_right_next = [disparity_to_depth(self.depthmap_right_next[i], self.baseline, self.focal_length1_right, self.params.width, 'depthmap_right_next') for i in range(4)]
#        self.depthmap_left = disparity_to_depth(self.disparity_left[0], self.baseline, self.focal_length1_left, self.params.width, 'depthmap_left')
#        self.depthmap_right = disparity_to_depth(self.disparity_right[0], self.baseline, self.focal_length1_right, self.params.width, 'depthmap_right')
#        self.depthmap_left_next =disparity_to_depth(self.disparity_left_next[0], self.baseline, self.focal_length1_left, self.params.width, 'depthmap_left_next')
#        self.depthmap_right_next = disparity_to_depth(self.disparity_right_next[0], self.baseline, self.focal_length1_right, self.params.width, 'depthmap_right_next')


        # generate estimates of left and right images
        self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disparity_right[i]) for i in range(4)]
        self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disparity_left[i]) for i in range(4)]

#        # generate left - right consistency
#        self.right_to_left_disparity = [self.generate_image_left(self.disparity_right[i], self.disparity_right[i]) for i in range(4)]
#        self.left_to_right_disparity = [self.generate_image_right(self.disparity_left[i], self.disparity_left[i]) for i in range(4)]

        # generate k th image
        self.left_k = projective_transformer_inv(self.left_next, self.focal_length1_left, self.focal_length2_left , self.c0_left, self.c1_left, self.depthmap_left_next[0], self.rotation_left, self.translation_left)
        self.right_k = projective_transformer_inv(self.right_next, self.focal_length1_right, self.focal_length2_right , self.c0_right, self.c1_right, self.depthmap_right_next[0], self.rotation_right, self.translation_right)

        # generate k+1 th image
        self.left_k_plus_one = projective_transformer(self.left, self.focal_length1_left, self.focal_length2_left , self.c0_left, self.c1_left, self.depthmap_left[0], self.rotation_left, self.translation_left)
        self.right_k_plus_one = projective_transformer(self.right, self.focal_length1_right, self.focal_length2_right , self.c0_right, self.c1_right, self.depthmap_right[0], self.rotation_right, self.translation_right)

        # DISPARITY SMOOTHNESS
        self.disp_left_smoothness  = self.get_disparity_smoothness(self.disparity_left,  self.left_pyramid)
        self.disp_right_smoothness = self.get_disparity_smoothness(self.disparity_right, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE DIFF
            # L1
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]
            # SSIM
            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]
            # PHOTOMETRIC CONSISTENCY (spatial loss)
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)


            # DISPARITY DIFF
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disparity[i] - self.disparity_left[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disparity[i] - self.disparity_right[i])) for i in range(4)]
            self.disp_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

#            # POSE CONSISTENCY 
#            self.l1_translation = tf.reduce_mean(tf.abs( tf.subtract(self.translation_left, self.translation_right) ))
#            self.l1_rotation = tf.reduce_mean(tf.abs( tf.subtract(self.rotation_left, self.rotation_right) ))
#            self.pose_loss = self.l1_translation + self.l1_rotation

#            # DISPARITY SMOOTHNESS
#            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
#            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
#            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)



            # PHOTOMETRIC REGISTRATION (temporal loss)
            # L1
            self.l1_left_k_plus_one = tf.reduce_mean(tf.abs( tf.subtract(self.left_k_plus_one, self.left_next) ))
            self.l1_right_k_plus_one = tf.reduce_mean(tf.abs( tf.subtract(self.right_k_plus_one, self.right_next) ))
            self.l1_left_k = tf.reduce_mean(tf.abs( tf.subtract(self.left_k, self.left) ))
            self.l1_right_k= tf.reduce_mean(tf.abs( tf.subtract(self.right_k, self.right) ))
            self.l1_left_temporal = self.l1_left_k_plus_one + self.l1_left_k 
            self.l1_right_temporal = self.l1_right_k_plus_one + self.l1_right_k  
            # SSIM
            self.ssim_left_k_plus_one = tf.reduce_mean(self.SSIM( self.left_k_plus_one,  self.left_next))
            self.ssim_right_k_plus_one = tf.reduce_mean(self.SSIM( self.right_k_plus_one,  self.right_next))
            self.ssim_left_k = tf.reduce_mean(self.SSIM( self.left_k,  self.left))
            self.ssim_right_k = tf.reduce_mean(self.SSIM( self.right_k,  self.right))
            self.ssim_left_temporal = self.ssim_left_k_plus_one + self.ssim_left_k    
            self.ssim_right_temporal = self.ssim_right_k_plus_one + self.ssim_right_k 
          
            # TEMPORAL LOSS
            self.image_loss_left_temporal  = self.params.alpha_image_loss * self.ssim_left_temporal  + (1 - self.params.alpha_image_loss) * self.l1_left_temporal
            self.image_loss_right_temporal  = self.params.alpha_image_loss * self.ssim_right_temporal  + (1 - self.params.alpha_image_loss) * self.l1_right_temporal
            self.image_loss_temporal = self.image_loss_left_temporal + self.image_loss_right_temporal
            
            # GEOMETRIC REGISTRATION LOSS
            

            # TOTAL LOSS
            self.total_loss = self.params.image_loss_weight * self.image_loss + self.params.temporal_loss_weight * self.image_loss_temporal + self.params.disp_loss_weight * self.disp_loss 
#            self.total_loss = self.params.image_loss_weight * self.image_loss + self.params.temporal_loss_weight * self.image_loss_temporal + self.params.disp_loss_weight * self.disp_loss + self.params.pose_loss_weight * self.pose_loss 
#            self.total_loss = self.params.image_loss_weight * self.image_loss + self.params.temporal_loss_weight * self.image_loss_temporal+ self.params.gradient_loss_weight * self.disp_gradient_loss
#+ self.params.disp_loss_weight * self.disp_loss 
#+ self.params.pose_loss_weight * self.pose_loss 
#+ self.params.gradient_loss_weight * self.disp_gradient_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('image_loss', self.image_loss, collections=self.model_collection)
            tf.summary.scalar('image_loss_temporal', self.image_loss_temporal, collections=self.model_collection)
#            tf.summary.scalar('disp_loss', self.disp_loss, collections=self.model_collection)
#            tf.summary.scalar('pose_loss', self.pose_loss, collections=self.model_collection)
#            tf.summary.scalar('disp_gradient_loss', self.disp_gradient_loss, collections=self.model_collection)
            tf.summary.image('left_k_plus_one',  self.left_k_plus_one,   max_outputs=1, collections=self.model_collection)
            tf.summary.image('left', self.left,  max_outputs=1, collections=self.model_collection)
            tf.summary.image('left_next',  self.left_next,   max_outputs=1, collections=self.model_collection)
            tf.summary.image('disparity_left',  self.disparity_left[0],   max_outputs=1, collections=self.model_collection)
#            tf.summary.image('depthmap_left_next',  self.depthmap_left_next[0],   max_outputs=1, collections=self.model_collection)
            
            

#            for i in range(4):
#                tf.summary.image('left_est_'  + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
#                tf.summary.image('disparity_left' + str(i), self.disparity_left[i],  max_outputs=4, collections=self.model_collection)
#                tf.summary.image('right_to_left_disparity' + str(i),  self.right_to_left_disparity[i],   max_outputs=4, collections=self.model_collection)
#                tf.summary.image('depthmap_left' + str(i),  self.depthmap_left[i],   max_outputs=1, collections=self.model_collection)

#            if self.params.full_summary:
#                tf.summary.image('left_est', self.left_est, max_outputs=4, collections=self.model_collection)
#                tf.summary.image('right_est', self.right_est, max_outputs=4, collections=self.model_collection)
#                tf.summary.image('ssim_left', self.ssim_left,  max_outputs=4, collections=self.model_collection)
#                tf.summary.image('ssim_right', self.ssim_right, max_outputs=4, collections=self.model_collection)
#                tf.summary.image('l1_left', self.l1_left,  max_outputs=4, collections=self.model_collection)
#                tf.summary.image('l1_right', self.l1_right, max_outputs=4, collections=self.model_collection)
#                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
#                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)


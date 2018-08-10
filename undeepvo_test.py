from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

import roslib
import sys
import rospy
import tf as ros_tf
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from undeepvo_model import *
from undeepvo_dataloader import *
#from average_gradients import *

parser = argparse.ArgumentParser(description='Undeepvo TensorFlow implementation.')

parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

class undeepvo:
    def __init__(self):

        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/undeepvo/image",Image,queue_size=10)
        self.bridge = CvBridge()
        self.image_sub_left = rospy.Subscriber("/kitti/left_color_image",Image,self.callback_left)    
        self.image_sub_right = rospy.Subscriber("/kitti/right_color_image",Image,self.callback_right)

        '''Initialize refresh parameters '''
        self.is_left_in  = False
        self.is_right_in = False
        self.is_start = False
        self.cparams = np.zeros([1,9])
        self.cparams[0,0] = 7.070912000000e+02
        self.cparams[0,1] = 6.018873000000e+02
        self.cparams[0,2] = 1.831104000000e+02
        self.cparams[0,3] = 7.070912000000e+02
        self.cparams[0,4] = 6.018873000000e+02
        self.cparams[0,5] = 1.831104000000e+02
        self.cparams[0,6] = 0.537166

        '''Initialize network for the VO estimation'''
        params = undeepvo_parameters(
            height=args.input_height,
            width=args.input_width,
            baseline = 0.54,
            batch_size=2,
            num_threads=1,
            num_epochs=1,
            alpha_image_loss=0,
            image_loss_weight=1.0, disp_loss_weight=0.0, pose_loss_weight=0.0, gradient_loss_weight=0.0, temporal_loss_weight=1.0,
            full_summary=False)

        self.left  = tf.placeholder(tf.float32, [1, args.input_height, args.input_width, 3])
        self.right = tf.placeholder(tf.float32, [1, args.input_height, args.input_width, 3])
        self.left_next  = tf.placeholder(tf.float32, [1, args.input_height, args.input_width, 3])
        self.right_next = tf.placeholder(tf.float32, [1, args.input_height, args.input_width, 3])
        self.cam_params = tf.placeholder(tf.float32, [1, 9]) 
        self.model = UndeepvoModel(params, "test", self.left, self.right, self.left_next, self.right_next, self.cam_params, None)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(self.sess, args.checkpoint_path)

        br = ros_tf.TransformBroadcaster()
        br.sendTransform((0,0,0),
                         ros_tf.transformations.quaternion_from_euler(0,0,0),
                         rospy.Time.now(),
                         "/undeepvo/Current",
                         "/undeepvo/World")

    def callback_left(self,data):
        input_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#        cv2.imshow("Image window", input_image)
#        cv2.waitKey(3)

        original_height, original_width, num_channels = input_image.shape

        self.cparams[0,7] = args.input_height/original_height
        self.cparams[0,8] = args.input_width/original_width

        input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
        if self.is_start == True:
            self.img_left = self.img_left_next
        self.img_left_next = np.expand_dims(input_image.astype(np.float32) / 255, axis=0)
        self.is_left_in = True

    def callback_right(self,data):
        input_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
        if self.is_start == True:
            self.img_right = self.img_right_next
        self.img_right_next = np.expand_dims(input_image.astype(np.float32) / 255, axis=0)
        self.is_right_in = True

    def refresh(self):
        "Check new sequence is comming"
        if self.is_start == True:
            if self.is_left_in == True & self.is_right_in == True:
                self.test_simple()
                self.is_left_in = False
                self.is_right_in = False
        else: 
            if self.is_left_in == True & self.is_right_in == True:
                self.is_start = True
                self.is_left_in = False
                self.is_right_in = False


    def test_simple(self):
        """Test function."""
        [disp, tran, rot] = self.sess.run([self.model.depthmap_left[0], self.model.translation_left, self.model.rotation_left], feed_dict={self.left: self.img_left, self.right: self.img_right, self.left_next: self.img_left_next, self.right_next: self.img_right_next, self.cam_params: self.cparams})

        #Publish Depth Image
        disp = disp.squeeze()
#        print(np.amin(disp))    
        disp_to_img = scipy.misc.imresize(disp.squeeze(), [args.input_height, args.input_width])
        cv2.imshow("disp_to_img", disp_to_img)
        cv2.waitKey(3)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(disp, "32FC1"))

        #Publish R and t
        tran = tran.squeeze()
        rot  = rot.squeeze()
#        print(tran,rot)
        br = ros_tf.TransformBroadcaster()
        br.sendTransform(tran,
                         ros_tf.transformations.quaternion_from_euler(rot[0],rot[1],rot[2]),
                         rospy.Time.now(),
                         "/undeepvo/Current",
                         "/undeepvo/World")


def main(_):


    #init rospy
    rospy.init_node('undeepvo', anonymous=True)
    rate = rospy.Rate(100) # 100hz

    ic = undeepvo()
    while not rospy.is_shutdown():
        ic.refresh()
        rate.sleep()

if __name__ == '__main__':
    tf.app.run()

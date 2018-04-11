
"""Undeepvo data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class UndeepvoDataloader(object):
    """undeepvo dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.right_image_batch = None
        self.left_next_image_batch  = None
        self.right_next_image_batch = None
        self.cam_params_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values
        
        # we load only one image for test, except if we trained a stereo model
        if mode == 'test':
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            left_image_o  = self.read_image(left_image_path)
        else:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            left_next_image_path  = tf.string_join([self.data_path, split_line[2]])
            right_next_image_path = tf.string_join([self.data_path, split_line[3]])
            cam_params = tf.string_to_number(split_line[4:11])
            height_o = tf.string_to_number(split_line[11])
            width_o = tf.string_to_number(split_line[12])

            left_image_o  = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)
            left_next_image_o  = self.read_image(left_next_image_path)
            right_next_image_o = self.read_image(right_next_image_path)

            # set cam_params shape
            cam_params = tf.reshape(cam_params, [7])
            cam_params = tf.expand_dims(cam_params,0)
            h_tensor = tf.expand_dims(tf.cast(tf.constant([self.params.height]), tf.float32),0)
            w_tensor = tf.expand_dims(tf.cast(tf.constant([self.params.width]), tf.float32),0)
            cam_params = tf.squeeze(tf.concat([cam_params, h_tensor/height_o, w_tensor/width_o],1))
#            print(h_tensor/height_o)
        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)
            left_next_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_next_image_o), lambda: left_next_image_o)
            right_next_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_next_image_o),  lambda: right_next_image_o)

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            left_image, right_image, left_next_image, right_next_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image, right_image, left_next_image, right_next_image), lambda: (left_image, right_image, left_next_image, right_next_image))

            # set image shape
            left_image.set_shape( [self.params.height, self.params.width, 3])
            right_image.set_shape([self.params.height, self.params.width, 3])
            left_next_image.set_shape( [self.params.height, self.params.width, 3])
            right_next_image.set_shape([self.params.height, self.params.width, 3])
 
            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.left_image_batch, self.right_image_batch, self.left_next_image_batch, self.right_next_image_batch, self.cam_params_batch = tf.train.shuffle_batch([left_image, right_image, left_next_image, right_next_image, cam_params], params.batch_size, capacity, min_after_dequeue, params.num_threads)

        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            self.left_image_batch.set_shape( [2, None, None, 3])

    def augment_image_pair(self, left_image, right_image, left_next_image, right_next_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma
        left_next_image_aug  = left_next_image  ** random_gamma
        right_next_image_aug = right_next_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness
        left_next_image_aug  =  left_next_image_aug * random_brightness
        right_next_image_aug = right_next_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug  *= color_image
        right_image_aug *= color_image
        left_next_image_aug  *= color_image
        right_next_image_aug *= color_image

        # saturate
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)
        left_next_image_aug  = tf.clip_by_value(left_next_image_aug,  0, 1)
        right_next_image_aug = tf.clip_by_value(right_next_image_aug, 0, 1)

        return left_image_aug, right_image_aug, left_next_image_aug, right_next_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]

        image  = tf.image.convert_image_dtype(image,  tf.float32) 
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image

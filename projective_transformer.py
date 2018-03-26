from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.layers import Lambda

def projective_transformer(input_images, focal_length, c0, c1, depthmap, rotation, translation, wrap_mode='border', name='projective_transformer', **kwargs):
    def output_shape(input_shape):

        return input_shape[0]

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, [-1, _num_channels])

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, focal_length, c0, c1, depthmap, rotation, translation):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

#            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            x_unproject = Lambda(lambda x: x[1]*(x[0]-c0)/focal_length, output_shape=output_shape)([x_t_flat, depthmap])
            y_unproject = Lambda(lambda x: x[1]*(x[0]-c1)/focal_length, output_shape=output_shape)([y_t_flat, depthmap])

            x_transformed = Lambda(lambda x: r11*x[0]+r12*x[1]+r13*x[3]+t1, output_shape=output_shape)([x_unproject,y_unproject,depthmap])
            y_transformed = Lambda(lambda x: r21*x[0]+r22*x[1]+r23*x[3]+t2, output_shape=output_shape)([x_unproject,y_unproject,depthmap])
            z_transformed = Lambda(lambda x: r31*x[0]+r32*x[1]+r33*x[3]+t3, output_shape=output_shape)([x_unproject,y_unproject,depthmap])

            x_t_flat = Lambda(lambda x: c0+focal_length*x[0]/x[1], output_shape=output_shape)([x_transformed,z_transformed])
            y_t_flat = Lambda(lambda x: c1+focal_length*x[0]/x[1], output_shape=output_shape)([y_transformed,z_transformed])      

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, [_num_batch, _height, _width, _num_channels])
            return output

    with tf.variable_scope(name):
        _num_batch    = input_images.shape[0]
        _height       = input_images.shape[1]
        _width        = input_images.shape[2]
        _num_channels = input_images.shape[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, focal_length, c0, c1, depthmap, rotation, translation)
        return output

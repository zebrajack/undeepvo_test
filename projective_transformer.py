from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.layers import Lambda

def projective_transformer(input_images, f0, f1, c0, c1, depthmap, rotation, translation, wrap_mode='border', name='projective_transformer', **kwargs):

    def _output_shape0(input_shape):
        shape = list(input_shape[0])

        shape[1] = 9

        return tuple(shape)

    def _output_shape(input_shape):

        return input_shape[0]

    def _rpy_to_mat(r, p, y):
        
        zeros = tf.zeros([_num_batch], tf.float32)
        ones = tf.ones([_num_batch], tf.float32)

        yawMatrix = tf.stack([tf.cos(y), -tf.sin(y), zeros,
        tf.sin(y), tf.cos(y), zeros,
        zeros, zeros, ones], axis=1)
        yawMatrix = tf.reshape(yawMatrix, [-1, 3, 3])

        pitchMatrix = tf.stack([tf.cos(p), zeros, tf.sin(p),
        zeros, ones, zeros,
        -tf.sin(p), zeros, tf.cos(p)], axis=1)
        pitchMatrix = tf.reshape(pitchMatrix, [-1, 3, 3])

        rollMatrix = tf.stack([ones, zeros, zeros,
        zeros, tf.cos(r), -tf.sin(r),
        zeros, tf.sin(r), tf.cos(r)], axis=1)
        rollMatrix = tf.reshape(rollMatrix, [-1, 3, 3])

        R = tf.matmul(tf.matmul(yawMatrix, pitchMatrix), rollMatrix)
        return R

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
            y = tf.clip_by_value(y, 0.0,  _height_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)            
            y1 = tf.cast(tf.minimum(y1_f,  _height_f - 1 + 2 * _edge_size), tf.int32)
           
            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_00 = base_y0 + x0
            idx_01 = base_y0 + x1
            idx_10 = base_y1 + x0
            idx_11 = base_y1 + x1
            
            im_flat = tf.reshape(im, [-1, _num_channels])

            pix_00 = tf.gather(im_flat, idx_00)
            pix_01 = tf.gather(im_flat, idx_01)
            pix_10 = tf.gather(im_flat, idx_10)
            pix_11 = tf.gather(im_flat, idx_11)
            
            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)
            weight_d = tf.expand_dims(y1_f - y, 1)
            weight_u = tf.expand_dims(y - y0_f, 1)

            weight_00 = weight_l*weight_d
            weight_01 = weight_r*weight_d
            weight_10 = weight_l*weight_u
            weight_11 = weight_r*weight_u

            return weight_00 * pix_00 + weight_01 * pix_01 + weight_10 * pix_10 + weight_11 * pix_11

#    def _transform(input_images, depthmap, rotation, translation):
#        with tf.variable_scope('transform'):
#            # grid of (x_t, y_t, 1), eq (1) in ref [1]
#            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
#                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

#            x_t_flat = tf.reshape(x_t, (1, -1))
#            y_t_flat = tf.reshape(y_t, (1, -1))

#            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
#            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))
#            
#            x_t_flat = tf.reshape(x_t_flat, [-1])
#            y_t_flat = tf.reshape(y_t_flat, [-1])

#            # flatten depthmap
#            depthmap_flat = tf.reshape(depthmap,[-1])

#            # get R_mat and t_mat
#            roll = rotation[:,0]
#            pitch = rotation[:,1]
#            yaw = rotation[:,2]
#            R_mat = _rpy_to_mat(roll,pitch,yaw)
##            Rt_mat = tf.transpose(R_mat,perm=[0, 2, 1])

#            t_mat = tf.reshape(translation,(_num_batch,3,1))
#            t_mat = tf.tile(t_mat, tf.stack([1,1,_width*_height]))

# 
#            # transform to new image plan
#            x_unproject = Lambda(lambda x: x[1]*(x[0]-_c0)/_focal_length, output_shape=_output_shape)([x_t_flat, depthmap_flat])
#            y_unproject = Lambda(lambda x: x[1]*(x[0]-_c1)/_focal_length, output_shape=_output_shape)([y_t_flat, depthmap_flat])
#            
#            x_unproject = tf.reshape(x_unproject,[8,-1])
#            y_unproject = tf.reshape(y_unproject,[8,-1])
#            z_unproject = tf.reshape(depthmap_flat,[8,-1])
#            
#            xyz = tf.stack([x_unproject,y_unproject,z_unproject], axis=1)

#            xyz_transformed = tf.add(tf.matmul(R_mat,xyz),t_mat)

#            x_transformed = tf.reshape(xyz_transformed[:,0,:],[-1])
#            y_transformed = tf.reshape(xyz_transformed[:,1,:],[-1])
#            z_transformed = tf.reshape(xyz_transformed[:,2,:],[-1])

#            x_t_flat = Lambda(lambda x: _c0+_focal_length*x[0]/x[1], output_shape=_output_shape)([x_transformed,z_transformed])
#            y_t_flat = Lambda(lambda x: _c1+_focal_length*x[0]/x[1], output_shape=_output_shape)([y_transformed,z_transformed])      

#            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

#            output = tf.reshape(
#                input_transformed, [_num_batch, _height, _width, _num_channels])
#            return output

    def _transform(input_images, depthmap, rotation, translation, f0, f1, c0, c1):
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

            # flatten depthmap
            depthmap_flat = tf.reshape(depthmap,[-1])

            # get R_mat and t_mat
            roll = rotation[:,0]
            pitch = rotation[:,1]
            yaw = rotation[:,2]
            R_mat = _rpy_to_mat(roll,pitch,yaw)

            t_mat = tf.reshape(translation,(_num_batch,3,1))
            t_mat = tf.tile(t_mat, tf.stack([1,1,_width*_height]))

            # get focal_length, principal0, principal1
            f0 = tf.tile(tf.reshape(f0,(_num_batch,1)), tf.stack([1,_width*_height]))
            f1 = tf.tile(tf.reshape(f1,(_num_batch,1)), tf.stack([1,_width*_height]))
            c0 = tf.tile(tf.reshape(c0,(_num_batch,1)), tf.stack([1,_width*_height]))
            c1 = tf.tile(tf.reshape(c1,(_num_batch,1)), tf.stack([1,_width*_height]))
            f0 = tf.reshape(f0,[-1])
            f1 = tf.reshape(f1,[-1])
            c0 = tf.reshape(c0,[-1])
            c1 = tf.reshape(c1,[-1])

            # transform to new image plan
            x_unproject = Lambda(lambda x: x[1]*(x[0]-x[3])/x[2], output_shape=_output_shape)([x_t_flat, depthmap_flat, f0, c0])
            y_unproject = Lambda(lambda x: x[1]*(x[0]-x[3])/x[2], output_shape=_output_shape)([y_t_flat, depthmap_flat, f1, c1])
            
            x_unproject = tf.reshape(x_unproject,[8,-1])
            y_unproject = tf.reshape(y_unproject,[8,-1])
            z_unproject = tf.reshape(depthmap_flat,[8,-1])
            
            xyz = tf.stack([x_unproject,y_unproject,z_unproject], axis=1)

            xyz_transformed = tf.add(tf.matmul(R_mat,xyz),t_mat)

            x_transformed = tf.reshape(xyz_transformed[:,0,:],[-1])
            y_transformed = tf.reshape(xyz_transformed[:,1,:],[-1])
            z_transformed = tf.reshape(xyz_transformed[:,2,:],[-1])

            x_t_flat = Lambda(lambda x: x[3]+x[2]*x[0]/x[1], output_shape=_output_shape)([x_transformed,z_transformed,f0,c0])
            y_t_flat = Lambda(lambda x: x[3]+x[2]*x[0]/x[1], output_shape=_output_shape)([y_transformed,z_transformed,f1,c1])      

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

#        _focal_length = focal_length
#        _c0 = c0
#        _c1 = c1

        
#        output = _transform(input_images, depthmap, rotation, translation)
        output = _transform(input_images, depthmap, rotation, translation, f0, f1, c0, c1)
        return output

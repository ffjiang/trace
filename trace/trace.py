# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import subprocess
import h5py
import tifffile
import tensorflow as tf
import numpy as np
import dataprovider.transform as transform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import snemi3d
from augmentation import batch_iterator
from augmentation import alternating_example_iterator
from thirdparty.segascorus import io_utils
from thirdparty.segascorus import utils
from thirdparty.segascorus.metrics import *


FOV = 189
OUTPT = 192
INPT = 380

'''
FOV = 161
OUTPT = 160
INPT = 320
'''

FULL_FOV = 191
FULL_INPT = 702
FULL_OUTPT = 512

tmp_dir = 'tmp/vnet_2017-residuals-elastic_deform/'


def weight_variable(name, shape):
  """
  One should generally initialize weights with a small amount of noise
  for symmetry breaking, and to prevent 0 gradients.
  Since we're using ReLU neurons, it is also good practice to initialize
  them with a slightly positive initial bias to avoid "dead neurons".
  """
  #initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def bias_variable(name, shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name, initializer=initial)

def unbiased_bias_variable(name, shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name, initializer=initial)

def conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def down_conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[2, 2], padding='SAME', dilation_rate= [dilation, dilation])

def same_conv2d(x, W, dilation=1):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def max_pool(x, dilation=1, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def conv2d_transpose(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def crop(x1, x2, batch_size):
    offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
    x2_shape = tf.shape(x2)
    size = tf.pack([x2_shape[1], x2_shape[2]])
    return tf.image.extract_glimpse(x1, size=size, offsets=offsets, centered=True)

def crop_and_concat(x1, x2, batch_size):
    return tf.concat(3, [crop(x1, x2, batch_size), x2])


def dropout(x, keep_prob):
    mask = tf.ones(x.get_shape()[3])
    dropoutMask = tf.nn.dropout(mask, keep_prob)
    return x * dropoutMask

def elastic_deform(image, labels, alpha, sigma, random_state=None):
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[0:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    el_image = np.zeros(shape=image.shape)
    for i in range(image.shape[2]):
        el_image[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape)

    el_labels = np.zeros(shape=labels.shape)
    for i in range(labels.shape[2]):
        el_labels[:,:,i] = map_coordinates(labels[:,:,i], indices, order=1).reshape(shape)

    return el_image, el_labels


def createHistograms(name2var):
    listOfSummaries = []
    for name, var in name2var.iteritems():
        listOfSummaries.append(tf.summary.histogram(name, var))
    return tf.summary.merge(listOfSummaries)

# Arguments:
#   - inputs: mini-batch of input images
#   - is_training: flag specifying whether to use mini-batch or population
#   statistics
#   - decay: the decay rate used to calculate exponential moving average
def batch_norm_layer(inputs, is_training, decay=0.9):
    epsilon = 1e-5
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    offset = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, offset, scale, epsilon)

def create_unet(image, target, keep_prob, is_training, layers=5, features_root=64, kernel_size=3, learning_rate=0.0001):
    '''
    Creates a new U-Net for the given parametrization.
    
    :param x: input tensor variable, shape [?, ny, nx, channels]
    :param keep_prob: dropout probability tensor
    :param layers: number of layers in the unet
    :param features_root: number of features in the first layer
    :param kernel_size: size of the convolutional kernel
    :param learning_rate: learning rate for the optimizer
    '''

    class UNet:
        in_node = image
        batch_size = tf.shape(in_node)[0] 
        in_size = tf.shape(in_node)[1]
        size = in_size

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        upconvs = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        histogram_dict = {}
        image_summaries = []

        # down layers
        for layer in range(layers):
            num_feature_maps = 2**layer * features_root
            layer_str = 'layer' + str(layer)

            # Input layer maps a 1-channel image to num_feature_maps channels
            if layer == 0:
                w1 = weight_variable(layer_str + '_w1', [kernel_size, kernel_size, 1, num_feature_maps]) 
            else:
                w1 = weight_variable(layer_str + '_w1', [kernel_size, kernel_size, num_feature_maps // 2, num_feature_maps]) 
            w2 = weight_variable(layer_str + '_w2', [kernel_size, kernel_size, num_feature_maps, num_feature_maps]) 
            w3 = weight_variable(layer_str + '_w3', [kernel_size, kernel_size, num_feature_maps, num_feature_maps]) 
            b1 = bias_variable(layer_str + '_b1', [num_feature_maps])
            b2 = bias_variable(layer_str + '_b2', [num_feature_maps])
            b3 = bias_variable(layer_str + '_b3', [num_feature_maps])

            h_conv1 = tf.nn.elu(same_conv2d(in_node, w1) + b1)
            h_conv2 = tf.nn.elu(same_conv2d(h_conv1, w2) + b2)

            in_node_cropped = crop(in_node, h_conv2, batch_size)
            if layer == 0:
                in_node_cropped = tf.tile(in_node_cropped, (1,1,1,num_feature_maps))
            else:
                in_node_cropped = tf.tile(in_node_cropped, (1,1,1,2))

            h_conv3 = h_conv2 + in_node_cropped
            dw_h_convs[layer] = h_conv3

            weights.append((w1, w2))
            convs.append((h_conv1, h_conv2, dw_h_convs[layer]))
            histogram_dict[layer_str + '_in_node'] = in_node
            histogram_dict[layer_str + '_w1'] = w1
            histogram_dict[layer_str + '_w2'] = w2
            histogram_dict[layer_str + '_w3'] = w3
            histogram_dict[layer_str + '_b1'] = b1
            histogram_dict[layer_str + '_b2'] = b2
            histogram_dict[layer_str + '_b3'] = b3
            histogram_dict[layer_str + '_activations'] = dw_h_convs[layer]

            size -= 4

            layer_packed = computeGridSummary(dw_h_convs[layer], num_feature_maps, size)
            image_summaries.append(tf.summary.image(layer_str + '  activations', layer_packed))


            # If not the bottom layer, do a max-pool (or down-convolution)
            if layer < layers - 1:
                w_d = weight_variable(layer_str + '_wd', [2, 2, num_feature_maps, 2 * num_feature_maps])
                b_d = bias_variable(layer_str + '_bd', [2 * num_feature_maps])
                #pools[layer] = tf.nn.elu(down_conv2d(dw_h_convs[layer], w_d) + b_d)
                pools[layer] = max_pool(dw_h_convs[layer])
                in_node = pools[layer]
                size //= 2


        in_node = dw_h_convs[layers-1]

        # Up layers
        # There is one less up layer than down layer.
        for layer in range(layers - 2, -1, -1):
            layer_str = 'layer_u' + str(layer)
            num_feature_maps = 2**layer * features_root

            wu = weight_variable(layer_str + '_wu', [kernel_size, kernel_size, num_feature_maps, num_feature_maps * 2])
            bu = bias_variable(layer_str + '_bu', [num_feature_maps])
            h_upconv = tf.nn.elu(conv2d_transpose(in_node, wu, stride=2) + bu)
            h_upconv_concat = crop_and_concat(dw_h_convs[layer], h_upconv, batch_size)
            upconvs[layer] = h_upconv_concat

            w1 = weight_variable(layer_str + '_w1', [kernel_size, kernel_size, num_feature_maps * 2, num_feature_maps])
            w2 = weight_variable(layer_str + '_w2', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            w3 = weight_variable(layer_str + '_w3', [kernel_size, kernel_size, num_feature_maps, num_feature_maps])
            b1 = bias_variable(layer_str + '_b1', [num_feature_maps])
            b2 = bias_variable(layer_str + '_b2', [num_feature_maps])
            b3 = bias_variable(layer_str + '_b3', [num_feature_maps])

            h_conv1 = tf.nn.elu(same_conv2d(h_upconv_concat, w1) + b1)
            h_conv2 = tf.nn.elu(same_conv2d(h_conv1, w2) + b2)

            #h_upconv_cropped = crop(h_upconv, h_conv2, batch_size)
            skip_connect_cropped = crop(dw_h_convs[layer], h_conv2, batch_size)

            in_node = h_conv2 + skip_connect_cropped# + h_upconv_cropped
            up_h_convs[layer] = in_node

            weights.append((w1, w2, w3))
            convs.append((h_conv1, in_node))
            histogram_dict[layer_str + '_wu'] = wu
            histogram_dict[layer_str + '_bu'] = bu
            histogram_dict[layer_str + '_h_upconv'] = h_upconv
            histogram_dict[layer_str + '_h_upconv_concat'] = h_upconv_concat
            histogram_dict[layer_str + '_w1'] = w1
            histogram_dict[layer_str + '_w2'] = w2
            histogram_dict[layer_str + '_w3'] = w3
            histogram_dict[layer_str + '_b1'] = b1
            histogram_dict[layer_str + '_b2'] = b2
            histogram_dict[layer_str + '_b3'] = b3
            histogram_dict[layer_str + '_activations'] = up_h_convs[layer]

            size *= 2
            size -= 4

            layer_packed = computeGridSummary(in_node, num_feature_maps, size)
            image_summaries.append(tf.summary.image(layer_str + '  activations', layer_packed))



        # Output map
        # features_root * 2 because there is one less up layer than
        # down layer
        w_o = weight_variable('w_o', [5, 5, features_root, 2]) #* 2, 2])
        b_o = bias_variable('b_o', [2])
        prediction = same_conv2d(in_node, w_o) + b_o
        sigmoid_prediction = tf.nn.sigmoid(prediction)

        histogram_dict['prediction'] = prediction
        histogram_dict['sigmoid prediction'] = sigmoid_prediction

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,target))

        loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)

        padding = (in_size - size) // 2

        image_summaries.append(tf.summary.image('input image', image))
        mirrored_input_summary = tf.summary.image('mirrored input image', image)
        image_summaries.append(tf.summary.image('boundary output patch', image[:,padding:-padding,padding:-padding,:]))
        validation_output_patch = tf.placeholder(tf.float32)
        validation_output_patch_summary = tf.summary.image('validation output patch', validation_output_patch)
        image_summaries.append(tf.summary.image('boundary targets', target[:,:,:,:1]))
        validation_target = tf.placeholder(tf.float32)
        validation_target_summary = tf.summary.image('validation boundary targets', validation_target[:,:,:,:1])

        image_summaries.append(tf.summary.image('boundary predictions', sigmoid_prediction[:,:,:,:1]))
        validation_prediction = tf.placeholder(tf.float32)
        validation_boundary_summary = tf.summary.image('validation boundary predictions', validation_prediction[:,:,:,:1])

        binary_prediction = tf.round(sigmoid_prediction) 
        pixel_error = tf.reduce_mean(tf.cast(tf.abs(binary_prediction - target), tf.float32))
        pixel_error_summary = tf.summary.scalar('pixel_error', pixel_error)
        training_pixel_error_summary = tf.summary.scalar('training pixel_error', pixel_error)
        validation_pixel_error = tf.placeholder(tf.float32)
        validation_pixel_error_summary = tf.summary.scalar('validation pixel_error', validation_pixel_error)


        rand_f_score = tf.placeholder(tf.float32)
        rand_f_score_merge = tf.placeholder(tf.float32)
        rand_f_score_split = tf.placeholder(tf.float32)
        vi_f_score = tf.placeholder(tf.float32)
        vi_f_score_merge = tf.placeholder(tf.float32)
        vi_f_score_split = tf.placeholder(tf.float32)

        training_rand_f_score_summary = tf.summary.scalar('training rand f score', rand_f_score)
        training_rand_f_score_merge_summary = tf.summary.scalar('training rand f merge score', rand_f_score_merge)
        training_rand_f_score_split_summary = tf.summary.scalar('training rand f split score', rand_f_score_split)
        training_vi_f_score_summary = tf.summary.scalar('training vi f score', vi_f_score)
        training_vi_f_score_merge_summary = tf.summary.scalar('training vi f merge score', vi_f_score_merge)
        training_vi_f_score_split_summary = tf.summary.scalar('training vi f split score', vi_f_score_split)

        validation_rand_f_score_summary = tf.summary.scalar('validation rand f score', rand_f_score)
        validation_rand_f_score_merge_summary = tf.summary.scalar('validation rand f merge score', rand_f_score_merge)
        validation_rand_f_score_split_summary = tf.summary.scalar('validation rand f split score', rand_f_score_split)
        validation_vi_f_score_summary = tf.summary.scalar('validation vi f score', vi_f_score)
        validation_vi_f_score_merge_summary = tf.summary.scalar('validation vi f merge score', vi_f_score_merge)
        validation_vi_f_score_split_summary = tf.summary.scalar('validation vi f split score', vi_f_score_split)


        training_score_summary_op = tf.summary.merge([training_rand_f_score_summary,
                                             training_rand_f_score_merge_summary,
                                             training_rand_f_score_split_summary,
                                             training_vi_f_score_summary,
                                             training_vi_f_score_merge_summary,
                                             training_vi_f_score_split_summary
                                            ])

        validation_score_summary_op = tf.summary.merge([validation_rand_f_score_summary,
                                             validation_rand_f_score_merge_summary,
                                             validation_rand_f_score_split_summary,
                                             validation_vi_f_score_summary,
                                             validation_vi_f_score_merge_summary,
                                             validation_vi_f_score_split_summary
                                            ])


        histograms = createHistograms(histogram_dict)

        summary_op = tf.summary.merge([histograms,
                                       loss_summary,
                                       pixel_error_summary,
                                       ])

        image_summary_op = tf.summary.merge(image_summaries)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return UNet()

def computeGridSummary(h_conv, num_maps, map_size, width=16, height=0):
    cx = width
    if height == 0:
        cy = num_maps // width
    else:
        cy = height

    # Compute image summaries of the num_maps feature maps
    iy = map_size
    ix = iy
    h_conv_packed = tf.reshape(h_conv[0], tf.pack([iy, ix, num_maps]))
    iy += 4
    ix += 4
    h_conv_packed = tf.image.resize_image_with_crop_or_pad(h_conv_packed, iy, ix)
    h_conv_packed = tf.reshape(h_conv_packed, tf.pack([iy, ix, cy, cx]))
    h_conv_packed = tf.transpose(h_conv_packed, (2,0,3,1)) # cy, iy,  cx, ix
    h_conv_packed = tf.reshape(h_conv_packed, tf.pack([1, cy * iy, cx * ix, 1]))
    return h_conv_packed

def import_image(path, sess, isInput, fov):
    with h5py.File(path, 'r') as f:
        # f['main'] has shape [z,y,x]
        image_data = f['main'][:].astype(np.float32)
        if isInput:
            image_data = _mirrorAcrossBorders(image_data, fov)[:,:,:,np.newaxis] / 255.0
        else:
            image_data = np.einsum('dzyx->zyxd', image_data[0:2])
        
        image_ph = tf.placeholder(dtype=image_data.dtype,
                                  shape=image_data.shape)
        # Setting trainable=False keeps the variable out of the
        # Graphkeys.TRAINABLE_VARIABLES collection in the graph, so we won't try
        # and update it when we train. Setting colltions = [] keeps the variable
        # out of the GraphKeys.VARIABLES collection used for saving and
        # restoring checkpoints.
        image_t = tf.Variable(image_ph, trainable=False, collections=[])

        sess.run(image_t.initializer,
            feed_dict={image_ph: image_data})
        del image_data

        return image_t


def train(n_iterations=200000):
        inpt_placeholder = tf.placeholder(tf.float32, [1, INPT, INPT, 1])
        inpt = tf.get_variable('input', [1, INPT, INPT, 1])
        assign_input = tf.assign(inpt, inpt_placeholder)

        target_placeholder = tf.placeholder(tf.float32, [1, OUTPT, OUTPT, 2])
        target = tf.get_variable('target', [1, OUTPT, OUTPT, 2])
        assign_target = tf.assign(target, target_placeholder)
        
        with tf.variable_scope('foo'):
            net = create_unet(inpt, target, keep_prob=0.8, is_training=True, learning_rate=0.00001)

        print ('Run tensorboard to visualize training progress')
        with tf.Session() as sess:
            training_input = import_image(snemi3d.folder()+'training-input.h5', sess, isInput=True, fov=FOV)
            training_labels = import_image(snemi3d.folder()+'training-affinities.h5', sess, isInput=False, fov=FOV)
            validation_input = import_image(snemi3d.folder()+'validation-input.h5', sess, isInput=True, fov=FOV)
            validation_labels = import_image(snemi3d.folder()+'validation-affinities.h5', sess, isInput=False, fov=FOV)

            num_layers = training_input.get_shape()[0]
            input_size = training_input.get_shape()[1]
            output_size = training_labels.get_shape()[1]

            num_validation_layers = validation_input.get_shape()[0]
            validation_input_size = validation_input.get_shape()[1]
            validation_output_size = validation_labels.get_shape()[1]

            training_input_slice = tf.Variable(tf.zeros([1, input_size, input_size, 1]), trainable=False, collections=[], name='input-slice')
            training_label_slice = tf.Variable(tf.zeros([1, output_size, output_size, 2]), trainable=False, collections=[], name='label-slice')

            sess.run(training_input_slice.initializer)
            sess.run(training_label_slice.initializer)

            getInputSlice = []
            getLabelSlice = []
            for layer in range(num_layers):
                getInputSlice.append(training_input_slice.assign(tf.gather(training_input, [layer])))
                getLabelSlice.append(training_label_slice.assign(tf.gather(training_labels, [layer])))


            print('create training and validation nets')
            with tf.variable_scope('foo', reuse=True):
                training_net = create_unet(training_input_slice, training_label_slice, keep_prob=1.0, is_training=True)
                validation_net = create_unet(inpt, target, keep_prob=1.0, is_training=True)
            print('done creating nets')

            summary_writer = tf.train.SummaryWriter(
                           snemi3d.folder()+tmp_dir, graph=sess.graph)

            sess.run(tf.global_variables_initializer())
            padding = 40
            for step_, (inputs, affinities) in enumerate(batch_iterator(1,INPT+padding,INPT+padding)):
                step = step_

                should_deform = np.random.randint(low=0, high=4)
                if should_deform < 3:
                    sigma = np.random.randint(low=35, high=60)
                    alpha = (sigma - 35) * 80 + 1000 # Sigma mapped uniformly to the interval [1000, 3000)
                    # Scale alpha randomly
                    if should_deform == 0:
                        alpha = alpha * 0.5
                    elif should_deform == 1:
                        alpha = alpha * 0.75

                    el_inputs, el_labels = elastic_deform(inputs[0], affinities[0], alpha=alpha, sigma=sigma)
                else:
                    el_inputs = inputs[0]
                    el_labels = affinities[0]

                el_inputs = el_inputs[padding//2:-(padding//2), padding//2:-(padding//2)]
                el_labels = el_labels[FOV//2 + padding//2:-(FOV//2 + padding//2), FOV//2 + padding//2:-(FOV//2 + padding//2)]
                
                sess.run(assign_input,
                        feed_dict={inpt_placeholder: np.expand_dims(el_inputs, axis=0)})
                sess.run(assign_target,
                        feed_dict={target_placeholder: np.expand_dims(el_labels[FOV//2:-(FOV//2),FOV//2:-(FOV//2)], axis=0)})
                sess.run(net.train_step)

                if step % 10 == 0:
                    print ('step :'+str(step))
                    summary = sess.run(net.summary_op)
                
                    summary_writer.add_summary(summary, step)

                if step % 500 == 0:
                    image_summary = sess.run(net.image_summary_op)
                
                    summary_writer.add_summary(image_summary, step)

                    # Save the variables to disk.
                    save_path = net.saver.save(sess, snemi3d.folder()+tmp_dir + 'model.ckpt')
                    print("Model saved in file: %s" % save_path)


                    # Measure validation error
                    combinedPrediction = np.zeros((num_validation_layers, validation_output_size, validation_output_size, 2))
                    overlappedComputations = np.zeros((num_validation_layers, validation_output_size, validation_output_size, 2))
                    for z in xrange(num_validation_layers):
                        print('z: ' + str(z) + '/' + str(num_validation_layers))
                        for y in (range(0, validation_input_size - INPT + 1, FOV) + [validation_input_size - INPT]):
                            print('y: ' + str(y) + '/' + str(validation_input_size - INPT + 1))
                            for x in (range(0, validation_input_size - INPT + 1, FOV) + [validation_input_size - INPT]):
                                input_patch = sess.run(validation_input[z:z+1,y:y+INPT,x:x+INPT,:])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred
                                        
                                # Apply flipping and rotations

                                # Rotate 90 anti-clockwise
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.rot90(pred[0,:,:,0], 3)
                                pred[0,:,:,1] = np.rot90(pred[0,:,:,1], 3)
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                                # Rotate 180
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.rot90(pred[0,:,:,0], 2)
                                pred[0,:,:,1] = np.rot90(pred[0,:,:,1], 2)
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                                # Rotate 270
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.rot90(pred[0,:,:,0])
                                pred[0,:,:,1] = np.rot90(pred[0,:,:,1])
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                                # Fliplr
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                input_patch[0,:,:,0] = np.fliplr(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.fliplr(pred[0,:,:,0])
                                pred[0,:,:,1] = np.fliplr(pred[0,:,:,1])
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                                # Fliplr and rotate 90
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.fliplr(np.rot90(pred[0,:,:,0], 3))
                                pred[0,:,:,1] = np.fliplr(np.rot90(pred[0,:,:,1], 3))
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                                # Fliplr and rotate 180
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.fliplr(np.rot90(pred[0,:,:,0], 2))
                                pred[0,:,:,1] = np.fliplr(np.rot90(pred[0,:,:,1], 2))
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                                # Fliplr and rotate 270
                                input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                                pred = sess.run(validation_net.sigmoid_prediction,
                                            feed_dict={inpt: input_patch})
                                pred[0,:,:,0] = np.fliplr(np.rot90(pred[0,:,:,0]))
                                pred[0,:,:,1] = np.fliplr(np.rot90(pred[0,:,:,1]))
                                combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred
 
                                overlappedComputations[z,y:y+OUTPT,x:x+OUTPT,:] += np.ones((OUTPT, OUTPT, 2)) * 8

                    validation_sigmoid_prediction = np.divide(combinedPrediction, overlappedComputations)
                    validation_binary_prediction = np.round(validation_sigmoid_prediction) 
                    validation_pixel_error = np.mean(np.absolute(validation_binary_prediction - sess.run(validation_labels)))
                    validation_pixel_error_summary = sess.run(validation_net.validation_pixel_error_summary,
                                                            feed_dict={validation_net.validation_pixel_error: validation_pixel_error})

                    summary_writer.add_summary(validation_pixel_error_summary, step)

                    validation_output_patch_summary = sess.run(validation_net.validation_output_patch_summary,
                            feed_dict={validation_net.validation_output_patch: sess.run(validation_input[:,FOV//2:FOV//2+OUTPT,FOV//2:FOV//2+OUTPT,:])})
                    validation_target_summary = sess.run(validation_net.validation_target_summary,
                            feed_dict={validation_net.validation_target: sess.run(validation_labels)})
                    validation_boundary_summary = sess.run(validation_net.validation_boundary_summary,
                            feed_dict={validation_net.validation_prediction: validation_sigmoid_prediction})

                    summary_writer.add_summary(validation_output_patch_summary, step)
                    summary_writer.add_summary(validation_target_summary, step)
                    summary_writer.add_summary(validation_boundary_summary, step)

                    validation_scores = _evaluateRandError('validation', validation_sigmoid_prediction, watershed_high=0.95)
                    validation_score_summary = sess.run(net.validation_score_summary_op,
                             feed_dict={net.rand_f_score: validation_scores['Rand F-Score Full'],
                                        net.rand_f_score_merge: validation_scores['Rand F-Score Merge'],
                                        net.rand_f_score_split: validation_scores['Rand F-Score Split'],
                                        net.vi_f_score: validation_scores['VI F-Score Full'],
                                        net.vi_f_score_merge: validation_scores['VI F-Score Merge'],
                                        net.vi_f_score_split: validation_scores['VI F-Score Split'],
                                })

                    summary_writer.add_summary(validation_score_summary, step)

                if step % 1000 == 0:
                    pass
                    '''
                    # Measure training error

                    training_sigmoid_prediction = np.empty((num_layers, output_size, output_size, 2))
                    for layer in range(num_layers):
                        sess.run(getInputSlice[layer])
                        sess.run(getLabelSlice[layer])
                        training_sigmoid_prediction[layer:layer+1] = sess.run(training_net.sigmoid_prediction)


                    training_scores = _evaluateRandError('training', training_sigmoid_prediction, watershed_high=0.95)
                    training_score_summary = sess.run(training_net.training_score_summary_op,
                             feed_dict={training_net.rand_f_score: training_scores['Rand F-Score Full'],
                                        training_net.rand_f_score_merge: training_scores['Rand F-Score Merge'],
                                        training_net.rand_f_score_split: training_scores['Rand F-Score Split'],
                                        training_net.vi_f_score: training_scores['VI F-Score Full'],
                                        training_net.vi_f_score_merge: training_scores['VI F-Score Merge'],
                                        training_net.vi_f_score_split: training_scores['VI F-Score Split'],
                                })

                    summary_writer.add_summary(training_score_summary, step)
                    '''

                if step == n_iterations:
                    break


def _mirrorAcrossBorders(data, fov):
    mirrored_data = np.pad(data, [(0, 0), (fov//2, fov//2), (fov//2, fov//2)], mode='reflect')
    return mirrored_data


def _evaluateRandError(dataset, sigmoid_prediction, watershed_high=0.9, watershed_low=0.3):
    # Save affinities to temporary file
    tmp_aff_file = dataset + '-tmp-affinities.h5'
    tmp_label_file = dataset + '-tmp-labels.h5'
    ground_truth_file = dataset + '-generated-labels.h5'

    '''
    reshapedAffs = np.einsum('zyxd->dzyx', sigmoid_prediction)
    affs = np.concatenate([reshapedAffs, reshapedAffs, np.zeros(reshapedAffs.shape)])
    with h5py.File(snemi3d.folder()+tmp_dir+tmp_aff_file,'w') as output_file:
        output_file.create_dataset('main', data=affs)
    '''

    with h5py.File(snemi3d.folder()+tmp_dir+tmp_aff_file,'w') as output_file:
        output_file.create_dataset('main', shape=(3, sigmoid_prediction.shape[0], sigmoid_prediction.shape[1], sigmoid_prediction.shape[2]))
        out = output_file['main']

        reshaped_pred = np.einsum('zyxd->dzyx', sigmoid_prediction)
        out[0:2,:,:,:] = reshaped_pred


    # Do watershed segmentation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     current_dir+"/thirdparty/watershed/watershed.jl",
                     snemi3d.folder()+tmp_dir+tmp_aff_file,
                     snemi3d.folder()+tmp_dir+tmp_label_file,
                     str(watershed_high),
                     str(watershed_low)])

    # Compute rand f score
    # --------------------

    # Parameters
    calc_rand_score = True
    calc_rand_error = False
    calc_variation_score = True
    calc_variation_information = False
    relabel2d = True
    foreground_restricted = True
    split_0_segment = True
    other = None

    seg1 = io_utils.import_file(snemi3d.folder()+tmp_dir+tmp_label_file)
    seg2 = io_utils.import_file(snemi3d.folder()+ground_truth_file)
    prep = utils.parse_fns(utils.prep_fns,
                            [relabel2d, foreground_restricted])
    seg1, seg2 = utils.run_preprocessing(seg1, seg2, prep)

    om = utils.calc_overlap_matrix(seg1, seg2, split_0_segment)

    #Calculating each desired metric
    metrics = utils.parse_fns( utils.metric_fns,
                                [calc_rand_score,
                                calc_rand_error,
                                calc_variation_score,
                                calc_variation_information] )

    results = {}
    for (name,metric_fn) in metrics:
        if relabel2d:
            full_name = "2D {}".format(name)
        else:
            full_name = name

        (f,m,s) = metric_fn( om, full_name, other )
        results["{} Full".format(name)] = f
        results["{} Merge".format(name)] = m
        results["{} Split".format(name)] = s

    print('Rand F-Score Full: ' + str(results['Rand F-Score Full']))
    print('Rand F-Score Split: ' + str(results['Rand F-Score Split']))
    print('Rand F-Score Merge: ' + str(results['Rand F-Score Merge']))
    print('VI F-Score Full: ' + str(results['VI F-Score Full']))
    print('VI F-Score Split: ' + str(results['VI F-Score Split']))
    print('VI F-Score Merge: ' + str(results['VI F-Score Merge']))

    return results


def predict():
    inpt = tf.get_variable('input', shape=[1, INPT, INPT, 1])
    inpt_placeholder = tf.placeholder(tf.float32, shape=(1, INPT, INPT, 1))
    assign_input = tf.assign(inpt, inpt_placeholder)
    target = tf.get_variable('target', shape=[1, OUTPT, OUTPT, 2]) # This does not do anything - it is just for the initialization of the net.
    with tf.variable_scope('foo'):
        net = create_unet(inpt, target, keep_prob=1.0, is_training=True)
    with h5py.File(snemi3d.folder()+'test-input.h5','r') as input_file:
        inputs = input_file['main'][:].astype(np.float32) / 255.0
        mirrored_inpt = _mirrorAcrossBorders(inputs, FOV)
        num_layers = mirrored_inpt.shape[0]
        input_shape = mirrored_inpt.shape[1]
        output_shape = mirrored_inpt.shape[1] - FOV + 1
        with h5py.File(snemi3d.folder()+'test-affinities.h5','w') as output_file:
            output_file.create_dataset('main', shape=(3,) + input_file['main'].shape)
            out = output_file['main']

            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder()+tmp_dir+'model.ckpt')
                print("Model restored.")

                combinedPrediction = np.zeros((num_layers, output_shape, output_shape, 2))
                overlappedComputations = np.zeros((num_layers, output_shape, output_shape, 2))
                for z in xrange(num_layers):
                    print('z: ' + str(z) + '/' + str(num_layers))
                    for y in (range(0, input_shape - INPT + 1, FOV) + [input_shape - INPT]):
                        print('y: ' + str(y) + '/' + str(input_shape - INPT + 1))
                        for x in (range(0, input_shape - INPT + 1, FOV) + [input_shape - INPT]):
                            input_patch = np.copy(mirrored_inpt[z:z+1,y:y+INPT,x:x+INPT]).reshape(1, INPT, INPT, 1)
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred
                                    
                            # Apply flipping and rotations

                            # Rotate 90 anti-clockwise
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.rot90(pred[0,:,:,0], 3)
                            pred[0,:,:,1] = np.rot90(pred[0,:,:,1], 3)
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            # Rotate 180
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.rot90(pred[0,:,:,0], 2)
                            pred[0,:,:,1] = np.rot90(pred[0,:,:,1], 2)
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            # Rotate 270
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.rot90(pred[0,:,:,0])
                            pred[0,:,:,1] = np.rot90(pred[0,:,:,1])
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            # Fliplr
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            input_patch[0,:,:,0] = np.fliplr(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.fliplr(pred[0,:,:,0])
                            pred[0,:,:,1] = np.fliplr(pred[0,:,:,1])
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            # Fliplr and rotate 90
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.fliplr(np.rot90(pred[0,:,:,0], 3))
                            pred[0,:,:,1] = np.fliplr(np.rot90(pred[0,:,:,1], 3))
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            # Fliplr and rotate 180
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.fliplr(np.rot90(pred[0,:,:,0], 2))
                            pred[0,:,:,1] = np.fliplr(np.rot90(pred[0,:,:,1], 2))
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            # Fliplr and rotate 270
                            input_patch[0,:,:,0] = np.rot90(input_patch[0,:,:,0])
                            pred = sess.run(net.sigmoid_prediction,
                                        feed_dict={inpt: input_patch})
                            pred[0,:,:,0] = np.fliplr(np.rot90(pred[0,:,:,0]))
                            pred[0,:,:,1] = np.fliplr(np.rot90(pred[0,:,:,1]))
                            combinedPrediction[z:z+1,y:y+OUTPT,x:x+OUTPT,:] += pred

                            overlappedComputations[z,y:y+OUTPT,x:x+OUTPT,:] += np.ones((OUTPT, OUTPT, 2)) * 8
                            

                sigmoid_prediction = np.divide(combinedPrediction, overlappedComputations)
                out[:2] = np.einsum("zyxd->dzyx", sigmoid_prediction)

                tifffile.imsave(snemi3d.folder() + tmp_dir + 'test-boundaries.tif', (out[0] + out[1]) / 2)


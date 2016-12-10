# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import os
import subprocess
import h5py
import tifffile
import tensorflow as tf
import numpy as np

import snemi3d
from augmentation import batch_iterator
from augmentation import alternating_example_iterator
from thirdparty.segascorus import io_utils
from thirdparty.segascorus import utils
from thirdparty.segascorus.metrics import *


FOV = 95
OUTPT = 101
INPT = OUTPT + FOV - 1

tmp_dir = 'tmp/moreFullyConvFOV=95lr=0.0001/'


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
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name, initializer=initial)

def unbiased_bias_variable(name, shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.get_variable(name, initializer=initial)

def conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID', dilation_rate= [dilation, dilation])

def same_conv2d(x, W, dilation=None):
  return tf.nn.convolution(x, W, strides=[1, 1], padding='SAME', dilation_rate= [dilation, dilation])

def max_pool(x, dilation=None, strides=[2, 2], window_shape=[2, 2]):
  return tf.nn.pool(x, window_shape=window_shape, dilation_rate= [dilation, dilation],
                       strides=strides, padding='VALID', pooling_type='MAX')

def createHistograms(name2var):
    listOfSummaries = []
    for name, var in name2var.iteritems():
        listOfSummaries.append(tf.summary.histogram(name, var))
    return tf.summary.merge(listOfSummaries)

def create_network(learning_rate=0.0001):
    class Net:
        map1 = 100
        map2 = 100
        map3 = 100
        map4 = 100
        map5 = 100
        map6 = 100
        map7 = 100
        map8 = 100
        mapfc = 400

        # layer 0
        image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        target = tf.placeholder(tf.float32, shape=[None, None, None, 2])
        input_summary = tf.summary.image('input image', image)
        mirrored_input_summary = tf.summary.image('mirrored input image', image)
        output_patch_summary = tf.summary.image('output patch', image[:,FOV//2:-(FOV//2),FOV//2:-(FOV//2),:])
        validation_output_patch_summary = tf.summary.image('validation output patch', image[:,FOV//2:-(FOV//2),FOV//2:-(FOV//2),:])
        target_x_summary = tf.summary.image('target x affinities', target[:,:,:,:1])
        validation_target_x_summary = tf.summary.image('validation target x affinities', target[:,:,:,:1])
        target_y_summary = tf.summary.image('target y affinities', target[:,:,:,1:])

        inpt = tf.shape(image)[1]
        out = tf.shape(image)[1] - FOV + 1


        # Convolutional/max-pool module 1

        W_conv1 = weight_variable('W_conv1', [6, 6, 1, map1])
        b_conv1 = unbiased_bias_variable('b_conv1', [map1])
        h_conv1 = tf.nn.elu(conv2d(image, W_conv1, dilation=1) + b_conv1)

        inpt -= 5
        h_conv1_packed = computeGridSummary(h_conv1, map1, inpt)
        h_conv1_image_summary = tf.summary.image('Layer 1 activations', h_conv1_packed)

        inpt -= 6
        W_conv2 = weight_variable('W_conv2', [7, 7, map1, map2])
        b_conv2 = unbiased_bias_variable('b_conv2', [map2])
        h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2, dilation=1) + b_conv2)

        h_conv2_packed = computeGridSummary(h_conv2, map2, inpt)
        h_conv2_image_summary = tf.summary.image('Layer 2 activations', h_conv2_packed)

        #h_pool1 = max_pool(h_conv2, strides=[1,1], dilation=1)


        # Convolutional/max-pool module 2

        W_conv3 = weight_variable('W_conv3', [7, 7, map2, map3])
        b_conv3 = unbiased_bias_variable('b_conv3', [map3])
        h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3, dilation=1) + b_conv3)

        inpt -= 6
        h_conv3_packed = computeGridSummary(h_conv3, map3, inpt)
        h_conv3_image_summary = tf.summary.image('Layer 3 activations', h_conv3_packed)

        W_conv4 = weight_variable('W_conv4', [7, 7, map3, map4])
        b_conv4 = unbiased_bias_variable('b_conv4', [map4])
        h_conv4 = tf.nn.elu(conv2d(h_conv3, W_conv4, dilation=1) + b_conv4)

        inpt -= 6
        h_conv4_packed = computeGridSummary(h_conv4, map4, inpt)
        h_conv4_image_summary = tf.summary.image('Layer 4 activations', h_conv4_packed)

        h_pool2 = max_pool(h_conv4, strides=[1,1], dilation=1)
        inpt -= 1


        # Convolutional/max-pool module 3

        W_conv5 = weight_variable('W_conv5', [7, 7, map4, map5])
        b_conv5 = unbiased_bias_variable('b_conv5', [map5])
        h_conv5 = tf.nn.elu(conv2d(h_pool2, W_conv5, dilation=2) + b_conv5)

        inpt -= 2 * 6
        h_conv5_packed = computeGridSummary(h_conv5, map5, inpt)
        h_conv5_image_summary = tf.summary.image('Layer 5 activations', h_conv5_packed)

        W_conv6 = weight_variable('W_conv6', [7, 7, map5, map6])
        b_conv6 = unbiased_bias_variable('b_conv6', [map6])
        h_conv6 = tf.nn.elu(conv2d(h_conv5, W_conv6, dilation=2) + b_conv6)

        inpt -= 2 * 6
        h_conv6_packed = computeGridSummary(h_conv6, map6, inpt)
        h_conv6_image_summary = tf.summary.image('Layer 6 activations', h_conv6_packed)

        #h_pool3 = max_pool(h_conv6, strides=[1,1], dilation=4)


        # Convolutional/max-pool module 4

        W_conv7 = weight_variable('W_conv7', [7, 7, map6, map7])
        b_conv7 = unbiased_bias_variable('b_conv7', [map7])
        h_conv7 = tf.nn.elu(conv2d(h_conv6, W_conv7, dilation=2) + b_conv7)

        inpt -= 2 * 6
        h_conv7_packed = computeGridSummary(h_conv7, map7, inpt)
        h_conv7_image_summary = tf.summary.image('Layer 7 activations', h_conv7_packed)

        W_conv8 = weight_variable('W_conv8', [7, 7, map7, map8])
        b_conv8 = unbiased_bias_variable('b_conv8', [map8])
        h_conv8 = tf.nn.elu(conv2d(h_conv7, W_conv8, dilation=2) + b_conv8)

        inpt -= 2 * 6
        h_conv8_packed = computeGridSummary(h_conv8, map8, inpt)
        h_conv8_image_summary = tf.summary.image('Layer 8 activations', h_conv8_packed)

        inpt -= 2
        h_pool4 = max_pool(h_conv8, strides=[1,1], dilation=2)


        # Fully-connected layer 1
        W_fc1 = weight_variable('W_fc1', [6, 6, map8, mapfc])
        b_fc1 = unbiased_bias_variable('b_fc1', [mapfc])
        h_fc1 = tf.nn.elu(conv2d(h_pool4, W_fc1, dilation=4) + b_fc1)

        # Compute image summaries of the 48 feature maps
        cx = 20
        cy = 20
        iy = out
        ix = out
        h_fc1_packed = tf.reshape(h_fc1[0], (iy, ix, mapfc))
        iy += 4
        ix += 4
        h_fc1_packed = tf.image.resize_image_with_crop_or_pad(h_fc1_packed, iy, ix)
        h_fc1_packed = tf.reshape(h_fc1_packed, (iy, ix, cy, cx))
        h_fc1_packed = tf.transpose(h_fc1_packed, (2,0,3,1)) # cy, iy,  cx, ix
        h_fc1_packed = tf.reshape(h_fc1_packed, (1, cy * iy, cx * ix, 1))
        h_fc1_image_summary = tf.summary.image('Layer fc1 activations', h_fc1_packed)


        # Fully connected layer 2
        W_fc2 = weight_variable('W_fc2', [1, 1, mapfc, 2])
        b_fc2 = unbiased_bias_variable('b_fc2', [2])
        prediction = conv2d(h_fc1, W_fc2, dilation=4) + b_fc2

        sigmoid_prediction = tf.nn.sigmoid(prediction)

        x_affinity_summary = tf.summary.image('x-affinity predictions', sigmoid_prediction[:,:,:,:1])
        validation_x_affinity_summary = tf.summary.image('validation x-affinity predictions', sigmoid_prediction[:,:,:,:1])
        y_affinity_summary = tf.summary.image('y-affinity predictions', sigmoid_prediction[:,:,:,1:])

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,target))
        #cross_entropy = tf.reduce_mean(tf.mul(sigmoid_cross_entropy, (target - 1) * (-9) + 1))
        loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)

        binary_prediction = tf.round(sigmoid_prediction) 
        pixel_error = tf.reduce_mean(tf.cast(tf.abs(binary_prediction - target), tf.float32))
        pixel_error_summary = tf.summary.scalar('pixel_error', pixel_error)
        training_pixel_error_summary = tf.summary.scalar('training pixel_error', pixel_error)
        validation_pixel_error_summary = tf.summary.scalar('validation pixel_error', pixel_error)
        avg_affinity = tf.reduce_mean(tf.cast(target, tf.float32))
        avg_affinity_summary = tf.summary.scalar('average_affinity', avg_affinity)


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

        val_rand_f_score_summary = tf.summary.scalar('validation rand f score', rand_f_score)
        val_rand_f_score_merge_summary = tf.summary.scalar('validation rand f merge score', rand_f_score_merge)
        val_rand_f_score_split_summary = tf.summary.scalar('validation rand f split score', rand_f_score_split)
        val_vi_f_score_summary = tf.summary.scalar('validation vi f score', vi_f_score)
        val_vi_f_score_merge_summary = tf.summary.scalar('validation vi f merge score', vi_f_score_merge)
        val_vi_f_score_split_summary = tf.summary.scalar('validation vi f split score', vi_f_score_split)


        training_score_summary_op = tf.summary.merge([training_rand_f_score_summary,
                                             training_rand_f_score_merge_summary,
                                             training_rand_f_score_split_summary,
                                             training_vi_f_score_summary,
                                             training_vi_f_score_merge_summary,
                                             training_vi_f_score_split_summary
                                            ])

        validation_score_summary_op = tf.summary.merge([val_rand_f_score_summary,
                                             val_rand_f_score_merge_summary,
                                             val_rand_f_score_split_summary,
                                             val_vi_f_score_summary,
                                             val_vi_f_score_merge_summary,
                                             val_vi_f_score_split_summary
                                            ])

        histograms = createHistograms({
            'W_conv1 weights'       :   W_conv1,
            'b_conv1 biases'        :   b_conv1,
            'h_conv1 activations'   :   h_conv1,
            'W_conv2 weights'       :   W_conv2,
            'b_conv2 biases'        :   b_conv2,
            'h_conv2 activations'   :   h_conv2,
            'W_conv3 weights'       :   W_conv3,
            'b_conv3 biases'        :   b_conv3,
            'h_conv3 activations'   :   h_conv3,
            'W_conv4 weights'       :   W_conv4,
            'b_conv4 biases'        :   b_conv4,
            'h_conv4 activations'   :   h_conv4,
            'W_conv5 weights'       :   W_conv5,
            'b_conv5 biases'        :   b_conv5,
            'h_conv5 activations'   :   h_conv5,
            'W_conv6 weights'       :   W_conv6,
            'b_conv6 biases'        :   b_conv6,
            'h_conv6 activations'   :   h_conv6,
            'W_conv7 weights'       :   W_conv7,
            'b_conv7 biases'        :   b_conv7,
            'h_conv7 activations'   :   h_conv7,
            'W_conv8 weights'       :   W_conv8,
            'b_conv8 biases'        :   b_conv8,
            'h_conv8 activations'   :   h_conv8,
            'W_fc1 weights'         :   W_fc1,
            'b_fc1 biases'          :   b_fc1,
            'h_fc1 activations'     :   h_fc1,
            'W_fc2 weights'         :   W_fc2,
            'b_fc2 biases'          :   b_fc2,
            'prediction activations':   prediction,
            'sigmoid prediction activations': sigmoid_prediction
        })

        summary_op = tf.summary.merge([histograms,
                                       loss_summary,
                                       pixel_error_summary,
                                       avg_affinity_summary
                                       ])

        image_summary_op = tf.summary.merge([input_summary,
                                             output_patch_summary,
                                             target_x_summary,
                                             target_y_summary,
                                             x_affinity_summary,
                                             y_affinity_summary,
                                             h_conv1_image_summary,
                                             h_conv2_image_summary,
                                             h_conv3_image_summary,
                                             h_conv4_image_summary,
                                             h_conv5_image_summary,
                                             h_conv6_image_summary,
                                             h_fc1_image_summary
                                             ])

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return Net()

def computeGridSummary(h_conv, num_maps, map_size):
    # Compute image summaries of the num_maps feature maps
    cx = 10
    cy = num_maps // 10
    iy = map_size
    ix = iy
    h_conv_packed = tf.reshape(h_conv[0], (iy, ix, num_maps))
    iy += 4
    ix += 4
    h_conv_packed = tf.image.resize_image_with_crop_or_pad(h_conv_packed, iy, ix)
    h_conv_packed = tf.reshape(h_conv_packed, (iy, ix, cy, cx))
    h_conv_packed = tf.transpose(h_conv_packed, (2,0,3,1)) # cy, iy,  cx, ix
    h_conv_packed = tf.reshape(h_conv_packed, (1, cy * iy, cx * ix, 1))
    return h_conv_packed

def import_image(path, sess, isInput):
    with h5py.File(path, 'r') as f:
        # f['main'] has shape [z,y,x]
        image_data = f['main'][:].astype(np.float32)
        if isInput:
            image_data = _mirrorAcrossBorders(image_data / 255.0, FOV)[:,:,:,np.newaxis]
        else:
            # Is label
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
        with tf.variable_scope('foo'):
            net = create_network()

        print ('Run tensorboard to visualize training progress')
        with tf.Session() as sess:
            training_input = import_image(snemi3d.folder()+'training-input.h5', sess, isInput=True)
            training_labels = import_image(snemi3d.folder()+'training-affinities.h5', sess, isInput=False)
            validation_input = import_image(snemi3d.folder()+'validation-input.h5', sess, isInput=True)
            validation_labels = import_image(snemi3d.folder()+'validation-affinities.h5', sess, isInput=False)

            summary_writer = tf.train.SummaryWriter(
                           snemi3d.folder()+tmp_dir, graph=sess.graph)

            sess.run(tf.global_variables_initializer())
            for step, (inputs, affinities) in enumerate(batch_iterator(FOV,OUTPT,INPT)):
                sess.run(net.train_step, 
                        feed_dict={net.image: inputs,
                                   net.target: affinities})

                if step % 10 == 0:
                    print ('step :'+str(step))
                    summary = sess.run(net.summary_op,
                        feed_dict={net.image: inputs,
                                   net.target: affinities})
                
                    summary_writer.add_summary(summary, step)

                if step % 1000 == 0:
                    image_summary = sess.run(net.image_summary_op,
                        feed_dict={net.image: inputs,
                                   net.target: affinities})
                
                    summary_writer.add_summary(image_summary, step)

                    # Save the variables to disk.
                    save_path = net.saver.save(sess, snemi3d.folder()+tmp_dir + 'model.ckpt')
                    print("Model saved in file: %s" % save_path)

                    # Measure training errror

                    num_training_layers = sess.run(tf.shape(training_input))[0]
                    output_size = sess.run(tf.shape(training_labels))[1]
                    training_sigmoid_prediction = np.zeros(shape=(num_training_layers, output_size, output_size, 2))
                    for z in range(num_training_layers):
                        training_sigmoid_prediction[z:z+1] = sess.run(net.sigmoid_prediction,
                                feed_dict={net.image: sess.run(tf.slice(training_input, [z,0,0,0], [1,-1,-1,-1])),
                                           net.target: sess.run(tf.slice(training_labels, [z,0,0,0], [1,-1,-1,-1]))})

                    training_scores = _evaluateRandError('training', training_sigmoid_prediction, watershed_high=0.95)
                    training_score_summary = sess.run(net.training_score_summary_op,
                             feed_dict={net.rand_f_score: training_scores['Rand F-Score Full'],
                                        net.rand_f_score_merge: training_scores['Rand F-Score Merge'],
                                        net.rand_f_score_split: training_scores['Rand F-Score Split'],
                                        net.vi_f_score: training_scores['VI F-Score Full'],
                                        net.vi_f_score_merge: training_scores['VI F-Score Merge'],
                                        net.vi_f_score_split: training_scores['VI F-Score Split'],
                                })

                    summary_writer.add_summary(training_score_summary, step)

                    # Measure validation error

                    validation_sigmoid_prediction, validation_pixel_error_summary, mirrored_input_summary, validation_output_patch_summary, validation_target_x_summary, validation_x_affinity_summary = \
                            sess.run([net.sigmoid_prediction, net.validation_pixel_error_summary, net.mirrored_input_summary,
                                        net.validation_output_patch_summary, net.validation_target_x_summary, net.validation_x_affinity_summary],
                                feed_dict={net.image: sess.run(validation_input),
                                           net.target: sess.run(validation_labels)})

                    summary_writer.add_summary(mirrored_input_summary, step)
                    summary_writer.add_summary(validation_pixel_error_summary, step)
                    summary_writer.add_summary(validation_output_patch_summary, step)
                    summary_writer.add_summary(validation_target_x_summary, step)
                    summary_writer.add_summary(validation_x_affinity_summary, step)

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

                if step == n_iterations:
                    break

def evaluatePixelError(dataset):
    from tqdm import tqdm
    with h5py.File(snemi3d.folder()+dataset+'-input.h5','r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        with h5py.File(snemi3d.folder()+dataset+'-affinities.h5','r') as label_file:
            inputShape = inpt.shape[1]
            outputShape = inpt.shape[1] - FOV + 1
            labels = label_file['main']

            with tf.variable_scope('foo'):
                net = create_network()
            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder()+tmp_dir+'model.ckpt')
                print("Model restored.")

                #TODO pad the image with zeros so that the ouput covers the whole dataset
                totalPixelError = 0.0
                for z in tqdm(xrange(inpt.shape[0])):
                    print ('z: {} of {}'.format(z,inpt.shape[0]))
                    reshapedLabel = np.einsum('dzyx->zyxd', labels[0:2,z:z+1,FOV//2:FOV//2+outputShape,FOV//2:FOV//2+outputShape])

                    pixelError = sess.run(net.pixel_error,
                            feed_dict={net.image: inpt[z].reshape(1, inputShape, inputShape, 1),
                                       net.target: reshapedLabel})

                    totalPixelError += pixelError

                print('Average pixel error: ' + str(totalPixelError / inpt.shape[0]))


def _mirrorAcrossBorders(data, fov):
    mirrored_data = np.pad(data, [(0, 0), (fov//2, fov//2), (fov//2, fov//2)], mode='reflect')
    return mirrored_data


def _evaluateRandError(dataset, sigmoid_prediction, watershed_high=0.9, watershed_low=0.3):
    # Save affinities to temporary file
    tmp_aff_file = dataset + '-tmp-affinities.h5'
    tmp_label_file = dataset + '-tmp-labels.h5'
    ground_truth_file = dataset + '-labels.h5'

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
    with h5py.File(snemi3d.folder()+'train-input.h5','r') as input_file:
        inpt = input_file['main'][:].astype(np.float32) / 255.0
        mirrored_inpt = _mirrorAcrossBorders(inpt, FOV)
        num_layers = mirrored_inpt.shape[0]
        input_shape = mirrored_inpt.shape[1]
        output_shape = mirrored_inpt.shape[1] - FOV + 1
        with h5py.File(snemi3d.folder()+'train-gen-affinities.h5','w') as output_file:
            output_file.create_dataset('main', shape=(3,)+input_file['main'].shape)
            out = output_file['main']

            with tf.variable_scope('foo'):
                net = create_network()
            with tf.Session() as sess:
                # Restore variables from disk.
                net.saver.restore(sess, snemi3d.folder()+tmp_dir+'model.ckpt')
                print("Model restored.")

                for z in range(num_layers):
                    pred = sess.run(net.sigmoid_prediction,
                            feed_dict={net.image: mirrored_inpt[z].reshape(1, input_shape, input_shape, 1)})
                    reshaped_pred = np.einsum('zyxd->dzyx', pred)
                    out[0:2,z] = reshaped_pred[:,0]
                
                tifffile.imsave(snemi3d.folder()+tmp_dir+'train-boundaries.tif', (out[0] + out[1]) / 2)


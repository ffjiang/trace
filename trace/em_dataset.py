from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

# I/O
import tifffile as tiff
import cremi.io as cremiio
import h5py

import download_data as down

from augmentation import *
from utils import *


class Dataset(object):
    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        raise NotImplementedError

    @staticmethod
    def prepare_predictions_for_neuroglancer(results_folder, split, predictions, label_type):
        """Prepare the provided labels to be visualized using Neuroglancer.

        :param label_type: The format of the predictions passed in
        :param results_folder: The location where we should save the h5 file
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'split']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        predictions = convert_between_label_types(label_type, SEGMENTATION_3D, predictions[0])
        # Create an affinities file
        with h5py.File(results_folder + split + '-predictions.h5', 'w') as output_file:
            output_file.create_dataset('main', shape=predictions.shape)

            # Reformat our predictions
            out = output_file['main']

            # Reshape and set in out
            '''
            for i in range(predictions.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(predictions[i], axis=0))
                out[0:2, i] = reshaped_pred[:, 0]
            '''
            out[:] = predictions

    @staticmethod
    def prepare_predictions_for_neuroglancer_affinities(results_folder, split, predictions, label_type):
        """Prepare the provided affinities to be visualized using Neuroglancer.

        :param label_type: The format of the predictions passed in
        :param results_folder: The location where we should save the h5 file
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'split']
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        pred_affinities = convert_between_label_types(label_type, AFFINITIES_3D, predictions[0])
        sha = pred_affinities.shape
        # Create an affinities file
        with h5py.File(results_folder + split + '-pred-affinities.h5', 'w') as output_file:
            # Create the dataset in the file
            new_shape = (3, sha[0], sha[1], sha[2])

            output_file.create_dataset('main', shape=new_shape)

            # Reformat our predictions
            out = output_file['main']

            '''
            for i in range(pred_affinities.shape[0]):
                reshaped_pred = np.einsum('zyxd->dzyx', np.expand_dims(pred_affinities[i], axis=0))
                # Copy over as many affinities as we got
                out[0:sha[3], i] = reshaped_pred[:, 0]
            '''
            reshaped_pred = np.einsum('zyxd->dzyx', pred_affinities)
            # Copy over as many affinities as we got
            out[0:sha[3]] = reshaped_pred[:]


class ISBIDataset(Dataset):
    label_type = BOUNDARIES

    def __init__(self, data_folder):
        """Wrapper for the ISBI dataset. The ISBI dataset as downloaded (via download_data.py) is as follows:

        train_input.tif: Training data for a stack of EM images from fish in the shape [num_images, x_size, y_size],
        where each value is found on the interval [0, 255], representing a stack of greyscale images.

        train_labels.tif: Training labels for the above EM images. Represent the ground truth of where the boundaries of
        all structures in the EM images exist.

        validation_input.tif, validation_labels.tif: Same as above, except represent a partitioned subset of the
        original training set.

        test_input.tif: Labelless testing images in the same format as above.

        :param data_folder: Path to where the ISBI data is found
        """
        self.data_folder = data_folder

        self.train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
        self.train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF) / 255.0

        self.validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
        self.validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF) / 255.0

        self.test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)

    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        """Prepare the provided labels

        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param results_folder: The location where we should save the dataset
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions)
        tiff.imsave(results_folder + split + '-predictions.tif', trans_predictions)


class SNEMI3DDataset(Dataset):
    label_type = SEGMENTATION_3D

    def __init__(self, data_folder):
        """Wrapper for the SNEMI3D dataset. The SNEMI3D dataset as downloaded (via download_data.py) is as follows:

        train_input.tif: Training data for a stack of EM images from fly in the shape [num_images, x_size, y_size],
        where each value is found on the interval [0, 255], representing a stack of greyscale images.

        train_labels.tif: Training labels for the above EM images. Represent the ground truth segmentation for the
        training data (each region has a unique ID).

        validation_input.tif, validation_labels.tif: Same as above, except represent a partitioned subset of the
        original training set.

        test_input.tif: Labelless testing images in the same format as above.

        :param data_folder: Path to where the SNEMI3D data is found
        """

        self.data_folder = data_folder

        self.train_inputs = tiff.imread(data_folder + down.TRAIN_INPUT + down.TIF)
        self.train_labels = tiff.imread(data_folder + down.TRAIN_LABELS + down.TIF)

        self.validation_inputs = tiff.imread(data_folder + down.VALIDATION_INPUT + down.TIF)
        self.validation_labels = tiff.imread(data_folder + down.VALIDATION_LABELS + down.TIF)

        self.test_inputs = tiff.imread(data_folder + down.TEST_INPUT + down.TIF)

    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type):
        """Prepare the provided labels to submit on the SNEMI3D competition.

        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param results_folder: The location where we should save the dataset
        :param predictions: Predictions for labels in some format, dictated by label_type
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions)
        tiff.imsave(results_folder + split + '-predictions.tif', trans_predictions)


class CREMIDataset(Dataset):
    label_type = SEGMENTATION_3D

    def __init__(self, data_folder):
        """Wrapper for the CREMI dataset. The CREMI dataset as downloaded (via download_data.py) is as follows:

        train.hdf: Training data for a stack of EM images from fly. Can be used to derive the inputs,
        in the shape [num_images, x_size, y_size] where each value is found on the interval [0, 255] representing
        a stack of greyscale images, and the labels, in the shape [num_images, x_size, y_size] where each value
        represents the unique id of the object at that position.

        validation.hdf: Same as above, except represents a partitioned subset of the original training set.

        test.hdf: Labelless testing set in the same format as above.

        :param data_folder: Path to where the CREMI data is found
        """
        self.data_folder = data_folder

        train_file = cremiio.CremiFile(data_folder + 'train.hdf', 'r')
        self.train_inputs = train_file.read_raw().data.value
        self.train_labels = train_file.read_neuron_ids().data.value
        train_file.close()

        validation_file = cremiio.CremiFile(data_folder + 'validation.hdf', 'r')
        self.validation_inputs = validation_file.read_raw().data.value
        self.validation_labels = validation_file.read_neuron_ids().data.value
        validation_file.close()

        # TODO(beisner): Decide if we need to load the test file every time (probably don't)

        test_file = cremiio.CremiFile(data_folder + 'test.hdf', 'r')
        self.test_inputs = test_file.read_raw().data.value
        test_file.close()


    def prepare_predictions_for_submission(self, results_folder, split, predictions, label_type, ground_truth=None):
        """Prepare a given segmentation prediction for submission to the CREMI competiton

        :param label_type: The type of label given in predictions (i.e. affinities-2d, boundaries, etc)
        :param results_folder: The location where we should save the dataset.
        :param split: The name of partition of the dataset we are predicting on ['train', 'validation', 'test']
        :param predictions: Predictions for labels in some format, dictated by label_type
        :param ground_truth: Ground truth labels, if preparing training or validatio data.
        """
        trans_predictions = convert_between_label_types(label_type, self.label_type, predictions, ground_truth=ground_truth)

        # Get the input we used
        input_file = cremiio.CremiFile(self.data_folder + split + '.hdf', 'r')
        raw = input_file.read_raw()
        inputs = raw.data.value
        resolution = raw.resolution

        input_file.close()

        pred_file = cremiio.CremiFile(results_folder + split + '-predictions.hdf', 'w')
        pred_file.write_raw(cremiio.Volume(inputs, resolution=resolution))
        pred_file.write_neuron_ids(cremiio.Volume(trans_predictions, resolution=resolution))
        pred_file.close()


class EMDatasetSampler(object):

    def __init__(self, dataset, input_size, z_input_size, batch_size=1, label_output_type=BOUNDARIES, flood_filling_mode=False):
        """Helper for sampling an EM dataset. The field self.training_example_op is the only field that should be
        accessed outside this class for training.

        :param input_size: The size of the field of view
        :param z_input_size: The size of the field of view in the z-direction
        :param batch_size: The number of images to stack together in a batch
        :param dataset: An instance of Dataset, namely SNEMI3DDataset, ISBIDataset, or CREMIDataset
        :param label_output_type: The format in which the dataset labels should be sampled, i.e. for training, taking
        on values 'boundaries', 'affinities-2d', etc.
        """

        # All inputs and labels come in with the shape: [n_images, x_dim, y_dim]
        # In order to generalize we, expand into 5 dimensions: [batch_size, z_dim, x_dim, y_dim, n_channels]

        # Extract the inputs and labels from the dataset
        self.__train_inputs = expand_3d_to_5d(dataset.train_inputs)
        self.__train_labels = expand_3d_to_5d(dataset.train_labels)
        self.__train_targets = convert_between_label_types(dataset.label_type, label_output_type,
                                                           expand_3d_to_5d(dataset.train_labels))

        # Crop to get rid of edge affinities
        self.__train_inputs = self.__train_inputs[:, 1:, 1:, 1:, :]
        self.__train_labels = self.__train_labels[:, 1:, 1:, 1:, :]
        self.__train_targets = self.__train_targets[:, 1:, 1:, 1:, :]

        self.__validation_inputs = expand_3d_to_5d(dataset.validation_inputs)
        self.__validation_labels = expand_3d_to_5d(dataset.validation_labels)
        self.__validation_targets = convert_between_label_types(dataset.label_type, label_output_type,
                expand_3d_to_5d(dataset.validation_labels))

        # Crop to get rid of edge affinities
        self.__validation_inputs = self.__validation_inputs[:, 1:, 1:, 1:, :]
        self.__validation_labels = self.__validation_labels[:, 1:, 1:, 1:, :]
        self.__validation_targets = self.__validation_targets[:, 1:, 1:, 1:, :]

        self.__test_inputs = expand_3d_to_5d(dataset.test_inputs)

        # Crop to get rid of edge affinities
        self.__test_inputs = self.__test_inputs[:, 1:, 1:, 1:, :] 

        # Stack the inputs and labels, so when we sample we sample corresponding labels and inputs
        train_stacked = np.concatenate((self.__train_inputs, self.__train_labels), axis=CHANNEL_AXIS)

        # Define inputs to the graph
        crop_pad = input_size // 10 * 4
        z_crop_pad = z_input_size // 4 * 2
        patch_size = input_size + crop_pad
        z_patch_size = z_input_size + z_crop_pad

        # Create dataset, and pad the dataset with mirroring
        self.__padded_dataset = np.pad(train_stacked, [[0, 0], [z_crop_pad, z_crop_pad], [crop_pad, crop_pad], [crop_pad, crop_pad], [0, 0]], mode='reflect')

        # Create dataset consisting of bad slices from the test set (for CREMI
        # A+ only atm)
        bad_slices = [24, 36, 69, 115, 116, 144, 145]
        #bad_slices = [0]
        self.__bad_data = self.__test_inputs[0, bad_slices]
        
        with tf.device('/cpu:0'):
            # The dataset is loaded into a constant variable from a placeholder
            # because a tf.constant cannot hold a dataset that is over 2GB.
            self.__image_ph = tf.placeholder(dtype=tf.float32, shape=self.__padded_dataset.shape)
            self.__dataset_constant = tf.Variable(self.__image_ph, trainable=False, collections=[])

            self.__bad_data_ph = tf.placeholder(dtype=tf.float32, shape=self.__bad_data.shape)
            self.__bad_data_constant = tf.Variable(self.__bad_data_ph, trainable=False, collections=[])

            # Sample and squeeze the dataset in multiple batches, squeezing so that we can perform the distortions
            crop_size = [1, z_patch_size, patch_size, patch_size, train_stacked.shape[4]]
            samples = []
            for i in range(batch_size):
                samples.append(tf.random_crop(self.__dataset_constant, size=crop_size))

            sample = tf.squeeze(samples, axis=1)

            # Flip a coin, and apply an op to sample (sample can be 5d or 4d)
            # Prob is the denominator of the probability (1 in prob chance)
            def randomly_map_and_apply_op(data, op, prob=2):
                should_apply = tf.random_uniform(shape=(), minval=0, maxval=prob, dtype=tf.int32)

                def tf_if(ex):
                    return tf.cond(tf.equal(0, should_apply), lambda: op(ex), lambda: ex)

                return tf.map_fn(tf_if, data)

            def randomly_apply_op(data, op, prob=2):
                should_apply = tf.random_uniform(shape=(), minval=0, maxval=prob, dtype=tf.int32)

                return tf.cond(tf.equal(0, should_apply), lambda: op(data), lambda: data)

            # Perform random mirroring, by applying the same mirroring to each image in the stack
            def mirror_each_image_in_stack_op(stack):
                return tf.map_fn(lambda img: tf.image.flip_left_right(img), stack)
            mirrored_sample = randomly_map_and_apply_op(sample, mirror_each_image_in_stack_op)

            # Randomly flip the 3D shape upside down
            flipped_sample = randomly_map_and_apply_op(mirrored_sample, lambda stack: tf.reverse(stack, axis=[0]))

            # Apply a random rotation to each stack
            def apply_random_rotation_to_stack(stack):
                # Get the random angle
                angle = tf.random_uniform(shape=(), minval=0, maxval=2 * math.pi)

                # Rotate each image by that angle
                return tf.map_fn(lambda img: tf.contrib.image.rotate(img, angle), stack)

            rotated_sample = tf.map_fn(apply_random_rotation_to_stack, flipped_sample)

            # IDEALLY, we'd have elastic deformation here, but right now too much overhead to compute
            # elastically_deformed_sample = tf.elastic_deformation(rotated_sample)
            elastically_deformed_sample = rotated_sample

            # Separate the image from the labels
            deformed_inputs = elastically_deformed_sample[:, :, :, :, :1]
            deformed_labels = elastically_deformed_sample[:, :, :, :, 1:]

            # Apply random gaussian blurring to the image
            def apply_random_blur_to_stack(stack):
                def apply_random_blur_to_slice(img):
                    sigma = tf.random_uniform(shape=(), minval=2, maxval=5, dtype=tf.float32)
                    return tf_gaussian_blur(img, sigma, size=5)

                return tf.map_fn(lambda img: randomly_apply_op(img, apply_random_blur_to_slice, prob=5), stack)

            blurred_inputs = tf.map_fn(lambda stack: apply_random_blur_to_stack(stack),
                                       deformed_inputs)

            # Apply missing data augmentation (black slices)
            def apply_random_missing_data(stack):
                def apply_missing_slice(img):
                    return tf.zeros(tf.shape(img))
                return tf.map_fn(lambda img: randomly_apply_op(img, apply_missing_slice, prob=50), stack)

            missing_data_inputs = tf.map_fn(lambda stack: apply_random_missing_data(stack), blurred_inputs)

            # Apply bad data augmentation (random input slices inserted
            # into input patch)
            def apply_random_bad_data(stack):
                z_size = tf.shape(stack)[0]
                # Choose a random slice in the stack to be replaced by a bad
                # slice.
                slice_to_apply_to = tf.random_uniform(shape=(), minval=z_crop_pad // 2, maxval=z_size - (z_crop_pad // 2), dtype=tf.int32)

                # Replace this random slice with a bad input slice taken
                # from the test set.
                random_slice = tf.random_crop(self.__bad_data_constant, size=tf.concat([(1,), tf.shape(stack)[1:]], axis=0))
                return tf.concat([stack[:slice_to_apply_to], random_slice, stack[slice_to_apply_to + 1:]], axis=0)


            bad_data_inputs = randomly_map_and_apply_op(missing_data_inputs, apply_random_bad_data, prob=10)

            #augmented_inputs = bad_data_inputs
            #augmented_labels = deformed_labels
            augmented_inputs = sample[:, :, :, :, :1]
            augmented_labels = sample[:, :, :, :, 1:]


            # Mess with the levels
            # leveled_image = tf.image.random_brightness(deformed_image, max_delta=0.15)
            # leveled_image = tf.image.random_contrast(leveled_image, lower=0.5, upper=1.5)

            # Affinitize the labels if applicable
            # TODO (ffjiang): Do the if applicable part
            if flood_filling_mode:
                print('Using flood-filling mode')
                shape = tf.shape(augmented_labels)

                # Choose the target segment to be the central voxel, or if that
                # is a background pixel, choose the target segment to be one of
                # the surrounding central voxels with the maximum segment id.
                target_segment = augmented_labels[0, shape[1] // 2, shape[2] // 2, shape[3] // 2, 0]
                target_segment = tf.cond(tf.equal(0.0, target_segment), 
                        lambda: tf.reduce_max(augmented_labels[0, shape[1] // 2 - 1:shape[1] // 2 + 1, shape[2] // 2 - 5:shape[2] // 2 + 5, shape[3] // 2 - 5:shape[3] // 2 + 5]), 
                        lambda: target_segment)
                targets = tf.cast(tf.equal(augmented_labels, target_segment), tf.float32)

                affs = affinitize(augmented_labels)
                middle_slice_gt = tf.tile(targets[:, shape[1] // 2:shape[1] // 2 + 1], multiples=[1,1,1,1,3])
                affs = tf.concat([affs[:, :shape[1] // 2], middle_slice_gt, affs[:, shape[1] // 2 + 1:]], axis=1)
                cropped_affs = affs[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
            else:
                targets = affinitize(augmented_labels)

            # Crop the image, to remove the padding that was added to allow safe augmentation.
            cropped_inputs = augmented_inputs[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]
            cropped_targets = targets[:, z_crop_pad // 2:-(z_crop_pad // 2), crop_pad // 2:-(crop_pad // 2), crop_pad // 2:-(crop_pad // 2), :]

            # Re-stack the image and targets
            if flood_filling_mode:
                self.training_example_op = tf.concat([tf.concat([cropped_inputs, cropped_affs, cropped_targets], axis=CHANNEL_AXIS)] * batch_size, axis=BATCH_AXIS)
            else:
                self.training_example_op = tf.concat([tf.concat([cropped_inputs, cropped_targets], axis=CHANNEL_AXIS)] * batch_size, axis=BATCH_AXIS)

    def initialize_session_variables(self, sess):
        sess.run(self.__dataset_constant.initializer, feed_dict={self.__image_ph: self.__padded_dataset})
        del self.__padded_dataset

        sess.run(self.__bad_data_constant.initializer, feed_dict={self.__bad_data_ph: self.__bad_data})
        del self.__bad_data

    def get_full_training_set(self):
        return self.__train_inputs, self.__train_labels, self.__train_targets

    def get_validation_set(self):
        return self.__validation_inputs, self.__validation_labels, self.__validation_targets

    def get_test_set(self):
        return self.__test_inputs

DATASET_DICT = {
    down.CREMI_A: CREMIDataset,
    down.CREMI_B: CREMIDataset,
    down.CREMI_C: CREMIDataset,
    down.ISBI: ISBIDataset,
    down.SNEMI3D: SNEMI3DDataset,
}

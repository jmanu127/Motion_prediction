import os
import tensorflow as tf
import torch

from utils.utils import boolean_mask

from .feature_extraction import _parse


def loading_data_from_dir(path, basename, n_take=0, batch_size=32):
    """
    Pei Sabrina Xu (May 2021)
    """
    files = []
    for file in os.listdir(path):
        if basename in file:
            files.append(os.path.join(path, file))

    if n_take == 0:
        dataset = tf.data.TFRecordDataset(files)
    else:
        dataset = tf.data.TFRecordDataset(files).take(n_take)

    dataset = dataset.map(_parse)
    dataset = dataset.batch(batch_size)

    return dataset


def preprocess_batch_data(inputs):
    # Collapse batch dimension and the agent per sample dimension.
    # Mask out agents that are never valid in the past.
    sample_is_valid = inputs['sample_is_valid']
    # TODO Ensure that states = input_state - sample_is_valid
    states = boolean_mask(inputs, 'input_states', sample_is_valid)
    gt_trajectory = boolean_mask(inputs, 'gt_future_states', sample_is_valid)
    gt_is_valid = boolean_mask(inputs, 'gt_future_is_valid', sample_is_valid)

    object_type = torch.masked_select(inputs['object_type'], sample_is_valid)
    tracks_to_predict = torch.masked_select(inputs['tracks_to_predict'], sample_is_valid)
    return sample_is_valid, states, gt_trajectory, gt_is_valid, object_type, tracks_to_predict



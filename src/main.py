import copy
import os
import yaml
import argparse
import time
import random

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from waymo_open_dataset.metrics.python import config_util_py as config_util

from utils.utils import tf2pytorch, boolean_mask
from data.data_loader import loading_data_from_dir, preprocess_batch_data
from utils.motion_metrics import MotionMetrics, preprocess_for_motion_metrics
from utils.motion_metrics import _default_metrics_config
from models.simple_model import SimpleModel
from models.seq2seq import lstm_seq2seq


parser = argparse.ArgumentParser(description='roadwayreco')
parser.add_argument('--config', default='./config.yaml')


def main():
    global args, device
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    # for data
    train_path = args.train_path
    train_basename = args.train_basename
    val_path = args.val_path
    val_basename = args.val_basename
    n_take = args.n_take
    use_map_feats = args.use_map_features
    save_best = args.save_best
    # for TensorBoard
    log_dir = args.log_dir
    # for model
    model_type = args.model_type
    # for loss
    loss_type = args.loss_type
    # for training
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train_dataset = loading_data_from_dir(train_path, train_basename, n_take=n_take, batch_size=batch_size)
    val_dataset = loading_data_from_dir(val_path, val_basename, n_take=n_take, batch_size=batch_size)

    # set up TensorBoard
    writer = SummaryWriter(log_dir)

    # set up model ##
    if model_type == "SimpleModel":     # with tensorflow
        model = SimpleModel(11, 80)
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        if model_type == "lstm":
            model = nn.LSTM(input_size=11, hidden_size=40, num_layers=5, batch_first=True, dropout=.2,bidirectional=True)
        elif model_type == "Seq2Seq":
            model = lstm_seq2seq(input_size=2, hidden_size=15)
            model.target_len = 80   # todo: parse from into config?
            model.training_prediction = 'recursive'
            model.teacher_forcing_ratio = 0.6
            model.dynamic_tf = False

        if torch.cuda.is_available():
            model = model.cuda()    # todo use send to device?

        if loss_type == "MSE":
            loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters())  # todo: take learning rate, and parse more hyper-parameters

    print(model)

    # set up motion metrics
    metrics_config = _default_metrics_config()
    train_motion_metrics = MotionMetrics(metrics_config)
    val_motion_metrics = MotionMetrics(metrics_config, device=device)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    best = float('inf')
    best_model = None
    for epoch in range(epochs):
        print('\nStart of epoch %d' % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the train_dataset.
        train_running_loss, train_metric_values = \
            iterate_by_data(train_dataset, model, use_map_feats, loss_fn, batch_size,
                            motion_metrics=train_motion_metrics, metrics_config=metrics_config,
                            mode='train', optimizer=optimizer, print_loss_very_n_step=10)

        # log the epoch loss
        writer.add_scalar('training epoch loss', train_running_loss, epoch)
        print("######### Training loss (for one epoch) at epoch {:3d}: {:10.0f}".format(epoch, float(train_running_loss)))
        # log the motion metrics
        for i, m in enumerate(
                ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, n in enumerate(metric_names):
                # print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))
                writer.add_scalar("{}/{}".format(m, n), tf2pytorch(train_metric_values[i, j]), epoch)

        # [DEBUG] display training time
        print("Training time of epoch {: >3d}: {:.4f} sec".format(epoch, time.time() - start_time))

        # Iterate over the batches of the val_dataset.
        val_running_loss, val_metric_values = \
            iterate_by_data(val_dataset, model, use_map_feats, loss_fn, batch_size,
                            motion_metrics=train_motion_metrics, metrics_config=metrics_config,
                            mode='val', print_loss_very_n_step=10)
        # log the epoch loss
        writer.add_scalar('validation epoch loss', val_running_loss, epoch)
        print("######### Validation loss (for one epoch) at epoch {:3d}: {:10.0f}".format(epoch, float(val_running_loss)))
        # log the motion metrics
        for i, m in enumerate(
                ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, n in enumerate(metric_names):
                # print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))
                writer.add_scalar("{}/{}".format(m, n), tf2pytorch(val_metric_values[i, j]), epoch)

        if val_running_loss < best:
            best = val_running_loss
            best_model = copy.deepcopy(model)

    if save_best:
        torch.save(best_model.state_dict(), 'checkpoints/' + model_type.lower() + '.pth')
        print('Best model saved.')
    print('Done!')


def iterate_by_data(dataset, model, use_map_feats, loss_fn, batch_size,
                    motion_metrics=None, metrics_config=None, mode='train', optimizer=None, print_loss_very_n_step=10):

    if mode == 'train':
        assert optimizer, 'optimizer is None in train mode'
    # Iterate over the batches of the train_dataset.
    running_loss = 0
    for step, batch in enumerate(dataset):
        # Each record in a batch represent 9s of sequential data
        if (not isinstance(model, SimpleModel)) or (not use_map_feats):  # note if use_map_feats, output of data loader will be torch.tensor ready, otherwise it will be tf.tensor
            for key in batch.keys():
                batch[key] = tf2pytorch(batch[key])
                if torch.cuda.is_available():
                    batch[key] = batch[key].cuda()
        # todo: double check if use_map_feats, all data is already send to device

        if mode == 'train':
            loss_value = train_step(model, optimizer, loss_fn, batch,
                                    motion_metrics=motion_metrics, metrics_config=metrics_config)
        elif mode == 'val':
            loss_value = val_step(model, loss_fn, batch,
                                  motion_metrics, metrics_config)

        running_loss += loss_value

        # Log every 10 batches.
        if step % print_loss_very_n_step == 0:
            print("Training loss (for one batch) at step {: >3d}: {:.0f}".format(step, float(loss_value)))
            print('Seen so far: %d samples' % ((step + 1) * batch_size))

    if motion_metrics:
        metric_values = motion_metrics.result()
    else:
        metric_values = None

    return running_loss, metric_values


def train_step(model, optimizer, loss_fn, inputs, motion_metrics=None, metrics_config=None):
    if not isinstance(model, SimpleModel):
        model.train()       # todo: is it a default and no need to add it here?
        # todo: double check if all data is already on device
        sample_is_valid, states, gt_trajectory, gt_is_valid, object_type, tracks_to_predict = preprocess_batch_data(inputs)
        # Set training target.
        prediction_start = metrics_config.track_history_samples + 1
        gt_targets = gt_trajectory[:, prediction_start:, :2]
        # weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)  # todo: how to use weights

        optimizer.zero_grad()
        if isinstance(model, nn.LSTM):
            states = states.transpose(1, 2)
            pred_trajectory, (h, c) = model(states)
            pred_trajectory = pred_trajectory.transpose(1, 2)

        if isinstance(model, lstm_seq2seq):
            # assert teacher_forcing_ratio, "teacher_forcing_ration must be parsed for lstm_seq2seq model"
            # Shape is [seq_length, batch_size, feature]
            # states = states.transpose(0,1).to(device)   # todo: check if the data is already on device
            states = states.transpose(0, 1)
            seq_len, batch_size, features = states.shape    # todo: double check if batch_size is consistent
            gt_targets = gt_targets.transpose(0, 1)
            target_len = model.target_len
            training_prediction = model.traning_prediciton
            teacher_forcing_ratio = model.teacher_forcing_ratio
            dynamic_tf = model.dynamic_tf

            # outputs tensor
            pred_trajectory = torch.zeros((target_len, batch_size, features)).to(device)
            # initialize hidden state
            encoder_hidden = model.encoder.init_hidden(batch_size)
            # zero the gradient
            optimizer.zero_grad()
            # encoder pred_trajectory
            encoder_output, encoder_hidden = model.encoder(states)
            # decoder with teacher forcing
            decoder_input = states[-1, :, :]  # shape: (batch_size, input_size)
            decoder_hidden = encoder_hidden

            if training_prediction == 'recursive':
                for t in range(target_len):
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                    pred_trajectory[t] = decoder_output
                    decoder_input = decoder_output

            if training_prediction == 'teacher_forcing':
                # use teacher forcing
                if random.random() < teacher_forcing_ratio:
                    for t in range(target_len):
                        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                        pred_trajectory[t] = decoder_output
                        decoder_input = gt_targets[t, :, :]

                # predict recursively
                else:
                    for t in range(target_len):
                        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                        pred_trajectory[t] = decoder_output
                        decoder_input = decoder_output

            if training_prediction == 'mixed_teacher_forcing':
                # predict using mixed teacher forcing
                for t in range(target_len):
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                    pred_trajectory[t] = decoder_output

                    # predict with teacher forcing
                    if random.random() < teacher_forcing_ratio:
                        decoder_input = gt_targets[t, :, :]

                    # predict recursively
                    else:
                        decoder_input = decoder_output

            # dynamic teacher forcing
            if dynamic_tf and teacher_forcing_ratio > 0:
                model.teacher_forcing_ratio = teacher_forcing_ratio - 0.02    # todo: double check with Pushkar whether it should be like this

            pred_trajectory = pred_trajectory.transpose(1, 0)
            gt_targets = gt_targets.transpose(1, 0)

        loss = loss_fn(pred_trajectory, gt_targets)     # todo: how to use weights here, like FocalLoss?
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # todo: keep it?
        optimizer.step()

        if motion_metrics:
            # for motion metrics
            # Fake the score since this model does not generate any score per predicted trajectory.
            pred_score = torch.ones(pred_trajectory.shape[0])
            mm = preprocess_for_motion_metrics(object_type, tracks_to_predict, pred_trajectory, pred_score, gt_trajectory, gt_is_valid)

    elif isinstance(model, SimpleModel):

        with tf.GradientTape() as tape:
            # Collapse batch dimension and the agent per sample dimension.
            # Mask out agents that are never valid in the past.
            sample_is_valid = inputs['sample_is_valid']
            states = tf.boolean_mask(inputs['input_states'], sample_is_valid)
            gt_trajectory = tf.boolean_mask(inputs['gt_future_states'], sample_is_valid)
            gt_is_valid = tf.boolean_mask(inputs['gt_future_is_valid'], sample_is_valid)
            # Set training target.
            prediction_start = metrics_config.track_history_samples + 1
            gt_targets = gt_trajectory[:, prediction_start:, :2]
            weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)
            pred_trajectory = model(states, training=True)
            loss = loss_fn(gt_targets, pred_trajectory, sample_weight=weights)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if motion_metrics:
            object_type = tf.boolean_mask(inputs['object_type'], sample_is_valid)
            # Fake the score since this model does not generate any score per predicted
            # trajectory.
            pred_score = tf.ones(shape=tf.shape(pred_trajectory)[:-2])

            # Only keep `tracks_to_predict` for evaluation.
            tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'],
                                                sample_is_valid)
            mm = dict()
            mm['pred_trajectory'] = tf.boolean_mask(pred_trajectory, tracks_to_predict)
            mm['pred_score'] = tf.boolean_mask(pred_score, tracks_to_predict)
            mm['gt_trajectory'] = tf.boolean_mask(gt_trajectory, tracks_to_predict)
            mm['gt_is_valid'] = tf.boolean_mask(gt_is_valid, tracks_to_predict)
            mm['object_type'] = tf.boolean_mask(object_type, tracks_to_predict)

    motion_metrics.update_state(mm['pred_trajectory'],
                                mm['pred_score'],
                                mm['gt_trajectory'],
                                mm['gt_is_valid'],
                                mm['object_type'])

    return loss


def val_step(model, loss_fn, inputs, motion_metrics=None, metrics_config=None):

    if not isinstance(model, SimpleModel):
        # model.train()

        sample_is_valid, states, gt_trajectory, gt_is_valid, object_type, tracks_to_predict \
            = preprocess_batch_data(inputs)
        # Set training target.
        prediction_start = metrics_config.track_history_samples + 1
        gt_targets = gt_trajectory[:, prediction_start:, :2]
        # weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)

        if isinstance(model, nn.LSTM):
            states = states.transpose(1, 2)

        with torch.no_grad():
            pred_trajectory, (h, c) = model(states)
            loss = loss_fn(pred_trajectory.transpose(1, 2), gt_targets)

        if isinstance(model, nn.LSTM):
            pred_trajectory = pred_trajectory.transpose(1, 2)

        if motion_metrics:
            # for motion metrics
            # Fake the score since this model does not generate any score per predicted trajectory.
            pred_score = torch.ones(pred_trajectory.shape[0])
            mm = preprocess_for_motion_metrics(
                object_type, tracks_to_predict, pred_trajectory, pred_score, gt_trajectory, gt_is_valid)
            motion_metrics.update_state(mm['pred_trajectory'],
                                        mm['pred_score'],
                                        mm['gt_trajectory'],
                                        mm['gt_is_valid'],
                                        mm['object_type'])

    elif isinstance(model, SimpleModel):
        raise NotImplementedError

    return loss


if __name__ == '__main__':
    main()

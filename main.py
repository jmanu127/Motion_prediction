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
from data.feature_extraction import _parse
from utils.motion_metrics import MotionMetrics
from utils.motion_metrics import _default_metrics_config
from models.simple_model import SimpleModel
from models.seq2seq import lstm_seq2seq


parser = argparse.ArgumentParser(description='roadwayreco')
parser.add_argument('--config', default='./config.yaml')


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    # for data
    path = args.path
    basename = args.basename
    n_take = args.n_take
    # for TensorBoard
    log_dir = args.log_dir
    # for model
    model_type = args.model_type
    # for training
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # load data
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

    # set up TensorBoard
    writer = SummaryWriter(log_dir)

    # set up model ##
    if model_type == "LSTM":
        model = nn.LSTM(input_size=11, hidden_size=80, batch_first=True)
        if torch.cuda.is_available():
            model = model.cuda()

        if args.loss_type == "MSE":
            loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters())  # todo: take learning rate, and parse more hyper-parameters

    if model_type == "SimpleModel":     # with tensorflow
        model = SimpleModel(11, 80)
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if model_type == "Seq2Seq":
        model = lstm_seq2seq(input_size = 2, hidden_size = 15)
        if torch.cuda.is_available():
            model = model.cuda()
        if args.loss_type == "MSE":
            loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

    # set up motion metrics
    metrics_config = _default_metrics_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    motion_metrics = MotionMetrics(metrics_config, device)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    for epoch in range(epochs):
        print('\nStart of epoch %d' % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        running_loss = 0
        for step, batch in enumerate(dataset):
            # Each record in a batch represent 9s of sequential data
            if not isinstance(model, SimpleModel):
                for key in batch.keys():
                    batch[key] = tf2pytorch(batch[key])            
            # train loop
            loss_value = train_step(model, optimizer, loss_fn, batch, metrics_config, motion_metrics, device)

            running_loss += loss_value

            # Log every 10 batches.
            if step % 10 == 0:
                print("Training loss (for one batch) at step {: >3d}: {:.0f}".format(step, float(loss_value)))
                print('Seen so far: %d samples' % ((step + 1) * batch_size))

            #  log the batch loss
            # writer.add_scalar('training batch loss', loss_value, epoch * len(batch) + step)
        # log the epoch loss
        writer.add_scalar('training epoch loss', running_loss/100, epoch)
        print("######### Training loss (for one epoch) at epoch {:3d}: {:10.0f}".format(epoch, float(running_loss)))

        train_metric_values = motion_metrics.result()
        # log the motion metrics
        for i, m in enumerate(
                ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, n in enumerate(metric_names):
                # print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))
                writer.add_scalar("{}/{}".format(m, n), tf2pytorch(train_metric_values[i, j]), epoch)
        # [DEBUG] display training time
        print("Training time of epoch {: >3d}: {:.4f} sec".format(epoch, time.time() - start_time))


def train_step(model, optimizer, loss_fn, inputs, metrics_config, motion_metrics, device):

    if not (isinstance(model, SimpleModel) or isinstance(model, lstm_seq2seq)) :
        model.train()
        # Record total loss

        # Collapse batch dimension and the agent per sample dimension.
        # Mask out agents that are never valid in the past.
        sample_is_valid = inputs['sample_is_valid']
        # TODO Ensure that states = input_state - sample_is_valid
        states = boolean_mask(inputs, 'input_states', sample_is_valid)
        gt_trajectory = boolean_mask(inputs, 'gt_future_states', sample_is_valid)
        gt_is_valid = boolean_mask(inputs, 'gt_future_is_valid', sample_is_valid)
        
        # Set training target.
        prediction_start = metrics_config.track_history_samples + 1
        gt_targets = gt_trajectory[:, prediction_start:, :2]
        
        if torch.cuda.is_available():
            states = states.cuda()
            gt_targets = gt_targets.cuda()
            
        
        weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)
        pred_trajectory, (h, c) = model(states.transpose(1, 2).to(device))

        optimizer.zero_grad()
        loss = loss_fn(pred_trajectory.transpose(1, 2), gt_targets.to(device))     # todo: how to use weights?
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        object_type = torch.masked_select(inputs['object_type'], sample_is_valid)

        # Fake the score since this model does not generate any score per predicted
        # trajectory.
        pred_score = torch.ones(pred_trajectory.shape[0])

        # Only keep `tracks_to_predict` for evaluation.a
        tracks_to_predict = torch.masked_select(inputs['tracks_to_predict'],
                                                sample_is_valid)
        mm_pred_trajectory = pred_trajectory.clone().transpose(1, 2)
        assert len(tracks_to_predict.shape) == 1, "mask is not 1D, cannot mask in this way"
        mm_pred_trajectory = mm_pred_trajectory[tracks_to_predict, :, :]
        mm_pred_score = pred_score[tracks_to_predict]
        mm_gt_trajectory = gt_trajectory[tracks_to_predict, :, :]
        mm_gt_is_valid = gt_is_valid[tracks_to_predict, :]
        mm_object_type = object_type[tracks_to_predict]

        # print("mm_pred_trajectory.shape = {}".format(mm_pred_trajectory.shape))
        # print("mm_pred_score.shape = {}".format(mm_pred_score.shape))
        # print("mm_gt_trajectory.shape = {}".format(mm_gt_trajectory.shape))
        # print("mm_gt_is_valid.shape = {}".format(mm_gt_is_valid.shape))
        # print("mm_object_type.shape = {}".format(mm_object_type.shape))

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

        object_type = tf.boolean_mask(inputs['object_type'], sample_is_valid)
        # Fake the score since this model does not generate any score per predicted
        # trajectory.
        pred_score = tf.ones(shape=tf.shape(pred_trajectory)[:-2])

        # Only keep `tracks_to_predict` for evaluation.
        tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'],
                                            sample_is_valid)

        mm_pred_trajectory = tf.boolean_mask(pred_trajectory, tracks_to_predict)
        mm_pred_score = tf.boolean_mask(pred_score, tracks_to_predict)
        mm_gt_trajectory = tf.boolean_mask(gt_trajectory, tracks_to_predict)
        mm_gt_is_valid = tf.boolean_mask(gt_is_valid, tracks_to_predict)
        mm_object_type = tf.boolean_mask(object_type, tracks_to_predict)
    
    elif isinstance(model, lstm_seq2seq):
        target_len = 80
        training_prediction = 'recursive'
        teacher_forcing_ratio = 0.6
        dynamic_tf = False
        model.train()
        # Record total loss

        # Collapse batch dimension and the agent per sample dimension.
        # Mask out agents that are never valid in the past.
        sample_is_valid = inputs['sample_is_valid']
        # TODO Ensure that states = input_state - sample_is_valid
        states = boolean_mask(inputs, 'input_states', sample_is_valid)
        gt_trajectory = boolean_mask(inputs, 'gt_future_states', sample_is_valid)
        gt_is_valid = boolean_mask(inputs, 'gt_future_is_valid', sample_is_valid)
        # Set training target.
        prediction_start = metrics_config.track_history_samples + 1
        gt_targets = gt_trajectory[:, prediction_start:, :2]
        weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)
        # Shape is [seq_length, batch_size, feature]
        states = states.transpose(0,1).to(device)
        seq_len,batch_size,features  = states.shape
        gt_targets = gt_targets.transpose(0,1).to(device)
        # outputs tensor
        pred_trajectory = torch.zeros((target_len, batch_size,features)).to(device)

        # initialize hidden state
        encoder_hidden = model.encoder.init_hidden(batch_size)

        # zero the gradient
        optimizer.zero_grad()

        # encoder pred_trajectory
        encoder_output, encoder_hidden = model.encoder(states)

        # decoder with teacher forcing
        decoder_input = states[-1, :, :]   # shape: (batch_size, input_size)
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
            teacher_forcing_ratio = teacher_forcing_ratio - 0.02  

        loss = loss_fn(pred_trajectory, gt_targets)
        pred_trajectory=pred_trajectory.transpose(1,0)
        # print("mm_pred_trajectory.shape = {}".format(mm_pred_trajectory.shape))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        object_type = torch.masked_select(inputs['object_type'], sample_is_valid)
        # Fake the score since this model does not generate any score per predicted
        # trajectory.
        pred_score = torch.ones(pred_trajectory.shape[0])

        # Only keep `tracks_to_predict` for evaluation.a
        tracks_to_predict = torch.masked_select(inputs['tracks_to_predict'],
                                                sample_is_valid)
        
        mm_pred_trajectory = pred_trajectory.clone()
        assert len(tracks_to_predict.shape) == 1, "mask is not 1D, cannot mask in this way"
        mm_pred_trajectory = mm_pred_trajectory[tracks_to_predict, :, :]
        mm_pred_score = pred_score[tracks_to_predict]
        mm_gt_trajectory = gt_trajectory[tracks_to_predict, :, :]
        mm_gt_is_valid = gt_is_valid[tracks_to_predict, :]
        mm_object_type = object_type[tracks_to_predict]

        # print("mm_pred_trajectory.shape = {}".format(mm_pred_trajectory.shape))
        # print("mm_pred_score.shape = {}".format(mm_pred_score.shape))
        # print("mm_gt_trajectory.shape = {}".format(mm_gt_trajectory.shape))
        # print("mm_gt_is_valid.shape = {}".format(mm_gt_is_valid.shape))
        # print("mm_object_type.shape = {}".format(mm_object_type.shape))

    motion_metrics.update_state(
        mm_pred_trajectory,
        mm_pred_score,
        mm_gt_trajectory,
        mm_gt_is_valid,
        mm_object_type)

    return loss


if __name__ == '__main__':
    main()

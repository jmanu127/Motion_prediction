import torch
import tensorflow as tf
from utils.utils import convert_torch_tensor_to_numpy

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2


class MotionMetrics(tf.keras.metrics.Metric):
    """Wrapper for motion metrics computation."""

    def __init__(self, config, device='cpu'):
        super().__init__()
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._metrics_config = config
        self._device = device

    def reset_state(self):
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []

    def update_state(self, prediction_trajectory, prediction_score,
                     ground_truth_trajectory, ground_truth_is_valid, object_type):
        self._prediction_trajectory.append(prediction_trajectory)
        self._prediction_score.append(prediction_score)
        self._ground_truth_trajectory.append(ground_truth_trajectory)
        self._ground_truth_is_valid.append(ground_truth_is_valid)
        self._object_type.append(object_type)

    def result(self):
        # [batch_size, steps, 2].
        prediction_trajectory = torch.cat(tuple(self._prediction_trajectory), 0)
        # [batch_size].
        prediction_score = torch.cat(tuple(self._prediction_score), 0)
        # [batch_size, gt_steps, 7].
        ground_truth_trajectory = torch.cat(tuple(self._ground_truth_trajectory), 0)
        # [batch_size, gt_steps].
        ground_truth_is_valid = torch.cat(tuple(self._ground_truth_is_valid), 0)
        # [batch_size].
        object_type = torch.cat(tuple(self._object_type), 0)
        object_type = object_type.type(torch.LongTensor)

        # We are predicting more steps than needed by the eval code. Subsample.
        interval = (
                self._metrics_config.track_steps_per_second //
                self._metrics_config.prediction_steps_per_second)
        prediction_trajectory = prediction_trajectory[:, (interval - 1)::interval]

        # Prepare these into shapes expected by the metrics computation.
        #
        # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        # top_k is 1 because we have a uni-modal model.
        # num_agents_per_joint_prediction is also 1 here.
        prediction_trajectory = prediction_trajectory.unsqueeze(1).unsqueeze(1)
        # [batch_size, top_k].
        prediction_score = prediction_score.unsqueeze(1)
        # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].
        ground_truth_trajectory = ground_truth_trajectory.unsqueeze(1)
        # [batch_size, num_agents_per_joint_prediction, gt_steps].
        ground_truth_is_valid = ground_truth_is_valid.unsqueeze(1)
        # [batch_size, num_agents_per_joint_prediction].
        object_type = object_type.unsqueeze(1)

        prediction_trajectory = tf.convert_to_tensor(
            convert_torch_tensor_to_numpy(prediction_trajectory, device=self._device))
        prediction_score = tf.convert_to_tensor(convert_torch_tensor_to_numpy(prediction_score, device=self._device))
        ground_truth_trajectory = tf.convert_to_tensor(
            convert_torch_tensor_to_numpy(ground_truth_trajectory, device=self._device))
        ground_truth_is_valid = tf.convert_to_tensor(
            convert_torch_tensor_to_numpy(ground_truth_is_valid, device=self._device))
        object_type = tf.convert_to_tensor(convert_torch_tensor_to_numpy(object_type, device=self._device))

        # [DEBUG]
        # print("prediction_trajectory.is_tfTneor = {}, shape = {}, dtype = {}".format(
        #     isinstance(prediction_trajectory, tf.Tensor), prediction_trajectory.shape, prediction_trajectory.dtype))
        # print("prediction_score.is_tfTensor= {}, shape = {}, dtype = {}".format(
        #     isinstance(prediction_score, tf.Tensor), prediction_score.shape, prediction_score.dtype))
        # print("ground_truth_trajectory.is_tfTensor = {}, shape = {}, dtype = {}".format(
        #     isinstance(ground_truth_trajectory, tf.Tensor), ground_truth_trajectory.shape, ground_truth_trajectory.dtype))
        # print("ground_truth_is_valid.is_tfTensor = {}, shape = {}, dtype = {}".format(
        #     isinstance(ground_truth_is_valid, tf.Tensor), ground_truth_is_valid.shape, ground_truth_is_valid.dtype))
        # print("object_type.is_tfTensor = {}, shape = {}, dtype = {}".format(
        #     isinstance(object_type, tf.Tensor), object_type.shape, object_type.dtype))

        return py_metrics_ops.motion_metrics(
            config=self._metrics_config.SerializeToString(),
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            object_type=object_type)


def _default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
      track_steps_per_second: 10
      prediction_steps_per_second: 2
      track_history_samples: 10
      track_future_samples: 80
      speed_lower_bound: 1.4
      speed_upper_bound: 11.0
      speed_scale_lower: 0.5
      speed_scale_upper: 1.0
      step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
      }
      step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
      }
      step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
      }
      max_predictions: 6
    """
    text_format.Parse(config_text, config)
    return config


def preprocess_for_motion_metrics(object_type, tracks_to_predict, pred_trajectory, pred_score, gt_trajectory, gt_is_valid):
    """
    Pei Sabrina Xu (May 2021)
    """
    # Only keep `tracks_to_predict` for evaluation.a

    assert len(tracks_to_predict.shape) == 1, "mask is not 1D, cannot mask in this way"
    mm = dict()
    mm['pred_trajectory'] = pred_trajectory[tracks_to_predict, :, :]
    mm['pred_score'] = pred_score[tracks_to_predict]
    mm['gt_trajectory'] = gt_trajectory[tracks_to_predict, :, :]
    mm['gt_is_valid'] = gt_is_valid[tracks_to_predict, :]
    mm['object_type'] = object_type[tracks_to_predict]

    # print("mm_pred_trajectory.shape = {}".format(mm_pred_trajectory.shape))
    # print("mm_pred_score.shape = {}".format(mm_pred_score.shape))
    # print("mm_gt_trajectory.shape = {}".format(mm_gt_trajectory.shape))
    # print("mm_gt_is_valid.shape = {}".format(mm_gt_is_valid.shape))
    # print("mm_object_type.shape = {}".format(mm_object_type.shape))

    return mm

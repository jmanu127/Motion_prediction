import torch
from tfrecord.torch.dataset import TFRecordDataset

# todo: clone the tfrecord source code (https://github.com/vahidk/tfrecord) as a dependency of your roadwayreco project

# todo: download a .tfrecord file first, then create the index file in the way that README described.

# todo: update your file path
tfrecord_path = "/data/uncompressed_tf_example_training_training_tfexample.tfrecord-00000-of-01000"
index_path = "/data/uncompressed_tf_example_training_training_tfexample.tfrecord-00000-of-01000_index"


"""
version # 1 simple test: 
"""
# description = {'state/current/velocity_x': None, 'traffic_light_state/future/state': None}

"""
version #2  list all keys for sanity check:
"""
description = {}
all_keys = ['traffic_light_state/future/x',
            'state/tracks_to_predict',
            'state/past/length',
            'state/difficulty_level',
            'state/past/velocity_x',
            'state/objects_of_interest',
            'traffic_light_state/future/z',
            'state/past/width',
            'state/past/z',
            'state/past/height',
            'traffic_light_state/past/x',
            'state/past/y',
            'traffic_light_state/current/timestamp_micros',
            'state/type',
            'state/current/velocity_y',
            'state/current/valid',
            'state/current/length',
            'state/past/velocity_y',
            'state/past/speed',
            'state/past/timestamp_micros',
            'state/past/x',
            'state/current/height',
            'state/future/length',
            'traffic_light_state/current/state',
            'traffic_light_state/future/state',
            'traffic_light_state/past/valid',
            'traffic_light_state/future/y',
            'traffic_light_state/past/y',
            'state/past/bbox_yaw',
            'roadgraph_samples/xyz',
            'state/future/velocity_y',
            'state/past/valid',
            'traffic_light_state/current/valid',
            'state/future/timestamp_micros',
            'state/id',
            'state/past/vel_yaw',
            'roadgraph_samples/valid',
            'traffic_light_state/current/x',
            'traffic_light_state/current/y',
            'state/future/velocity_x',
            'state/current/velocity_x',
            'state/current/x',
            'state/future/height',
            'state/current/bbox_yaw',
            'traffic_light_state/future/valid',
            'traffic_light_state/past/state',
            'traffic_light_state/current/id',
            'traffic_light_state/past/timestamp_micros',
            'roadgraph_samples/id',
            'state/future/x',
            'roadgraph_samples/dir',
            'state/current/y',
            'state/future/bbox_yaw',
            'state/current/speed',
            'state/future/vel_yaw',
            'traffic_light_state/past/id',
            'state/current/width',
            'state/current/timestamp_micros',
            'traffic_light_state/past/z',
            'state/is_sdc',
            'traffic_light_state/future/timestamp_micros',
            'traffic_light_state/future/id',
            'state/future/z',
            'state/current/z',
            'roadgraph_samples/type',
            'state/future/y',
            'state/future/speed',
            'state/future/valid',
            'state/current/vel_yaw',
            'traffic_light_state/current/z',
            'state/future/width']

# all_keys = ['scenario/id']

for key in all_keys:
    description[key] = None

# read data
dataset = TFRecordDataset(tfrecord_path, index_path, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)
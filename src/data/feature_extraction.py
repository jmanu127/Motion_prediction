import tensorflow as tf
import torch
import numpy as np

from utils.utils import tf2pytorch

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

# def get_features_description(features):
#     features_description = {}
#     for feature in features:
#         features_description.update(feature)
#     return features_description

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)


def map_feature_extractor(decoded_dict: dict, T=2):
    """
    Utility function for extracting features from HD map data.

    Input:
        decoded_dict (dict) - single scenario meta-data per https://waymo.com/open/data/motion/tfexample
        T (int) - Number of meters to extend outward from left and right when determining left/right edge types

    Output:
        decoded_dict (dict) - single scenario meta-data with the following new keys:

                            1) map_features/past/road, map_features/current/road, map_features/future/road
                                
                                * Enum for which roadway each actor is driving on (freeway, surface street, or bike lane)
                            
                            2) map_features/past/left_edge, map_features/current/left_edge, map_features/future/left_edge

                                * Enum for which type of road edge exists on the left side of each actor (solid single yellow, solid double white, etc.)

                            3) map_features/past/right_edge, map_features/current/right_edge, map_features/future/right_edge

                                * Enum for which type of road edge exists on the right side of each actor (solid single yellow, solid double white, etc.)
                            
                            4) map_features/past/nearest_tcd, map_features/current/nearest_tcd, map_features/future/nearest_tcd

                                * Enum for which traffic control device (stop sign, traffic light, cross walk, etc.) is closest to each actor

                            5) map_features/past/log_tcd_dist, map_features/current/log_tcd_dist, map_features/future/log_tcd_dist

                                * Log distance to nearest traffic control device for each actor
                            
    """
    ## Get current HW device ##
    gpu_available = torch.cuda.is_available()
    device = 'cpu'
    if(gpu_available):
        device = torch.cuda.current_device()

    ## Instantiate dictionary to store the points of all 20 map types ##
    type_dict = {x: [] for x in range(1, 21)}

    ## Extract features for all valid points (i.e. types > 0) ##

    # Filter out non-valid roadway points
    mask = decoded_dict['roadgraph_samples/type'] >= 1
    valid_types_indicies = torch.nonzero(mask.squeeze())

    # Partition each valid point by type
    for i in range(1, 20):
        type_dict[i] = decoded_dict['roadgraph_samples/xyz'][decoded_dict['roadgraph_samples/type'].squeeze() == i].to(device)

    ## Construct actor position tensors of size [num_actors, num_steps, 3] to make linear manipulations easier with roadgraph_samples/xyz ##
    state_past_pos = torch.cat([decoded_dict['state/past/x'][:, :, None], decoded_dict['state/past/y'][:, :, None], decoded_dict['state/past/z'][:, :, None]], -1).to(device)
    state_current_pos = torch.cat([decoded_dict['state/current/x'][:, :, None], decoded_dict['state/current/y'][:, :, None], decoded_dict['state/current/z'][:, :, None]], -1).to(device)
    state_future_pos = torch.cat([decoded_dict['state/future/x'][:, :, None], decoded_dict['state/future/y'][:, :, None], decoded_dict['state/future/z'][:, :, None]], -1).to(device)

    ## Construct left actor positions w.r.t yaw ##
    state_past_left_pos = torch.cat([(state_past_pos[:,:,0] + T*torch.cos((np.pi / 2) + decoded_dict['state/past/vel_yaw']).to(device))[:,:,None], 
                                    (state_past_pos[:,:,1] + T*torch.sin((np.pi / 2) + decoded_dict['state/past/vel_yaw']).to(device))[:,:,None], 
                                    state_past_pos[:,:,2][:,:,None]], -1).to(device)
    
    state_current_left_pos = torch.cat([(state_current_pos[:,:,0] + T*torch.cos((np.pi / 2) + decoded_dict['state/current/vel_yaw']).to(device))[:,:,None], 
                                        (state_current_pos[:,:,1] + T*torch.sin((np.pi / 2) + decoded_dict['state/current/vel_yaw']).to(device))[:,:,None], 
                                        state_current_pos[:,:,2][:,:,None]], -1).to(device)

    state_future_left_pos = torch.cat([(state_future_pos[:,:,0] + T*torch.cos((np.pi / 2) + decoded_dict['state/future/vel_yaw']).to(device))[:,:,None], 
                                        (state_future_pos[:,:,1] + T*torch.sin((np.pi / 2) + decoded_dict['state/future/vel_yaw']).to(device))[:,:,None], 
                                        state_future_pos[:,:,2][:,:,None]], -1).to(device)

    ## Construct right actor positions w.r.t yaw ##
    state_past_right_pos = torch.cat([(state_past_pos[:,:,0] + T*torch.cos(-(np.pi / 2) + decoded_dict['state/past/vel_yaw']).to(device))[:,:,None], 
                                    (state_past_pos[:,:,1] + T*torch.sin(-(np.pi / 2) + decoded_dict['state/past/vel_yaw']).to(device))[:,:,None], 
                                    state_past_pos[:,:,2][:,:,None]], -1).to(device)

    state_current_right_pos = torch.cat([(state_current_pos[:,:,0] + T*torch.cos(-(np.pi / 2) + decoded_dict['state/current/vel_yaw']).to(device))[:,:,None], 
                                        (state_current_pos[:,:,1] + T*torch.sin(-(np.pi / 2) + decoded_dict['state/current/vel_yaw']).to(device))[:,:,None], 
                                        state_current_pos[:,:,2][:,:,None]], -1).to(device)

    state_future_right_pos = torch.cat([(state_future_pos[:,:,0] + T*torch.cos(-(np.pi / 2) + decoded_dict['state/future/vel_yaw']).to(device))[:,:,None], 
                                        (state_future_pos[:,:,1] + T*torch.sin(-(np.pi / 2) + decoded_dict['state/future/vel_yaw']).to(device))[:,:,None], 
                                        state_future_pos[:,:,2][:,:,None]], -1).to(device)

    # Past Road Type
    past_dist_freeway = torch.cdist(type_dict[1].repeat(128,1,1), state_past_pos).min(dim=1).values[None, :, :] if type_dict[1].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_surfacestreet = torch.cdist(type_dict[2].repeat(128,1,1), state_past_pos).min(dim=1).values[None, :, :] if type_dict[2].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_bikelane = torch.cdist(type_dict[3].repeat(128,1,1), state_past_pos).min(dim=1).values[None, :, :] if type_dict[3].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    decoded_dict['map_features/past/road'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([1, 128, 10]).to(device),
                                                                  past_dist_freeway,
                                                                  past_dist_surfacestreet, 
                                                                  past_dist_bikelane], 0).argmin(dim=0).cpu().numpy())
    
    # Past Left Edge
    past_dist_brokensinglewhite = torch.cdist(type_dict[6].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[6].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_solidsinglewhite = torch.cdist(type_dict[7].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[7].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_soliddoublewhite = torch.cdist(type_dict[8].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[8].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_brokensingleyellow = torch.cdist(type_dict[9].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[9].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_brokendoubleyellow = torch.cdist(type_dict[10].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[10].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_solidsingleyellow = torch.cdist(type_dict[11].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[11].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_soliddoubleyellow = torch.cdist(type_dict[12].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[12].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_passingdoubleyellow = torch.cdist(type_dict[13].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[13].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_roadedgeboundary = torch.cdist(type_dict[15].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[15].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_roadedgemedian = torch.cdist(type_dict[16].repeat(128,1,1), state_past_left_pos).min(dim=1).values[None, :, :] if type_dict[16].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)

    decoded_dict['map_features/past/left_edge'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([6, 128, 10]).to(device),
                                                                                past_dist_brokensinglewhite,
                                                                                past_dist_solidsinglewhite, 
                                                                                past_dist_soliddoublewhite,
                                                                                past_dist_brokensingleyellow,
                                                                                past_dist_brokendoubleyellow,
                                                                                past_dist_solidsingleyellow,
                                                                                past_dist_soliddoubleyellow,
                                                                                past_dist_passingdoubleyellow,
                                                                                np.inf * torch.ones([1, 128, 10]).to(device),
                                                                                past_dist_roadedgeboundary,
                                                                                past_dist_roadedgemedian], 0).argmin(dim=0).cpu().numpy())
    
    # Past Right Edge
    past_dist_brokensinglewhite = torch.cdist(type_dict[6].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[6].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_solidsinglewhite = torch.cdist(type_dict[7].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[7].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_soliddoublewhite = torch.cdist(type_dict[8].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[8].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_brokensingleyellow = torch.cdist(type_dict[9].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[9].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_brokendoubleyellow = torch.cdist(type_dict[10].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[10].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_solidsingleyellow = torch.cdist(type_dict[11].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[11].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_soliddoubleyellow = torch.cdist(type_dict[12].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[12].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_passingdoubleyellow = torch.cdist(type_dict[13].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[13].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_roadedgeboundary = torch.cdist(type_dict[15].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[15].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_roadedgemedian = torch.cdist(type_dict[16].repeat(128,1,1), state_past_right_pos).min(dim=1).values[None, :, :] if type_dict[16].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    
    decoded_dict['map_features/past/right_edge'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([6, 128, 10]).to(device),
                                                                        past_dist_brokensinglewhite,
                                                                        past_dist_solidsinglewhite, 
                                                                        past_dist_soliddoublewhite,
                                                                        past_dist_brokensingleyellow,
                                                                        past_dist_brokendoubleyellow,
                                                                        past_dist_solidsingleyellow,
                                                                        past_dist_soliddoubleyellow,
                                                                        past_dist_passingdoubleyellow,
                                                                        np.inf * torch.ones([1, 128, 10]).to(device),
                                                                        past_dist_roadedgeboundary,
                                                                        past_dist_roadedgemedian], 0).argmin(dim=0).cpu().numpy())
    
    # Past Nearest TCD
    past_dist_stopsign = torch.cdist(type_dict[17].repeat(128,1,1), state_past_pos).min(dim=1).values[None, :, :] if type_dict[17].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_crosswalk = torch.cdist(type_dict[18].repeat(128,1,1), state_past_pos).min(dim=1).values[None, :, :] if type_dict[18].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_dist_speedbump = torch.cdist(type_dict[19].repeat(128,1,1), state_past_pos).min(dim=1).values[None, :, :] if type_dict[19].shape[0] > 0 else np.inf * torch.ones([1, 128, 10]).to(device)
    past_tcd_tensor = torch.cat([np.inf * torch.ones([16, 128, 10]),
                            past_dist_stopsign,
                            past_dist_crosswalk, 
                            past_dist_speedbump], 0)

    decoded_dict['map_features/past/nearest_tcd'] = tf.convert_to_tensor(past_tcd_tensor.argmin(dim=0).cpu().numpy())

    
    ## Current features ##

    # Current Road Type
    current_dist_freeway = torch.cdist(type_dict[1].repeat(128,1,1), state_current_pos).min(dim=1).values[None, :, :] if type_dict[1].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_surfacestreet = torch.cdist(type_dict[2].repeat(128,1,1), state_current_pos).min(dim=1).values[None, :, :] if type_dict[2].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_bikelane = torch.cdist(type_dict[3].repeat(128,1,1), state_current_pos).min(dim=1).values[None, :, :] if type_dict[3].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    
    decoded_dict['map_features/current/road'] = tf.convert_to_tensor(torch.cat([current_dist_freeway,
                                                                    current_dist_surfacestreet, 
                                                                    current_dist_bikelane], 0).argmin(dim=0).cpu().numpy())

    # Current left edge
    current_dist_brokensinglewhite = torch.cdist(type_dict[6].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[6].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_solidsinglewhite = torch.cdist(type_dict[7].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[7].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_soliddoublewhite = torch.cdist(type_dict[8].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[8].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_brokensingleyellow = torch.cdist(type_dict[9].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[9].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_brokendoubleyellow = torch.cdist(type_dict[10].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[10].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_solidsingleyellow = torch.cdist(type_dict[11].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[11].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_soliddoubleyellow = torch.cdist(type_dict[12].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[12].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_passingdoubleyellow = torch.cdist(type_dict[13].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[13].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_roadedgeboundary = torch.cdist(type_dict[15].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[15].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_roadedgemedian = torch.cdist(type_dict[16].repeat(128,1,1), state_current_left_pos).min(dim=1).values[None, :, :] if type_dict[16].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    decoded_dict['map_features/current/left_edge'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([6, 128, 1]).to(device),
                                                                                    current_dist_brokensinglewhite,
                                                                                    current_dist_solidsinglewhite, 
                                                                                    current_dist_soliddoublewhite,
                                                                                    current_dist_brokensingleyellow,
                                                                                    current_dist_brokendoubleyellow,
                                                                                    current_dist_solidsingleyellow,
                                                                                    current_dist_soliddoubleyellow,
                                                                                    current_dist_passingdoubleyellow,
                                                                                    np.inf * torch.ones([1, 128, 1]).to(device),
                                                                                    current_dist_roadedgeboundary,
                                                                                    current_dist_roadedgemedian], 0).argmin(dim=0).cpu().numpy())
    
    # Current right edge
    current_dist_brokensinglewhite = torch.cdist(type_dict[6].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[6].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_solidsinglewhite = torch.cdist(type_dict[7].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[7].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_soliddoublewhite = torch.cdist(type_dict[8].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[8].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_brokensingleyellow = torch.cdist(type_dict[9].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[9].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_brokendoubleyellow = torch.cdist(type_dict[10].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[10].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_solidsingleyellow = torch.cdist(type_dict[11].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[11].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_soliddoubleyellow = torch.cdist(type_dict[12].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[12].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_passingdoubleyellow = torch.cdist(type_dict[13].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[13].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_roadedgeboundary = torch.cdist(type_dict[15].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[15].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_roadedgemedian = torch.cdist(type_dict[16].repeat(128,1,1), state_current_right_pos).min(dim=1).values[None, :, :] if type_dict[16].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    decoded_dict['map_features/current/right_edge'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([6, 128, 1]).to(device),
                                                                                    current_dist_brokensinglewhite,
                                                                                    current_dist_solidsinglewhite, 
                                                                                    current_dist_soliddoublewhite,
                                                                                    current_dist_brokensingleyellow,
                                                                                    current_dist_brokendoubleyellow,
                                                                                    current_dist_solidsingleyellow,
                                                                                    current_dist_soliddoubleyellow,
                                                                                    current_dist_passingdoubleyellow,
                                                                                    np.inf * torch.ones([1, 128, 1]).to(device),
                                                                                    current_dist_roadedgeboundary,
                                                                                    current_dist_roadedgemedian], 0).argmin(dim=0).cpu().numpy())
    
    # Current Nearest TCD
    current_dist_stopsign = torch.cdist(type_dict[17].repeat(128,1,1), state_current_pos).min(dim=1).values[None, :, :] if type_dict[17].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_crosswalk = torch.cdist(type_dict[18].repeat(128,1,1), state_current_pos).min(dim=1).values[None, :, :] if type_dict[18].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_dist_speedbump = torch.cdist(type_dict[19].repeat(128,1,1), state_current_pos).min(dim=1).values[None, :, :] if type_dict[19].shape[0] > 0 else np.inf * torch.ones([1, 128, 1]).to(device)
    current_tcd_tensor = torch.cat([np.inf * torch.ones([16, 128, 1]).to(device),
                                    current_dist_stopsign,
                                    current_dist_crosswalk, 
                                    current_dist_speedbump], 0)

    decoded_dict['map_features/current/nearest_tcd'] = tf.convert_to_tensor(current_tcd_tensor.argmin(dim=0).cpu().numpy())

        
    ## Future features ##
    
    # Future Road Type
    future_dist_freeway = torch.cdist(type_dict[1].repeat(128,1,1), state_future_pos).min(dim=1).values[None, :, :] if type_dict[1].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_surfacestreet = torch.cdist(type_dict[2].repeat(128,1,1), state_future_pos).min(dim=1).values[None, :, :] if type_dict[2].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_bikelane = torch.cdist(type_dict[3].repeat(128,1,1), state_future_pos).min(dim=1).values[None, :, :] if type_dict[3].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    
    decoded_dict['map_features/future/road'] = tf.convert_to_tensor(torch.cat([future_dist_freeway,
                                                                            future_dist_surfacestreet, 
                                                                            future_dist_bikelane], 0).argmin(dim=0).cpu().numpy())
    
    # Future Left edge
    future_dist_brokensinglewhite = torch.cdist(type_dict[6].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[6].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_solidsinglewhite = torch.cdist(type_dict[7].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[7].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_soliddoublewhite = torch.cdist(type_dict[8].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[8].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_brokensingleyellow = torch.cdist(type_dict[9].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[9].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_brokendoubleyellow = torch.cdist(type_dict[10].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[10].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_solidsingleyellow = torch.cdist(type_dict[11].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[11].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_soliddoubleyellow = torch.cdist(type_dict[12].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[12].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_passingdoubleyellow = torch.cdist(type_dict[13].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[13].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_roadedgeboundary = torch.cdist(type_dict[15].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[15].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_roadedgemedian = torch.cdist(type_dict[16].repeat(128,1,1), state_future_left_pos).min(dim=1).values[None, :, :] if type_dict[16].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    decoded_dict['map_features/future/left_edge'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([5, 128, 80]).to(device),
                                                                                    future_dist_brokensinglewhite,
                                                                                    future_dist_solidsinglewhite, 
                                                                                    future_dist_soliddoublewhite,
                                                                                    future_dist_brokensingleyellow,
                                                                                    future_dist_brokendoubleyellow,
                                                                                    future_dist_solidsingleyellow,
                                                                                    future_dist_soliddoubleyellow,
                                                                                    future_dist_passingdoubleyellow,
                                                                                    np.inf * torch.ones([1, 128, 80]).to(device),
                                                                                    future_dist_roadedgeboundary,
                                                                                    future_dist_roadedgemedian], 0).argmin(dim=0).cpu().numpy())
    
    # Future Right edge
    future_dist_brokensinglewhite = torch.cdist(type_dict[6].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[6].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_solidsinglewhite = torch.cdist(type_dict[7].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[7].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_soliddoublewhite = torch.cdist(type_dict[8].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[8].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_brokensingleyellow = torch.cdist(type_dict[9].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[9].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_brokendoubleyellow = torch.cdist(type_dict[10].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[10].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_solidsingleyellow = torch.cdist(type_dict[11].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[11].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_soliddoubleyellow = torch.cdist(type_dict[12].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[12].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_passingdoubleyellow = torch.cdist(type_dict[13].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[13].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_roadedgeboundary = torch.cdist(type_dict[15].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[15].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_roadedgemedian = torch.cdist(type_dict[16].repeat(128,1,1), state_future_right_pos).min(dim=1).values[None, :, :] if type_dict[16].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    decoded_dict['map_features/future/right_edge'] = tf.convert_to_tensor(torch.cat([np.inf * torch.ones([6, 128, 80]).to(device),
                                                                            future_dist_brokensinglewhite,
                                                                            future_dist_solidsinglewhite, 
                                                                            future_dist_soliddoublewhite,
                                                                            future_dist_brokensingleyellow,
                                                                            future_dist_brokendoubleyellow,
                                                                            future_dist_solidsingleyellow,
                                                                            future_dist_soliddoubleyellow,
                                                                            future_dist_passingdoubleyellow,
                                                                            np.inf * torch.ones([1, 128, 80]).to(device),
                                                                            future_dist_roadedgeboundary,
                                                                            future_dist_roadedgemedian], 0).argmin(dim=0).cpu().numpy())
    
    # Future Nearest TCD
    future_dist_stopsign = torch.cdist(type_dict[17].repeat(128,1,1), state_future_pos).min(dim=1).values[None, :, :] if type_dict[17].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_crosswalk = torch.cdist(type_dict[18].repeat(128,1,1), state_future_pos).min(dim=1).values[None, :, :] if type_dict[18].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_dist_speedbump = torch.cdist(type_dict[19].repeat(128,1,1), state_future_pos).min(dim=1).values[None, :, :] if type_dict[19].shape[0] > 0 else np.inf * torch.ones([1, 128, 80]).to(device)
    future_tcd_tensor = torch.cat([np.inf * torch.ones([16, 128, 80]).to(device),
                                    future_dist_stopsign,
                                    future_dist_crosswalk, 
                                    future_dist_speedbump], 0)
    
    decoded_dict['map_features/future/nearest_tcd'] = tf.convert_to_tensor(future_tcd_tensor.argmin(dim=0).cpu().numpy())


def _parse(value, use_map_feats=False):
    decoded_example = tf.io.parse_single_example(value, features_description)

    if(use_map_feats):
        temp_dict = {}
        for key in decoded_example.keys():
            temp_dict[key] = tf2pytorch(decoded_example[key])
      	
        map_feature_extractor(temp_dict)
        past_states = tf.stack([
        decoded_example['state/past/x'],
        decoded_example['state/past/y'],
        decoded_example['state/past/length'],
        decoded_example['state/past/width'],
        decoded_example['state/past/bbox_yaw'],
        decoded_example['state/past/velocity_x'],
        decoded_example['state/past/velocity_y'],
        temp_dict['map_features/past/road'],
        temp_dict['map_features/past/left_edge'],
        temp_dict['map_features/past/right_edge'],
        temp_dict['map_features/past/nearest_tcd']
        ], -1)
        
        cur_states = tf.stack([
        decoded_example['state/current/x'],
        decoded_example['state/current/y'],
        decoded_example['state/current/length'],
        decoded_example['state/current/width'],
        decoded_example['state/current/bbox_yaw'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/current/velocity_y'],
        temp_dict['map_features/current/road'],
        temp_dict['map_features/current/left_edge'],
        temp_dict['map_features/current/right_edge'],
        temp_dict['map_features/current/nearest_tcd']
        ], -1)
        
        future_states = tf.stack([
        decoded_example['state/future/x'],
        decoded_example['state/future/y'],
        decoded_example['state/future/length'],
        decoded_example['state/future/width'],
        decoded_example['state/future/bbox_yaw'],
        decoded_example['state/future/velocity_x'],
        decoded_example['state/future/velocity_y'],
        temp_dict['map_features/future/road'],
        temp_dict['map_features/future/left_edge'],
        temp_dict['map_features/future/right_edge'],
        temp_dict['map_features/future/nearest_tcd']
        ], -1)
        
    else:
        past_states = tf.stack([
        decoded_example['state/past/x'],
        decoded_example['state/past/y'],
        decoded_example['state/past/length'],
        decoded_example['state/past/width'],
        decoded_example['state/past/bbox_yaw'],
        decoded_example['state/past/velocity_x'],
        decoded_example['state/past/velocity_y']
        ], -1)       

        cur_states = tf.stack([
        decoded_example['state/current/x'],
        decoded_example['state/current/y'],
        decoded_example['state/current/length'],
        decoded_example['state/current/width'],
        decoded_example['state/current/bbox_yaw'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/current/velocity_y']
        ], -1)
        
        future_states = tf.stack([
        decoded_example['state/future/x'],
        decoded_example['state/future/y'],
        decoded_example['state/future/length'],
        decoded_example['state/future/width'],
        decoded_example['state/future/bbox_yaw'],
        decoded_example['state/future/velocity_x'],
        decoded_example['state/future/velocity_y']
        ], -1) 

    input_states = tf.concat([past_states, cur_states], 1)[..., :2]


    gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0
    gt_future_is_valid = tf.concat(
        [past_is_valid, current_is_valid, future_is_valid], 1)

    # If a sample was not seen at all in the past, we declare the sample as
    # invalid.
    sample_is_valid = tf.reduce_any(
        tf.concat([past_is_valid, current_is_valid], 1), 1)
    inputs = {
        'input_states': input_states,
        'gt_future_states': gt_future_states,
        'gt_future_is_valid': gt_future_is_valid,
        'object_type': decoded_example['state/type'],
        'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
        'sample_is_valid': sample_is_valid,
    }
    return inputs


def parse_tensor(value, features_description):
    decoded_example = tf.io.parse_single_example(value, features_description)
    for key in decoded_example.keys():
        decoded_example[key] = tf2pytorch(decoded_example[key])
    past_states = torch.stack([
        decoded_example['state/past/x'],
        decoded_example['state/past/y'],
        decoded_example['state/past/length'],
        decoded_example['state/past/width'],
        decoded_example['state/past/bbox_yaw'],
        decoded_example['state/past/velocity_x'],
        decoded_example['state/past/velocity_y']
    ], -1)

    cur_states = torch.stack([
        decoded_example['state/current/x'],
        decoded_example['state/current/y'],
        decoded_example['state/current/length'],
        decoded_example['state/current/width'],
        decoded_example['state/current/bbox_yaw'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/current/velocity_y']
    ], -1)

    input_states = torch.cat([past_states, cur_states], 1)[..., :2]

    future_states = torch.stack([
        decoded_example['state/future/x'],
        decoded_example['state/future/y'],
        decoded_example['state/future/length'],
        decoded_example['state/future/width'],
        decoded_example['state/future/bbox_yaw'],
        decoded_example['state/future/velocity_x'],
        decoded_example['state/future/velocity_y']
    ], -1)

    gt_future_states = torch.cat([past_states, cur_states, future_states], 1)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0
    gt_future_is_valid = torch.cat(
        [past_is_valid, current_is_valid, future_is_valid], 1)

    # If a sample was not seen at all in the past, we declare the sample as
    # invalid.
    sample_is_valid = torch.cat([past_is_valid, current_is_valid], 1)
    sample_is_valid = sample_is_valid.ne(1).any(axis=1)

    inputs = {
        'input_states': input_states,
        'gt_future_states': gt_future_states,
        'gt_future_is_valid': gt_future_is_valid,
        'object_type': decoded_example['state/type'],
        'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
        'sample_is_valid': sample_is_valid,
    }

    return inputs

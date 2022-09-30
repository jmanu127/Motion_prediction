# roadwayreco
Spring 2021 Capstone Project

## Evaluation Metrics
[Waymo Challenge Doc](https://waymo.com/open/challenges/2021/motion-prediction/)

### Variables need for MotionMetrics:
* pred_trajectory
* pred_score (Fake the score since the simple model does not generate any score per predicted trajectory.)
* gt_trajectory: inputs['gt_future_states']
* gt_is_valid: inputs['gt_is_valid']
* object_type: inputs['object_type']

Other input fields needed:
- inputs['tracks_to_predict'] (the demo only keep this one for evaluation)
- inputs['sample_is_valid'] (for masking)

### Waymo Evaluation Metrics 
* **Average Precision (AP)**
    * **Trajectory bucket** for the ground truth of the objects to be predicted: straight / straight-left / straight-right / left / right / left u-turn / right u-turn / stationary
* **miss rate**
* minADE (Average Displacement Error)
* minFDE (Final Displacement Error)
* Overlap rate.

### Submission Rules
* Open date?
* You can only submit against the Test Set 3 times every 30 days
* ?? Only submissions that run faster than 70 ms/frame on a Nvidia Tesla V100 GPU will be eligible to win the challenge.


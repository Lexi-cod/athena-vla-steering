## athena_feedback batch 20260720_113830
- task: put the red mug on the left plate  (target red_coffee_mug_1)
- mode: ATHENA steer-AWAY (feedback)  gamma=4.0  window_steps=4  diverge_thresh=-0.1
- rollouts: 12 (offset 0)
- success: 0/12 (0%)
- target object grasped (max displacement): 0/12
- target object moved at all (>1e-4 m): 1/12
- rollouts where steering triggered >=1x: 12/12
- csv: athena_feedback_batch_20260720_113830.csv

## paired_athena_feedback (same-noise-seed) 20260720_123922
- task: put the red mug on the left plate  (target red_coffee_mug_1)
- UNSTEERED vs ATHENA-away  gamma=4.0  window=4  diverge_thresh=-0.1
- paired trials: 12 (init states 0..11)
- unsteered success: 0/12   away success: 0/12
- flipped to success (away succeeded where unsteered didn't, same seed): 0/12
- behavior changed (different grasp or target-disp delta >1e-3, same seed): 2/12
- csv: paired_athena_feedback_20260720_123922.csv

## athena_feedback_middle_bowl batch 20260720_130116
- task[middle_bowl]: put the middle black bowl on the plate  (spatial (middle vs front/back bowl))
- mode: ATHENA steer-AWAY (feedback)  gamma=4.0  window=4  diverge_thresh=-0.1
- rollouts: 12 (offset 0)
- success: 4/12 (33%)   target grasped: 4/12   target moved(>1e-4): 4/12
- steering triggered >=1x: 9/12
- csv: athena_feedback_middle_bowl_batch_20260720_130116.csv

## paired_athena_middle_bowl (same-noise-seed) 20260720_134145
- task[middle_bowl]: put the middle black bowl on the plate  (spatial (middle vs front/back bowl))
- UNSTEERED vs ATHENA-away  gamma=4.0  window=4  diverge_thresh=-0.1
- paired trials: 12
- unsteered success: 5/12   away success: 5/12   (net = +0)
- flipped TO success (steering helped, same seed): 1/12
- flipped TO failure (steering HURT, same seed): 1/12
- behavior changed (same seed): 8/12
- csv: paired_athena_middle_bowl_20260720_134145.csv

## render_noise_floor[middle_bowl] 20260720_154222
- task: put the middle black bowl on the plate
- BOTH arms unsteered, identical init state + per-replan noise_seed
- paired trials: 12
- OUTCOME disagreement (success vs failure): 2/12 (17%)
- GRASPED-OBJECT disagreement: 3/12 (25%)
- interpretation: a steering effect must exceed this floor to be detectable
- csv: render_noise_floor_middle_bowl_20260720_154222.csv

## athena_feedback_orange_juice_steeroff batch 20260720_161608
- task[orange_juice]: pick up the orange juice and put it in the basket  (identity fixation (orange juice vs 6 other grocery items))
- mode: STEER-OFF baseline  gamma=4.0  window=4  diverge_thresh=-0.1
- rollouts: 12 (offset 0)
- success: 0/12 (0%)   target grasped: 0/12   target moved(>1e-4): 1/12
- steering triggered >=1x: 0/12
- csv: athena_feedback_orange_juice_steeroff_batch_20260720_161608.csv

## athena_feedback_orange_juice batch 20260720_170230
- task[orange_juice]: pick up the orange juice and put it in the basket  (identity fixation (orange juice vs 6 other grocery items))
- mode: ATHENA steer-AWAY (feedback)  gamma=4.0  window=4  diverge_thresh=999.0
- rollouts: 12 (offset 0)
- success: 0/12 (0%)   target grasped: 0/12   target moved(>1e-4): 1/12
- steering triggered >=1x: 12/12
- csv: athena_feedback_orange_juice_batch_20260720_170230.csv


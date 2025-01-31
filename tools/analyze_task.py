import numpy as np
import glob
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def analyze_dataset(data_path):
    # Get all npz files
    files = glob.glob(os.path.join(data_path, "*.npz"))
    
    # Statistics we want to track
    total_episodes = len(files)
    episode_lengths = []
    success_rate = 0
    ee_pos_ranges = []
    ee_rot_ranges = []
    gripper_changes = []
    
    for f in tqdm(files):
        with np.load(f, allow_pickle=True) as data:
            episode = data['arr_0']
            episode_len = len(episode)
            episode_lengths.append(episode_len)
            
            # Track EE position and rotation ranges
            ee_positions = np.array([step['obs']['ee_pos'] for step in episode])
            ee_rotations = np.array([step['obs']['ee_euler'] for step in episode])
            gripper_states = np.array([step['obs']['gripper_open'] for step in episode])
            
            ee_pos_range = np.max(ee_positions, axis=0) - np.min(ee_positions, axis=0)
            ee_rot_range = np.max(ee_rotations, axis=0) - np.min(ee_rotations, axis=0)
            gripper_changes.append(np.sum(np.abs(np.diff(gripper_states)) > 0.5))
            
            ee_pos_ranges.append(ee_pos_range)
            ee_rot_ranges.append(ee_rot_range)
    
    stats = {
        "total_episodes": total_episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "avg_ee_pos_range": np.mean(ee_pos_ranges, axis=0),
        "avg_ee_rot_range": np.mean(ee_rot_ranges, axis=0),
        "avg_gripper_changes": np.mean(gripper_changes),
    }
    
    return stats

# Analyze both datasets
can_stats = analyze_dataset("data/can")
stack_stats = analyze_dataset("data/stack")

print("\nCan Task Statistics:")
print("=" * 50)
for k, v in can_stats.items():
    print(f"{k}: {v}")

print("\nStack Task Statistics:")
print("=" * 50)
for k, v in stack_stats.items():
    print(f"{k}: {v}")
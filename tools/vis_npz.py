# PAL TOOL
# for record_sim.py, we get npz files per episode
# not visible in the UI
# this tool is to visualize the npz files

import numpy as np
import open3d as o3d
import os
import argparse
import pdb
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the NPZ file
can_data = np.load('data/can/demo00010.npz', allow_pickle=True)


# official collection
print("Arrays in the file:", can_data.files)

# Print contents of each array
for name in can_data.files:
    print(f"\nArray '{name}':")

print('\n'.join([f"{k}: {v.shape}" for k,v in can_data['arr_0'][0]['obs'].items()]))

pdb.set_trace()

# OUR collection
stack_data = np.load('data/stack/demo00040.npz', allow_pickle=True)
print('\n'.join([f"{k}: {v.shape}" for k,v in stack_data['arr_0'][0]['obs'].items()]))
pdb.set_trace()



# compare them two
print("\nComparing shapes between datasets:")
print("=" * 50)
print("Can dataset observation shapes:")
for k, v in can_data['arr_0'][0]['obs'].items():
    print(f"{k}: {v.shape}")

print("\nStack dataset observation shapes:")
for k, v in stack_data['arr_0'][0]['obs'].items():
    print(f"{k}: {v.shape}")

print("\nShape differences:")
print("-" * 50)
all_keys = set(can_data['arr_0'][0]['obs'].keys()) | set(stack_data['arr_0'][0]['obs'].keys())
for k in all_keys:
    if k in can_data['arr_0'][0]['obs'] and k in stack_data['arr_0'][0]['obs']:
        if can_data['arr_0'][0]['obs'][k].shape != stack_data['arr_0'][0]['obs'][k].shape:
            print(f"{k}: can={can_data['arr_0'][0]['obs'][k].shape} vs stack={stack_data['arr_0'][0]['obs'][k].shape}")
    elif k in can_data['arr_0'][0]['obs']:
        print(f"{k}: Only in can dataset with shape {can_data['arr_0'][0]['obs'][k].shape}")
    else:
        print(f"{k}: Only in stack dataset with shape {stack_data['arr_0'][0]['obs'][k].shape}")

pdb.set_trace()

import os
import glob
import numpy as np

print('Checking cache...')
bad_count = 0
for f in glob.glob('.cache_tacact_n80_front/*.npy'):
    try:
        np.load(f, allow_pickle=True)
    except Exception as e:
        os.remove(f)
        bad_count += 1

print(f'Done! Removed {bad_count} bad files.')

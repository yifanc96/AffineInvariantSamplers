
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))



dim = [4, 8, 16, 32, 64, 128]
stretch_move_result = [118.8, 213.2, 387.1, 808.4, 1401.1, 3021.3]
side_move_result = [42.3, 73.1, 140.7, 302, 500.6, 1398.3]
walk_move_result = [] # no subsampled


dim_larger = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
EKM_result = [14.8, 20.6, 24.1, 28.1, 41.8, 42, 50.8, 60.3, 64.1 ] # no subsampled
LWM_result = [12.2, 17.8, 22.2, 26.3, 34.9, 35.4, 43.3, 56, 60.1] # no subsampled
HWM_result = [3.46, 3.54, 4.05, 3.84, 4.5, 5.19, 7.05, 9.90, 12.83] # no subsampled
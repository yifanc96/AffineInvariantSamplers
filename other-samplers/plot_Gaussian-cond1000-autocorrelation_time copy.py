
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))



dim = [4, 8, 16, 32, 64, 128]
stretch_move_result = [66.5, 131.9, 280, 577.5, 1009.7, 2043.6]
side_move_result = [34.4, 62.1, 126.7, 256.9, 491.8, 1000.1]
walk_move_result = [] # no subsampled

dim_larger = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
EKM_result = [12.5, 13.3, 14.6, 20.3, 25.8, 26.5, 33.4, 41.0, 50.1] # no subsampled
LWM_result = [13.6, 13.1, 15.8, 19.6, 22.1, 28.6, 33.3, 43.2, 49.7] # no subsampled
HWM_result = [2.1, 2.2, 2.2, 2.3, 3.52, 3.66, 5.49, 8.2, 10.95] # no subsampled
import cv2
import numpy as np

depth1 = cv2.imread("./data/mass_train_data_rgb_depth/depth_10000000a35c8f1b_20211125013924.png")
depth2 = cv2.imread("./data/mass_train_data_rgb_depth/depth_10000000a35c8f1b_20211125013924.png", cv2.IMREAD_UNCHANGED)

print(np.amax(depth1))
print(depth1.shape)

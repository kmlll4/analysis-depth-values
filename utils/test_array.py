import numpy as np
import cv2

mask = np.array([[0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,1,1,1],
                 [0,0,0,0,0],
                 [0,0,0,0,0]
                 ], dtype=np.uint8)

depth = np.array([[0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,0,2500,2400,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]
                  ], dtype=np.uint16)

floor = 2550
one_pig = depth[mask!=0]
height = floor - one_pig
print(height[one_pig!=0].mean())
print(height.mean())
# print(one_pig[mask!=0].mean())

# indices = np.where(mask!=0)
# for row_index, col_index in zip(indices[0], indices[1]):
#     print("Row:", row_index, "Column:", col_index)


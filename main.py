from analysis_depth_values import pkl_loader, decoder
import cv2

pkl_path = "/home/kmiyazaki/Workspace/analysis-depth-values/data/latest_from_xxx_abs_path_only_floor_depth.pkl"
df = pkl_loader(pkl_path)
df.sort_values(by="filename")
print(df["filename"])

# df = df[["polygon", "filename"]]
# # print(df.info())

for _, row in df[:20].iterrows():
    rle = row["polygon"].__dict__["_rle"]
    mask = decoder(rle)
    cv2.imwrite(row["filename"], mask)

# print(df)

# print(type(df["polygon"][0]))
# rle = df["polygon"][0].__dict__["_rle"]
# mask = decoder(rle)

# cv2.imwrite("segmentation.png", mask)
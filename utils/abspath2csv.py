from analysis_depth_values import pkl_loader, decoder
import cv2

pkl_path = "/home/kmiyazaki/Workspace/analysis-depth-values/data/latest_from_xxx_abs_path_only_floor_depth.pkl"
df = pkl_loader(pkl_path)

# df=df["abs_path"]
# is_duplicated = df.drop_duplicates()
df.to_csv("ファイル名.csv")


# print(is_duplicated)
import pandas as pd

df = pd.read_csv('depth_values.csv')
print(df.info())

nonzero = df[df["count_zero"]!=0]

print(nonzero["diff"].max())
print(nonzero["diff"].mean())

# cnt = 0
# c = 0
# for index, row in df.iterrows():
#     if row["erorr"] != 0.0:
#         e = round(row["erorr"], 2)
#         print(row["filename"], row["bbox"], f"{e}%")
#         cnt += 1
#     else:
#         c+=1

# print("Instance :", c+cnt)
# print("NonZero :", cnt)

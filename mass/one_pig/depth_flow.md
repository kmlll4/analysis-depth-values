
- 改善前
```
resize
depth = depth[y:y+h, x:x+w] # 豚切り抜き
depth 回転
w,l 
縮小
height 変換
前処理
mask
```

- 改善後
```
depth[depth > floor_depth] = floor_depth
resize  
depth = depth[y:y+h, x:x+w] # 豚切り抜き
depth 回転
W,L

height = np.where(depth != 0, floor_depth - depth.astype(int), 0)
height 正規化(mask含む)
height 回転
height 縮小
mask
前処理

```

- evaluation.py
depth[depth > pig.floor_depth] = pig.floor_depth
resize
depth = depth[y0:y0 + h, x0:x0 + w] # 豚切り抜き

height = np.where(depth != 0, pig.floor_depth - depth.astype(int), 0)
height 正規化(mask含む)
height 回転
height 縮小
mask
前処理

w,lはpklから


- メモ
・re_1
depth[depth > floor_depth] = floor_depthの追加
・re_2
正規化追加

・re_3
rotaterをseqへ

・re_4

depth[depth > floor_depth] = floor_depth
resize
depth = depth[y:y+h, x:x+w] # 豚切り抜き
depth 回転
w,l 
縮小
height 変換
height 正規化
前処理
mask

・w,l
改善後にw,lをpklから引っ張ってくるように変更
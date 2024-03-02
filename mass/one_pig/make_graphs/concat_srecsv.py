# -*- coding:utf-8 -*-
import os
import pandas as pd

df_list = []

base_output_dir = "/workspace/mass/ecopork-main/output/one_pig_tracking_result"
bokeh_output_dir = "/workspace/mass/ecopork-main/output/one_pig_tracking_result/bokeh"
dir_list = os.listdir(base_output_dir)

for target_dir in dir_list:

    if target_dir.endswith("jpg") or target_dir.endswith("csv") or target_dir.endswith("bokeh"):
        continue

    data_dir = os.path.join(base_output_dir, target_dir)
    pig_id = target_dir.split("_")[0]
    target_weight = float(target_dir.split("_")[1])

    # csv_file_path = os.path.join(data_dir, '{}_sre.csv'.format(pig_id))
    csv_file_path = os.path.join(data_dir, "{}_sre_pos.csv".format(pig_id))

    df = pd.read_csv(csv_file_path)

    # id_df = pd.DataFrame([[pig_id, target_weight]]*len(df), columns=['pig_id', 'true_weight'])

    # df = pd.concat([id_df, df], axis=1)

    df_list.append(df)

pd.concat(df_list).to_csv(os.path.join(bokeh_output_dir, "concat_sre_pos.csv"), index=False)

#!/usr/bin/env python
# coding: utf-8

# ## 動画データ解析
# 動画データ（20210621_132120.bag）に体重推計をかけ、同じ豚でどれだけ推計値がばらつくか検証する

import csv
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib
import seaborn as sns
import os
from tqdm import tqdm

sns.set()
sns.set(font="IPAexGothic")


def df_filter(df, eval_type, error_th=4):
    """
    description

    Parameters
    ----------


    Returns
    -------

    """
    df = df.copy()
    # df['>4%'] = df['error'].apply(lambda x : True if x > error_th  else False)
    df[">4%"] = df["error"] > error_th
    # df['filter'] = ~(~df['occulusion'] * ~df['anno_val'] * ~df['edge_filter'] * ~df['keypoint_filter'])  # arya_filter
    if eval_type == "sre":
        df["filter"] = df["edge_filter"] | df["anomaly_wl"] | df["anomaly_mask"] | df["incomplete_mask"]  # sre_filter
    elif eval_type == "araya":
        df["filter"] = df["edge_filter"] | df["anomaly_mask"] | df["incomplete_mask"]  # for araya origin
    # df['filter'] = ~(~df['occulusion'] * ~df['anno_val'] * ~df['edge_filter'] * ~df['keypoint_filter'])  # araya_filter

    df.loc[(df[">4%"] == False) & (df["filter"] == False), "誤差区分"] = "<4%"
    df.loc[(df[">4%"] == False) & (df["filter"] == True), "誤差区分"] = "<4%, Filterで除外"
    df.loc[(df[">4%"] == True) & (df["filter"] == False), "誤差区分"] = ">=4%"
    df.loc[(df[">4%"] == True) & (df["filter"] == True), "誤差区分"] = ">=4%, Filterで除外"

    return df


def save_histgram(df, output_path):
    """
    description

    Parameters
    ----------


    Returns
    -------

    """
    fig = plt.figure()
    # plt.xticks(rotation=45)

    # plt.figure(figsize=(10,10))
    fig.set_figheight(6)
    fig.set_figwidth(20)

    # 余白を設定
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    ax1 = fig.add_subplot(2, 2, 1)  # 1行２列の１番目
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)

    flatui = [
        "#6666FF",
        "#993366",
        "#0000FF",
        "#CC0000",
    ]
    # sns.palplot(sns.color_palette(flatui, 4))
    current_palette = sns.color_palette(flatui, 4)
    sns.set_palette(current_palette)

    # sns.histplot(data=df, x = 'pred', binwidth=1 ,hue='',multiple='stack', hue_order=['<4%, Filterで除外','>=4%, Filterで除外','<4%', '>=4%',] ,ax=ax1)
    # sns.histplot(data=df, x = 'pred', binwidth=1 ,hue='>4%',multiple='stack', hue_order=[False,True] ,ax=ax2)
    sns.histplot(data=df, x="w", binwidth=10, hue="誤差区分", multiple="stack", hue_order=["<4%", ">=4%"], ax=ax1, binrange=(0, 2000), bins=30)
    sns.histplot(data=df, x="l", binwidth=10, hue="誤差区分", multiple="stack", hue_order=["<4%", ">=4%"], ax=ax3, binrange=(0, 2000), bins=30)
    sns.histplot(data=df, x="pred", binwidth=1, hue="誤差区分", multiple="stack", hue_order=["<4%", ">=4%"], ax=ax2, binrange=(40, 140), bins=30)

    figure = sns.histplot().get_figure()
    figure.savefig(output_path)

    ax1.set_ylim(0, 90)
    ax2.set_ylim(0, 90)
    ax3.set_ylim(0, 90)

    plt.title(output_path.split("/")[-1].split(".")[0])

    plt.clf()
    plt.close()


def save_histgram_movie(df, movie_path):
    """
    description

    Parameters
    ----------


    Returns
    -------

    """
    ## 1frameずつのグラフ画像書き出し
    os.makedirs(movie_path, exist_ok=True)
    for i in tqdm(range(len(df))):
        if i >= 1200:
            break

        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(10)
        # 余白を設定
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        ax1 = fig.add_subplot(1, 1, 1)

        # sns.histplot(data=df[:i+2], x = 'pred', binwidth=1 ,hue='',multiple='stack', hue_order=['<4%, Filterで除外','>=4%, Filterで除外','<4%','>=4%'] ,ax=ax1)
        sns.histplot(data=df[: i + 2], x="pred", binwidth=1, hue="誤差区分", multiple="stack", hue_order=["<4%, Filterで除外", ">=4%, Filterで除外", "<4%", ">=4%"], ax=ax1, binrange=(40, 140), bins=30)

        ax1.set_ylim(0, 90)
        ax1.set_xlim(40, 140)

        plt.title(output_path.split("/")[-1].split(".")[0])

        plt.savefig(os.path.join(movie_path, "{:05d}.jpg".format(i)))
        plt.clf()
        plt.close()


def run(pig_id, csv_path, output_path, movie_path, eval_type):

    df = pd.read_csv(csv_path)
    df = df_filter(df, eval_type)
    save_histgram(df, output_path)
    # save_histgram_movie(df, movie_path)

    mean = df["pred"].mean()
    print(pig_id + "平均値:" + str(mean))

    # rate = len(df[df[''] == '<4%']) / (len(df[df[''] == '<4%']) + len(df[df[''] == '>=4%']))
    # min_value = df.loc[(df['']=='<4%') | (df['']=='>=4%'), 'pred'].min()
    # max_value = df.loc[(df['']=='<4%') | (df['']=='>=4%'), 'pred'].max()

    # return rate, min_value, max_value
    return df


if __name__ == "__main__":
    base_data_dir = "/workspace/mass/ecopork-main/output/one_pig_tracking_result"
    dir_list = os.listdir(base_data_dir)

    all_pig_df_list = []

    for target_dir in dir_list:
        if not os.path.isdir(os.path.join(base_data_dir, target_dir)):
            continue

        data_dir = os.path.join(base_data_dir, target_dir)
        pig_id = target_dir.split("_")[0]
        pig_weight = target_dir.split("_")[1]

        csv_path = os.path.join(data_dir, pig_id + "_sre.csv")
        output_path = os.path.join(data_dir, pig_id + "_sre.jpg")
        movie_path = os.path.join(data_dir, "graph_sre")
        if os.path.exists(csv_path):
            sre_df = run(pig_id, csv_path, output_path, movie_path, "sre")
            sre_df["model_type"] = "sre"

        csv_path = os.path.join(data_dir, pig_id + "_sre_v2.csv")
        output_path = os.path.join(data_dir, pig_id + "_sre_v2.jpg")
        movie_path = os.path.join(data_dir, "graph_sre_v2")
        if os.path.exists(csv_path):
            sre_v2_df = run(pig_id, csv_path, output_path, movie_path, "sre")
            sre_v2_df["model_type"] = "sre_v2"

        csv_path = os.path.join(data_dir, pig_id + "_araya.csv")
        output_path = os.path.join(data_dir, pig_id + "_araya.jpg")
        movie_path = os.path.join(data_dir, "graph_araya")
        if os.path.exists(csv_path):
            araya_df = run(pig_id, csv_path, output_path, movie_path, "araya")
            araya_df["model_type"] = "araya"

        pig_df = pd.concat([sre_df, sre_v2_df, araya_df])
        pig_df["data_dir"] = target_dir
        pig_df["pig_id"] = pig_id
        pig_df["pig_weight"] = float(pig_weight)

        pig_df = pig_df[["data_dir", "model_type", "pig_id", "pig_weight", "pred", "error", "誤差区分"]].copy()
        all_pig_df_list.append(pig_df)

    # 結果まとめ
    all_pig_df = pd.concat(all_pig_df_list)
    all_pig_df = all_pig_df[(all_pig_df["誤差区分"] == "<4%") | (all_pig_df["誤差区分"] == ">=4%")].copy()
    all_pig_df["誤差範囲内"] = 0
    all_pig_df.loc[all_pig_df["誤差区分"] == "<4%", "誤差範囲内"] = 1
    all_pig_df["体重帯"] = all_pig_df["pig_weight"].astype(int) // 10 * 10
    all_pig_df["error"] = all_pig_df["pred"] - all_pig_df["pig_weight"]

    pig_group = all_pig_df.groupby(["data_dir", "model_type", "pig_id", "pig_weight"])
    pig_summary = pd.DataFrame(data=[], index=pig_group.count().index)
    pig_summary["min"] = pig_group["pred"].min()
    pig_summary["max"] = pig_group["pred"].max()
    pig_summary["mean"] = pig_group["pred"].mean()
    pig_summary["data_count"] = pig_group["誤差区分"].count()
    pig_summary["good_count"] = pig_group["誤差範囲内"].sum()

    pig_summary["rate"] = pig_summary["good_count"] / pig_summary["data_count"]
    pig_summary.to_csv("test.csv")

    all_pig_df = all_pig_df[all_pig_df["model_type"] == "araya"].copy()

    # 体重帯別
    unique_key_df = all_pig_df[["体重帯"]].drop_duplicates("体重帯").sort_values("体重帯").reset_index(drop=True)

    data_list = []
    xticks_list = []
    xticks_label_list = []

    for idx, row in unique_key_df.iterrows():
        data_list.append(all_pig_df[(all_pig_df["体重帯"] == row["体重帯"])]["error"].copy())
        xticks_list.append(idx + 1)
        xticks_label_list.append(row["体重帯"])

    # バイオリンプロット
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.violinplot(data_list)
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label_list)
    ax.set_xlabel("weight")
    ax.set_yticks([-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_ylabel("error(kg)")

    ax.plot([0, 10], [0, 0], c="red")
    ax.set_xlim([0, 10])
    ax.set_ylim([-80, 80])

    plt.title("araya")

    plt.savefig("vaiolinplot_weight.jpg")
    plt.clf()
    plt.close()

    # 豚ID別
    unique_key_df = all_pig_df[["data_dir", "pig_id", "pig_weight"]].drop_duplicates("data_dir").sort_values("pig_weight").reset_index(drop=True)

    data_list = []
    xticks_list = []
    xticks_label_list = []

    for idx, row in unique_key_df.iterrows():
        data_list.append(all_pig_df[(all_pig_df["data_dir"] == row["data_dir"])]["error"].copy())
        xticks_list.append(idx + 1)
        xticks_label_list.append("{}_{}kg".format(row["pig_id"], row["pig_weight"]))

    # バイオリンプロット
    plt.rcParams["figure.subplot.bottom"] = 0.25

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.violinplot(data_list)
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label_list, rotation=270)
    ax.set_xlabel("pig_id (sorted by weight)")
    ax.set_yticks([-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_ylabel("error(kg)")

    ax.plot([0, 32], [0, 0], c="red")
    ax.set_xlim([0, 32])
    ax.set_ylim([-80, 80])

    plt.title("araya")

    plt.savefig("vaiolinplot_pig_id.jpg")
    plt.clf()
    plt.close()

    print(data_list[0].min(), data_list[0].max())

    # result_csv_lines = [['pig_id', 'pig_weight',
    #                      'sre_rate', 'sre_v2_rate', 'araya_rate',
    #                      'sre_min', 'sre_max',
    #                      'sre_v2_min', 'sre_v2_max',
    #                      'araya_min', 'araya_max']]

    #     sre_rate = '-'
    #     sre_v2_rate = '-'
    #     araya_rate = '-'
    #     sre_min = '-'
    #     sre_max = '-'
    #     sre_v2_min = '-'
    #     sre_v2_max = '-'
    #     araya_min = '-'
    #     araya_max = '-'

    #     csv_line = [pig_id, pig_weight,
    #                 str(sre_rate), str(sre_v2_rate), str(araya_rate),
    #                 str(sre_min), str(sre_max),
    #                 str(sre_v2_min), str(sre_v2_max),
    #                 str(araya_min), str(araya_max)]
    #     result_csv_lines.append(csv_line)

    # with open(os.path.join(base_data_dir, 'eval_result.csv'), 'w', newline='') as file_:
    #     writer = csv.writer(file_, lineterminator='\n')
    #     writer.writerows(result_csv_lines)

# 各計測項目の計測値をグラフにするプログラム、入力はCSVファイル、フォーマットは"8_cases_results.csv"を参照
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
import japanize_matplotlib

# Read the CSV file (please replace with your actual CSV file path)
# csv_file_path = "aaa_data.csv"
csv_file_path = "8_cases_results.csv"
# csv_file_path = "8_cases_full_bbox_only.csv"
df = pd.read_csv(csv_file_path)
stent_added = False
separate_group = True

langage = "jp"
# langage = "en"

# Ensure the date column format is correct
if csv_file_path == "aaa_data.csv":
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
else:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    
if csv_file_path == "aaa_data_stent_length_20250519.csv":
    stent_added = True

# Automatically determine the earliest date for each ID and calculate the number of months (starting from 0 at the first image)
df["months_since_first"] = df.groupby("id")["date"].transform(lambda x: (x - x.min()).dt.days // 30)

# inverse the LRC value
# df['LRC_main'] = np.where(df['LRC_main'] != 0, 1 / df['LRC_main'], np.nan)  # set zero to nan
# df['LRC_left'] = np.where(df['LRC_left'] != 0, 1 / df['LRC_left'], np.nan)    # keep 0 as 0
# df['LRC_right'] = np.where(df['LRC_right'] != 0, 1 / df['LRC_right'], np.nan)    # keep 0 as 0
# OR
df['LRC_main'] = np.where(df['LRC_main'] != 0, 1 / df['LRC_main'], 0)    # keep 0 as 0
df['LRC_left'] = np.where(df['LRC_left'] != 0, 1 / df['LRC_left'], 0)    # keep 0 as 0
df['LRC_right'] = np.where(df['LRC_right'] != 0, 1 / df['LRC_right'], 0)    # keep 0 as 0

# Keep two decimal places
df["aaa_volume"] = df["aaa_volume"].round(2)
df["max_short_diameter_ellipse"] = df["max_short_diameter_ellipse"].round(2)
if stent_added:
    df["stent_length"] = df["stent_length"].round(2)
# df["max_short_diameter_no_ellipse"] = df["max_short_diameter_no_ellipse"].round(2)

# Get the list of unique IDs
unique_ids = df["id"].unique()

selected_ids = ["2012EN23", "2012EN27", "2012GO19", "2012PO12","2012EN17", "2018EAF11","2018EAF9", "2015EEN4"]
custom_labels = {
    "2018EAF11": "拡大あり-A", # 2018EAF11
    "2018EAF9": "拡大あり-B", # 2018EAF9
    "2012EN17": "拡大あり-C", # 2012EN17
    "2015EEN4": "拡大あり-D", # 2015EEN4
    "2012EN23": "拡大なし-E", # 2012EN23
    "2012EN27": "拡大なし-F", # 2012EN27
    "2012GO19": "拡大なし-G", # 2012GO19
    "2012PO12": "拡大なし-H", # 2012PO12
}

df_filtered = df[df["id"].isin(selected_ids)]

# Chart variable names and y-axis labels
plot_variables = {
    "aaa_volume": "大動脈瘤体積(cm³)",
    "max_short_diameter_ellipse": "最大短径(mm)",
    "stent_length": "ステント中心線全長(mm)",
    "bbox_x": "AAA内bounding boxのxサイズ(mm)",
    "bbox_y": "AAA内bounding boxのyサイズ(mm)",
    "bbox_z": "AAA内bounding boxのzサイズ(mm)",
    "boundingBox_volume": "Bounding boxの体積(cm³)",
    "LRC_main":"ステントmain部分のLRC",
    "LRC_left":"ステントleft leg部分のLRC",
    "LRC_right":"ステントright leg部分のLRC",
    "DSB_main":"ステントmain部分のDSB",
    "DSB_left":"ステントleft leg部分のDSB",
    "DSB_right":"ステントright leg部分のDSB",
}
# plot_variables = {
#     "bbox_x": "bounding boxのxサイズ(mm)",
#     "bbox_y": "bounding boxのyサイズ(mm)",
#     "bbox_z": "bounding boxのzサイズ(mm)",
#     "boundingBox_volume": "Bounding boxの体積(cm³)",
# }

if langage == "en":
    custom_labels = {
    "2018EAF11": "Enlarged case-2018EAF11",
    "2012EN17": "Enlarged case-2012EN17",
    "2018EAF9": "Enlarged case-2018EAF9",
    "2015EEN4": "Enlarged case-2015EEN4",
    "2012EN23": "Non-enlarged case-2012EN23",
    "2012EN27": "Non-enlarged case-2012EN27",
    "2012GO19": "Non-enlarged case-2012GO19",
    "2012PO12": "Non-enlarged case-2012PO12",
    }
    plot_variables = {
    "aaa_volume": "AAA volume(cm³)",
    "max_short_diameter_ellipse": "Short-axis diameter(mm)",
    "stent_length": "Stent length in AAA(mm)",
    "bbox_x": "bounding box's x size in AAA(mm)",
    "bbox_y": "bounding box's y size in AAA(mm)",
    "bbox_z": "bounding box's z size in AAA(mm)",
    "boundingBox_volume": "Bounding box volume(cm³)",
    
    }

# Font size
plt.rcParams.update({"font.size": 16})

# calculate the overall min and max for each variable among all selected groups
y_range_dict = {}
for var in plot_variables.keys():
    min_v = df_filtered[var].min()
    max_v = df_filtered[var].max()
    # Add a small margin (5%) to avoid points sticking to the axis edge
    delta = (max_v - min_v) * 0.05 if max_v > min_v else 1
    y_range_dict[var] = (min_v - delta, max_v + delta)

group_ids = {
    "拡大なし": ["2012EN23", "2012EN27", "2012GO19", "2012PO12"],
    "拡大あり": ["2018EAF11", "2018EAF9", "2012EN17", "2015EEN4"]
}

color_palette = plt.get_cmap('tab10')
uid_list = list(selected_ids)
color_cycle = cycle([color_palette(i) for i in range(10)])
color_mapping = {uid: next(color_cycle) for uid in uid_list}

# color_mapping = {}
# for group, ids in group_ids.items():
#     if group == "拡大なし":
#         cmap = cm.Blues
#         lower, upper = 0.5, 0.8  # 取值范围调整为0.5~0.8
#     else:
#         cmap = cm.Oranges
#         lower, upper = 0.3, 0.7  # 可以根据需要调整
#     n = len(ids)
#     for i, uid in enumerate(ids):
#         # 如果有多个症例，则均匀在指定区间取值
#         if n > 1:
#             value = lower + (upper - lower) * (i / (n - 1))
#         else:
#             value = (lower + upper) / 2
#         color_mapping[uid] = cmap(value)

def plot_group(ids_to_plot, plot_title_suffix=""):
    df_sub = df_filtered[df_filtered["id"].isin(ids_to_plot)]
    for var, ylabel in plot_variables.items():
        plt.figure(figsize=(8, 6))
        for uid in ids_to_plot:
            subset = df_sub[df_sub["id"] == uid]
            label = custom_labels.get(uid, uid)
            color = color_mapping.get(uid, "black")
            plt.plot(subset["months_since_first"], subset[var], marker="o", linestyle="-",
                     label=label, linewidth=3, color=color)
        if langage == "en":
            plt.xlabel("Monthes after EVAR", fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
        else:
            plt.xlabel("術後経過月数", fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
        plt.legend(fontsize=14, title_fontsize=16)
        plt.grid(True)
        # Set the y-axis range to be consistent across groups
        plt.ylim(y_range_dict[var])
        plt.savefig(f"./figures/{ylabel}{plot_title_suffix}.png")
        plt.close()

if separate_group:
    for group, ids in group_ids.items():
        suffix = f"_{group}" if langage == "jp" else f"_{'Enlarged' if group=='拡大あり' else 'NonEnlarged'}"
        plot_group(ids, plot_title_suffix=suffix)
else:
    plot_group(selected_ids)

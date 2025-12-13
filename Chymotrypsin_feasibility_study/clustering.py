import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


# マスター側：2列だけに統一（既に2列ならそのまま、そうでなければ計算して2列化）
def _ensure_green_red(df_master_raw: pd.DataFrame) -> pd.DataFrame:
    cols = set(df_master_raw.columns)
    if {"green", "red"}.issubset(cols):
        out = df_master_raw.loc[:, ["green", "red"]].copy()
    elif {
        "FITC_Sum",
        "FITC_Max",
        "FITC_Min",
        "mCherry_Sum",
        "mCherry_Max",
        "mCherry_Min",
    }.issubset(cols):
        out = pd.DataFrame(
            {
                "green": df_master_raw["FITC_Sum"]
                - (df_master_raw["FITC_Max"] + df_master_raw["FITC_Min"]),
                "red": df_master_raw["mCherry_Sum"]
                - (df_master_raw["mCherry_Max"] + df_master_raw["mCherry_Min"]),
            }
        )
    else:
        raise ValueError(
            "master_csv に必要な列がありません。`green, red` もしくは "
            "`FITC_Sum/FITC_Max/FITC_Min/mCherry_Sum/mCherry_Max/mCherry_Min` を含めてください。"
        )
    # サイズ削減のため、本当に2列だけにする
    return out


def clustering_assign(df_new, df_master) -> pd.DataFrame:
    # 新規データ側：まず green / red を計算して2列に整える
    need_cols = [
        "FITC_Sum",
        "FITC_Max",
        "FITC_Min",
        "mCherry_Sum",
        "mCherry_Max",
        "mCherry_Min",
    ]
    df_new = df_new.dropna(subset=need_cols).copy()

    df_new = df_new.assign(
        green=lambda d: d["FITC_Sum"] - (d["FITC_Max"] + d["FITC_Min"]),
        red=lambda d: d["mCherry_Sum"] - (d["mCherry_Max"] + d["mCherry_Min"]),
    )
    # 以降の散布図やQCで使うため、元の列は保持しつつ、クラスタリングに使うビューを2列に限定
    df_new_view = df_new.loc[:, ["green", "red"]].copy()
    df_new_view["origin"] = "df_new"

    df_master_2col = _ensure_green_red(df_master)
    df_master_2col = df_master_2col.dropna(subset=["green", "red"]).copy()
    df_master_2col["origin"] = "df_master"

    # クラスタリング用の統合データ（2列＋originのみ）
    df_cluster = pd.concat([df_new_view, df_master_2col], ignore_index=True)
    df_cluster = df_cluster.dropna(subset=["green", "red"]).reset_index(drop=True)

    # 角度θと距離rを計算（green=x, red=y として極座標化）
    theta = np.arctan2(df_cluster["green"], df_cluster["red"])
    r = np.sqrt(df_cluster["green"] ** 2 + df_cluster["red"] ** 2)
    features = np.vstack([theta, r]).T

    # 標準化 → KMeans(=5)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(features_scaled)

    # 参照点に最も近い重心でラベル再割当て
    centroids = kmeans.cluster_centers_
    reference_points = np.array(
        [
            [-1.1, -1.1],  # => 0
            [-0.1, 0.9],  # => 1
            [0.4, 0.3],  # => 2
            [0.8, -1.4],  # => 3
            [3.2, -0.6],  # => 4
        ]
    )
    matched_indices = pairwise_distances_argmin(centroids, reference_points)
    cluster_to_new_label = {i: matched_indices[i] for i in range(len(matched_indices))}
    new_labels = np.vectorize(cluster_to_new_label.get)(kmeans.labels_)

    df_cluster["cluster_label"] = new_labels

    # group 付与（クラスタ → グループ）
    df_cluster["group"] = None
    df_cluster.loc[df_cluster["cluster_label"] == 0, "group"] = 1  # CTRB1
    df_cluster.loc[df_cluster["cluster_label"].isin([1, 2, 3]), "group"] = 2  # CTRB2_2
    df_cluster.loc[df_cluster["cluster_label"] == 4, "group"] = 3  # CTRB2_sub

    # 以降で使うのは新規データのみ（green/red + 付与ラベル）
    df_labeled_sample = df_cluster[df_cluster["origin"] == "df_new"].copy()

    return df_labeled_sample, df_cluster, new_labels


def scatter_image_output(df_new, fields, sample, output):
    scatter_fig = plt.figure(figsize=(8.7, 8))
    axes = scatter_fig.gca()
    axes.scatter(
        df_new[df_new.group == 1].green, df_new[df_new.group == 1].red, c="red", s=5
    )
    axes.scatter(
        df_new[df_new.group == 2].green, df_new[df_new.group == 2].red, c="blue", s=5
    )
    axes.scatter(
        df_new[df_new.group == 3].green, df_new[df_new.group == 3].red, c="green", s=5
    )
    axes.set_xlabel("Green fluorescence intensity (a.u.)", fontsize=24)
    axes.set_ylabel("Red fluorescence intensity (a.u.)", fontsize=24)
    axes.set_xlim(0, 150000)
    axes.set_ylim(0, 150000)
    axes.tick_params(labelsize=24)

    scatter_fig.tight_layout()
    axes.set_title(f"{sample}_{fields.data_name}", fontsize=16)
    output(
        scatter_fig,
        f"{sample}_{fields.data_name}_scatter.png",
        "scatter",
        bbox_inches="tight",
    )


def paramater_calculator(df_new, total_wells):
    # 以降の比率計算は df_new の group を使用
    cluster_1 = (df_new["group"] == 1).sum() / total_wells
    cluster_2 = (df_new["group"] == 2).sum() / total_wells
    cluster_3 = (df_new["group"] == 3).sum() / total_wells

    ctrb2_unit = (cluster_2 * 2500 / 0.01) * 0.415 / 1000

    if cluster_1 + cluster_2 + cluster_3 == 0:
        ctrb1_per_ctrb2 = 0
    else:
        ctrb1_per_ctrb2 = cluster_1 / (cluster_1 + cluster_2 + cluster_3)

    return cluster_1, cluster_2, cluster_3, ctrb2_unit, ctrb1_per_ctrb2

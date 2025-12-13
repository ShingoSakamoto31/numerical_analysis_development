import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

THRESHOLD = 12578
SPECIFICITY_95_LINE = 0.00023  # 対照群全体に対する95%特異度ライン


def cela3a_filtering(df_new, df_filter):
    # ここからフィルタリングの操作
    # 新しい列を初期化
    df_new["6 spots_average"] = None
    df_new["6 spots_count"] = 0

    for mpi in range(1, 13):
        df_new_field = df_new[df_new["MultiPointIndex"] == mpi]
        df_filter_field = df_filter[df_filter["MultiPointIndex"] == mpi]

        for idx, row in df_new_field.iterrows():
            cx, cy = row["CenterX"], row["CenterY"]

            # CenterX/Y ±範囲でマッチ（MultiPointIndexはすでに限定済み）
            matched_aux = df_filter_field[
                (df_filter_field["Center2X"].between(cx - 8, cx + 8))
                & (df_filter_field["Center2Y"].between(cy - 10, cy + 10))
            ]

            count_matched = len(matched_aux)

            if count_matched > 0:
                avg_mCherry = matched_aux["_6_spots_mean"].mean()
                df_new.at[idx, "6 spots_average"] = avg_mCherry
                df_new.at[idx, "6 spots_count"] = count_matched

    df_cela3a = df_new[(df_new["mCherry_mean"] - df_new["6 spots_average"]) < 300]
    green_cela3a = (
        df_cela3a.FITC_Sum - (df_cela3a.FITC_Max + df_cela3a.FITC_Min)
    ).to_numpy()
    red_cela3a = (
        df_cela3a.mCherry_Sum - (df_cela3a.mCherry_Max + df_cela3a.mCherry_Min)
    ).to_numpy()

    return df_cela3a, green_cela3a, red_cela3a


def scatter_image_output(green_cela3a, red_cela3a, fields, sample, output):
    scatter_fig = plt.figure(figsize=(8.7, 8))
    axes = scatter_fig.gca()
    axes.scatter(green_cela3a, red_cela3a, color="black", s=3)
    axes.vlines(THRESHOLD, 0, 50000, colors="red", linestyles="dashed", linewidth=3)
    axes.set_xlabel("Green fluorescence intensity (a.u.)", fontsize=24)
    axes.set_ylabel("Red fluorescence intensity (a.u.)", fontsize=24)
    axes.set_xlim(0, 50000)
    axes.set_ylim(0, 50000)
    axes.tick_params(labelsize=24)

    scatter_fig.tight_layout()
    axes.set_title(f"{fields.data_name}_{sample}_scatter.png", fontsize=16)
    output(
        scatter_fig,
        f"{fields.data_name}_{sample}_scatter.png",
        "scatter",
        bbox_inches="tight",
    )


def calculate_cela3a(green_cela3a, total_wells):
    cela3a = np.size(green_cela3a[green_cela3a > THRESHOLD]) / total_wells
    cela3a_unit = (cela3a * 100 / 0.01) * 0.415

    return cela3a, cela3a_unit


def calculate_required_number_of_measurements(green_cela3a, total_wells):
    lower, _ = proportion_confint(
        count=np.size(green_cela3a[green_cela3a > THRESHOLD]),
        nobs=total_wells,
        alpha=0.005,
        method="beta",
    )  # 99.5%信頼区間

    print(type(lower))  # 一応、lowerの型がfloatであることを目視で確認しておく
    assert isinstance(
        lower, float
    )  # スクリプトの実装上、lowerがfloat型であることは確実なので、lowerがfloatでない場合に例外を起こすこととする（すると比較の式の警告が消える）

    required_number_of_measurements = 3 if lower < SPECIFICITY_95_LINE else 1

    return required_number_of_measurements

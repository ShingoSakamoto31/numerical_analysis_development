import matplotlib.pyplot as plt
import numpy as np

from chymotrypsin_output import (
    LAMBDA_CV_STANDARD,
    INSUFFICIENT_WELL_FIELD_COUNT_STANDARD,
    GREEN_CV_STANDARD,
    RED_CV_STANDARD,
    GREEN_MEAN_INTENSITY_STANDARD,
    RED_MEAN_INTENSITY_STANDARD,
    RQC_RED_MEAN_INTENSITY_STANDARD,
    TOTAL_LAMBDA_LOW_STANDARD,
    TOTAL_LAMBDA_HIGH_STANDARD,
    RQC_TOTAL_LAMBDA_LOW_STANDARD,
    RQC_TOTAL_LAMBDA_HIGH_STANDARD,
    RQC_CTRB1_PER_CTRB2_LOW_STANDARD,
    RQC_CTRB1_PER_CTRB2_HIGH_STANDARD,
    RQC_CTRB2_LOW_STANDARD,
    RQC_CTRB2_HIGH_STANDARD,
    PQC_CTRB1_PER_CTRB2_LOW_STANDARD,
    PQC_CTRB1_PER_CTRB2_HIGH_STANDARD,
    PQC_CTRB2_LOW_STANDARD,
    PQC_CTRB2_HIGH_STANDARD,
)


def lane_qc_conductor(
    fields,
    well_filtered_fields,
    df_new,
    sample,
    new_labels,
    df_cluster,
    total_lambda,
    output,
) -> tuple:
    # QC画像データ出力
    x = fields.field_numbers
    qc_fig = plt.figure(figsize=(15, 10))

    # 散布図
    qc_fig_scatter = qc_fig.add_subplot(2, 3, 1)
    qc_fig_scatter.scatter(
        df_new[df_new.group == 1].green, df_new[df_new.group == 1].red, c="red", s=5
    )
    qc_fig_scatter.scatter(
        df_new[df_new.group == 2].green, df_new[df_new.group == 2].red, c="blue", s=5
    )
    qc_fig_scatter.scatter(
        df_new[df_new.group == 3].green, df_new[df_new.group == 3].red, c="green", s=5
    )
    qc_fig_scatter.set_xlabel("Green fluorescence intensity (a.u.)", fontsize=16)
    qc_fig_scatter.set_ylabel("Red fluorescence intensity (a.u.)", fontsize=16)
    qc_fig_scatter.set_xlim(0, 150000)
    qc_fig_scatter.set_ylim(0, 150000)
    qc_fig_scatter.set_title(fields.data_name, fontsize=16)
    qc_fig_scatter.tick_params(labelsize=16)

    # lambda
    qc_fig_lambda = qc_fig.add_subplot(2, 3, 2)
    _lambda = [
        None if len(v) == 0 else len(v) / int(v["All_well_count"].iat[0])
        for v in well_filtered_fields
    ]

    # ウェル数が5000を超える視野のデータのみを用いてlambda CVを求める。
    sufficient_field_signals = [v for v in _lambda if v is not None]
    lambda_cv = np.round(
        np.std(sufficient_field_signals) / np.average(sufficient_field_signals), 3
    )
    lambda_cv_result = "NG" if lambda_cv > LAMBDA_CV_STANDARD else None
    lambda_cv_title_color = "red" if lambda_cv_result == "NG" else "black"

    # グラフのY軸
    y = [float(0) if v is None else v for v in _lambda]

    qc_fig_lambda.bar(x, y, color="blue")
    qc_fig_lambda.set_xlabel("Slice", fontsize=16)
    qc_fig_lambda.set_ylabel("lambda", fontsize=16)
    qc_fig_lambda.tick_params(labelsize=16)
    qc_fig_lambda.set_title(
        f"lambda CV = {lambda_cv}", fontsize=16, color=lambda_cv_title_color
    )

    # well count
    qc_fig_well = qc_fig.add_subplot(2, 3, 3)
    y = [0 if len(v) == 0 else int(v["All_well_count"].iat[0]) for v in fields]
    insufficient_well_field_count = np.size([v for v in y if v <= 5000])
    insufficient_well_field_count_result = (
        "NG"
        if insufficient_well_field_count > INSUFFICIENT_WELL_FIELD_COUNT_STANDARD
        else None
    )
    well_count_title_color = (
        "red" if insufficient_well_field_count_result == "NG" else "black"
    )

    qc_fig_well.bar(x, y, color="blue")
    qc_fig_well.set_xlabel("Slice", fontsize=16)
    qc_fig_well.set_ylabel("Total well", fontsize=16)
    qc_fig_well.tick_params(labelsize=16)
    qc_fig_well.set_title(
        f"< 5000 well: {insufficient_well_field_count}",
        fontsize=16,
        color=well_count_title_color,
    )

    # green signal intensity
    qc_fig_green = qc_fig.add_subplot(2, 3, 4)
    _green_intensity = [
        None
        if len(v) == 0
        else np.mean((v["FITC_Sum"] - (v["FITC_Max"] + v["FITC_Min"])).to_numpy())
        for v in well_filtered_fields
    ]
    sufficient_fields_green = [v for v in _green_intensity if v is not None]
    y = [float(0) if v is None else float(v) for v in _green_intensity]

    green_cv = np.round(
        np.std(sufficient_fields_green) / np.average(sufficient_fields_green), 3
    )
    green_cv_result = "NG" if green_cv > GREEN_CV_STANDARD else None
    green_cv_title_color = "red" if green_cv_result == "NG" else "black"

    green_mean = float(np.average(sufficient_fields_green))
    green_mean_result = "NG" if green_mean < GREEN_MEAN_INTENSITY_STANDARD else None

    qc_fig_green.bar(x, y, color="blue")
    qc_fig_green.set_xlabel("Slice", fontsize=16)
    qc_fig_green.set_ylabel("Green signal intensity (AU)", fontsize=16)
    qc_fig_green.tick_params(labelsize=16)
    qc_fig_green.set_title(
        f"green intensity CV = {green_cv}", fontsize=16, color=green_cv_title_color
    )

    # red signal intensity
    qc_fig_red = qc_fig.add_subplot(2, 3, 5)
    _red_intensity = [
        None
        if len(v) == 0
        else np.mean(
            (v["mCherry_Sum"] - (v["mCherry_Max"] + v["mCherry_Min"])).to_numpy()
        )
        for v in well_filtered_fields
    ]
    sufficient_fields_red = [v for v in _red_intensity if v is not None]
    y = [float(0) if v is None else float(v) for v in _red_intensity]

    red_cv = np.round(
        np.std(sufficient_fields_red) / np.average(sufficient_fields_red), 3
    )
    red_cv_result = "NG" if red_cv > RED_CV_STANDARD else None
    red_cv_title_color = "red" if red_cv_result == "NG" else "black"

    red_mean = float(np.average(sufficient_fields_red))
    if sample == "rQC":
        red_mean_standard = RQC_RED_MEAN_INTENSITY_STANDARD
    else:
        red_mean_standard = RED_MEAN_INTENSITY_STANDARD
    red_mean_result = "NG" if red_mean < red_mean_standard else None

    qc_fig_red.bar(x, y, color="blue")
    qc_fig_red.set_xlabel("Slice", fontsize=16)
    qc_fig_red.set_ylabel("Red signal intensity (AU)", fontsize=16)
    qc_fig_red.tick_params(labelsize=16)
    qc_fig_red.set_title(
        f"Red intensity CV = {red_cv}", fontsize=16, color=red_cv_title_color
    )

    # マスターデータも含めたクラスタリングの結果
    qc_fig_cluster = qc_fig.add_subplot(2, 3, 6)
    # 色の設定（クラスタ番号に対応させる）
    cluster_colors = {0: "red", 1: "blue", 2: "dodgerblue", 3: "lightblue", 4: "orange"}

    for label in np.unique(new_labels):
        mask = new_labels == label
        data = df_cluster.loc[mask]

        qc_fig_cluster.scatter(
            data["green"],
            data["red"],
            label=f"Cluster {label}",
            color=cluster_colors.get(label, "gray"),
            s=0.01,
            alpha=0.7,
        )

    qc_fig_cluster.set_title("Angle + Distance-based Clustering")
    qc_fig_cluster.set_xlabel("x")
    qc_fig_cluster.set_ylabel("y")
    qc_fig_cluster.set_xlim(0, 200000)
    qc_fig_cluster.set_ylim(0, 150000)
    qc_fig_cluster.legend(
        title="Cluster Label", markerscale=50, fontsize=8, loc="center right"
    )

    # total lambda result
    if sample == "rQC":
        total_lambda_low_standard = RQC_TOTAL_LAMBDA_LOW_STANDARD
        total_lambda_high_standard = RQC_TOTAL_LAMBDA_HIGH_STANDARD
    else:
        total_lambda_low_standard = TOTAL_LAMBDA_LOW_STANDARD
        total_lambda_high_standard = TOTAL_LAMBDA_HIGH_STANDARD

    total_lambda_result = (
        "NG"
        if (total_lambda < total_lambda_low_standard)
        or (total_lambda_high_standard < total_lambda)
        else None
    )

    # 判定
    if "NG" in (
        lambda_cv_result,
        insufficient_well_field_count_result,
        green_cv_result,
        red_cv_result,
    ):
        lane_result = "NG"
        qc_fig.suptitle(f"{fields.data_name}_QC (NG).png", fontsize=16, color="red")

    elif "NG" in (green_mean_result, red_mean_result, total_lambda_result):
        lane_result = "check"
        qc_fig.suptitle(f"{fields.data_name}_QC (check).png", fontsize=16, color="red")

    else:
        lane_result = None
        qc_fig.suptitle(f"{fields.data_name}_QC (OK)", fontsize=16)

    qc_fig.tight_layout()
    output(qc_fig, f"{fields.data_name}_QC.png", "qc", bbox_inches="tight")

    return (
        lane_result,
        lambda_cv,
        lambda_cv_result,
        insufficient_well_field_count,
        insufficient_well_field_count_result,
        green_cv,
        green_cv_result,
        red_cv,
        red_cv_result,
        green_mean,
        green_mean_result,
        red_mean,
        red_mean_standard,
        red_mean_result,
        total_lambda_low_standard,
        total_lambda_high_standard,
        total_lambda_result,
    )


def QC_sample_qc_conductor(sample, ctrb1_per_ctrb2, ctrb2_unit) -> tuple:
    # rQC, pQCの判定
    if sample == "rQC" or sample == "pQC":
        if sample == "rQC":
            rqc_ctrb1_per_ctrb2 = ctrb1_per_ctrb2
            rqc_ctrb2 = ctrb2_unit
            pqc_ctrb1_per_ctrb2 = None
            pqc_ctrb2 = None

            if (
                (RQC_CTRB1_PER_CTRB2_LOW_STANDARD < rqc_ctrb1_per_ctrb2)
                & (rqc_ctrb1_per_ctrb2 < RQC_CTRB1_PER_CTRB2_HIGH_STANDARD)
                & (RQC_CTRB2_LOW_STANDARD < rqc_ctrb2)
                & (rqc_ctrb2 < RQC_CTRB2_HIGH_STANDARD)
            ):
                qc_result = None
            else:
                qc_result = "NG"

        else:
            pqc_ctrb1_per_ctrb2 = ctrb1_per_ctrb2
            pqc_ctrb2 = ctrb2_unit
            rqc_ctrb1_per_ctrb2 = None
            rqc_ctrb2 = None

            if (
                (PQC_CTRB1_PER_CTRB2_LOW_STANDARD < pqc_ctrb1_per_ctrb2)
                and (pqc_ctrb1_per_ctrb2 < PQC_CTRB1_PER_CTRB2_HIGH_STANDARD)
                and (PQC_CTRB2_LOW_STANDARD < pqc_ctrb2)
                and (pqc_ctrb2 < PQC_CTRB2_HIGH_STANDARD)
            ):
                qc_result = None
            else:
                qc_result = "NG"

    else:
        rqc_ctrb1_per_ctrb2 = rqc_ctrb2 = pqc_ctrb1_per_ctrb2 = pqc_ctrb2 = (
            qc_result
        ) = None

    return rqc_ctrb1_per_ctrb2, rqc_ctrb2, pqc_ctrb1_per_ctrb2, pqc_ctrb2, qc_result

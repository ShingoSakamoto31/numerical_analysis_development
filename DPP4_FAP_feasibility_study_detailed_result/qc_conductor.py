import matplotlib.pyplot as plt
import numpy as np

from dpp4_fap_output import (
    GREEN_CV_STANDARD,
    GREEN_MEAN_INTENSITY_STANDARD,
    INSUFFICIENT_WELL_FIELD_COUNT_STANDARD,
    LAMBDA_CV_STANDARD,
    QC_VALUE_DPP4_HOMODIMER_RATIO_HIGH_STANDARD,
    QC_VALUE_DPP4_HOMODIMER_RATIO_LOW_STANDARD,
    QC_VALUE_FAP_HIGH_STANDARD,
    QC_VALUE_FAP_LOW_STANDARD,
    RED_CV_STANDARD,
    TOTAL_LAMBDA_HIGH_STANDARD,
    TOTAL_LAMBDA_LOW_STANDARD,
)


def lane_qc_conductor(
    fields, well_filtered_fields, green, red, total_lambda, output
) -> tuple:
    # QC画像データ出力
    x = fields.field_numbers
    qc_fig = plt.figure(figsize=(15, 10))

    # 散布図
    qc_fig_scatter = qc_fig.add_subplot(2, 3, 1)
    qc_fig_scatter.scatter(green, red, c="black", s=1)
    qc_fig_scatter.set_xlabel("Green fluorescence intensity (a.u.)", fontsize=16)
    qc_fig_scatter.set_ylabel("Red fluorescence intensity (a.u.)", fontsize=16)
    qc_fig_scatter.set_xlim(0, 25000)
    qc_fig_scatter.set_ylim(0, 100000)
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

    qc_fig_red.bar(x, y, color="blue")
    qc_fig_red.set_xlabel("Slice", fontsize=16)
    qc_fig_red.set_ylabel("Red signal intensity (AU)", fontsize=16)
    qc_fig_red.tick_params(labelsize=16)
    qc_fig_red.set_title(
        f"Red intensity CV = {red_cv}", fontsize=16, color=red_cv_title_color
    )

    # total lambda result
    total_lambda_result = (
        "NG"
        if (total_lambda < TOTAL_LAMBDA_LOW_STANDARD)
        or (TOTAL_LAMBDA_HIGH_STANDARD < total_lambda)
        else None
    )

    if "NG" in (
        lambda_cv_result,
        insufficient_well_field_count_result,
        green_cv_result,
        red_cv_result,
    ):
        lane_result = "NG"
        qc_fig.suptitle(f"{fields.data_name}_QC (NG).png", fontsize=16, color="red")

    elif "NG" in (total_lambda_result, green_mean_result):
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
        total_lambda_result,
    )


def QC_sample_qc_conductor(sample, fap_homodimer, dpp4_homodimer_ratio) -> tuple:
    if sample == "QC":
        qc_fap = fap_homodimer
        qc_dpp4_homodimer_ratio = dpp4_homodimer_ratio
        if (
            (QC_VALUE_FAP_LOW_STANDARD < qc_fap)
            & (qc_fap < QC_VALUE_FAP_HIGH_STANDARD)
            & (QC_VALUE_DPP4_HOMODIMER_RATIO_LOW_STANDARD < qc_dpp4_homodimer_ratio)
            & (qc_dpp4_homodimer_ratio < QC_VALUE_DPP4_HOMODIMER_RATIO_HIGH_STANDARD)
        ):
            qc_result = None
        else:
            qc_result = "NG"
    else:
        qc_fap = qc_dpp4_homodimer_ratio = qc_result = None

    return qc_fap, qc_dpp4_homodimer_ratio, qc_result

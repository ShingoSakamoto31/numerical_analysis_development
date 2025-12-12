import pandas as pd

LAMBDA_CV_STANDARD = 0.15
INSUFFICIENT_WELL_FIELD_COUNT_STANDARD = 4
GREEN_CV_STANDARD = 0.1
RED_CV_STANDARD = 0.1
GREEN_MEAN_INTENSITY_STANDARD = 3600
TOTAL_LAMBDA_LOW_STANDARD = 0.02
TOTAL_LAMBDA_HIGH_STANDARD = 0.093
QC_VALUE_FAP_LOW_STANDARD = 0.0101
QC_VALUE_FAP_HIGH_STANDARD = 0.0214
QC_VALUE_DPP4_HOMODIMER_RATIO_LOW_STANDARD = 0.506
QC_VALUE_DPP4_HOMODIMER_RATIO_HIGH_STANDARD = 0.628


def result(
    fields_data_name: str,
    sample: str,
    total_wells: int,
    dpp4_homodimer: float | None,
    dpp4_fap_heterodimer: float | None,
    fap_homodimer: float | None,
    dpp4_homodimer_ratio: float | None,
    lane_result: str | None,
    lambda_cv: float | None,
    lambda_cv_result: str | None,
    insufficient_well_field_count: int,
    insufficient_well_field_count_result: str | None,
    green_cv: float | None,
    green_cv_result: str | None,
    red_cv: float | None,
    red_cv_result: str | None,
    green_mean: float | None,
    green_mean_result: str | None,
    total_lambda: float | None,
    total_lambda_result: str | None,
    qc_fap: float | None,
    qc_dpp4_homodimer_ratio: float | None,
    qc_result: str | None,
    fields_image_analysis_method: str,
    analysis_time: str,
    script_dir: str,
) -> pd.Series:
    return pd.Series(
        {
            "data": fields_data_name,
            "sample_name": sample,
            "all_well": total_wells,
            "DPP4_homodimer": dpp4_homodimer,
            "DPP4_FAP_heterodimer": dpp4_fap_heterodimer,
            "FAP_homodimer": fap_homodimer,
            "DPP4/FAP": dpp4_homodimer_ratio,
            "lane_result": lane_result,
            "lambda_cv": lambda_cv,
            "lambda_cv:standard": LAMBDA_CV_STANDARD,
            "lambda_cv:result": lambda_cv_result,
            "insufficient_well_field": insufficient_well_field_count,
            "insufficient_well_field_count:standard": INSUFFICIENT_WELL_FIELD_COUNT_STANDARD,
            "insufficient_well_field_count:result": insufficient_well_field_count_result,
            "green_cv": green_cv,
            "green_cv:standard": GREEN_CV_STANDARD,
            "green_cv:result": green_cv_result,
            "red_cv": red_cv,
            "red_cv:standard": RED_CV_STANDARD,
            "red_cv:result": red_cv_result,
            "green_mean": green_mean,
            "green_mean_standard": GREEN_MEAN_INTENSITY_STANDARD,
            "green_mean_result": green_mean_result,
            "total_lambda": total_lambda,
            "total_lambda_low_standard": TOTAL_LAMBDA_LOW_STANDARD,
            "total_lambda_high_standard": TOTAL_LAMBDA_HIGH_STANDARD,
            "total_lambda_result": total_lambda_result,
            "QC_value_FAP": qc_fap,
            "QC_value_FAP_low_standard": QC_VALUE_FAP_LOW_STANDARD,
            "QC_value_FAP_high_standard": QC_VALUE_FAP_HIGH_STANDARD,
            "QC_value_DPP4_homodimer_ratio": qc_dpp4_homodimer_ratio,
            "QC_value_DPP4_homodimer_ratio_low_standard": QC_VALUE_DPP4_HOMODIMER_RATIO_LOW_STANDARD,
            "QC_value_DPP4_homodimer_ratio_high_standard": QC_VALUE_DPP4_HOMODIMER_RATIO_HIGH_STANDARD,
            "QC_result": qc_result,
            "image_method": fields_image_analysis_method,
            "analysis_time": analysis_time,
            "analysis_method": script_dir,
        }
    )

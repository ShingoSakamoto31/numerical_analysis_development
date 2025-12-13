import pandas as pd


LAMBDA_CV_STANDARD = 0.3
INSUFFICIENT_WELL_FIELD_COUNT_STANDARD = 4
GREEN_CV_STANDARD = 0.2
RED_CV_STANDARD = 0.2
TOTAL_LAMBDA_LOW_STANDARD = 0.002
TOTAL_LAMBDA_HIGH_STANDARD = 0.05
RQC_LAMBDA_LOW_STANDARD = 0.0101
RQC_LAMBDA_HIGH_STANDARD = 0.0201
PQC_LAMBDA_LOW_STANDARD = 0.001


def result(
    fields_data_name: str,
    sample: str,
    total_wells: int,
    cela3a: float | None,
    cela3a_unit: float | None,
    required_number_of_measurements: int | None,
    lane_result: str | None,
    lambda_cv: float | None,
    lambda_cv_result: str | None,
    insufficient_well_field_count: int,
    insufficient_well_field_count_result: str | None,
    green_cv: float | None,
    green_cv_result: str | None,
    red_cv: float | None,
    red_cv_result: str | None,
    total_lambda: float | None,
    total_lambda_result: str | None,
    rqc_lambda: float | None,
    pqc_high_red_intensity_lambda: float | None,
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
            "CELA3A_lambda": cela3a,
            "CELA3A": cela3a_unit,
            "required_number_of_measurements": required_number_of_measurements,
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
            "total_lambda": total_lambda,
            "total_lambda_low_standard": TOTAL_LAMBDA_LOW_STANDARD,
            "total_lambda_high_standard": TOTAL_LAMBDA_HIGH_STANDARD,
            "total_lambda_result": total_lambda_result,
            "rQC_lambda": rqc_lambda,
            "rQC_lambda_low_standard": RQC_LAMBDA_LOW_STANDARD,
            "rQC_lambda_high_standard": RQC_LAMBDA_HIGH_STANDARD,
            "pQC_high_red_intensity_lambda": pqc_high_red_intensity_lambda,
            "pQC_high_red_intensity_lambda_low_standard": PQC_LAMBDA_LOW_STANDARD,
            "QC_result": qc_result,
            "image_method": fields_image_analysis_method,
            "analysis_time": analysis_time,
            "analysis_method": script_dir,
        }
    )

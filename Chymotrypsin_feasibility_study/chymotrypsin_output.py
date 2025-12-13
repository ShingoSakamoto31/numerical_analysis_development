import pandas as pd

LAMBDA_CV_STANDARD = 0.22
INSUFFICIENT_WELL_FIELD_COUNT_STANDARD = 4
GREEN_CV_STANDARD = 0.11
RED_CV_STANDARD = 0.11
GREEN_MEAN_INTENSITY_STANDARD = 47000
RED_MEAN_INTENSITY_STANDARD = 54000
RQC_RED_MEAN_INTENSITY_STANDARD = 50000
TOTAL_LAMBDA_LOW_STANDARD = 0.0012
TOTAL_LAMBDA_HIGH_STANDARD = 0.018
RQC_TOTAL_LAMBDA_LOW_STANDARD = 0.0306
RQC_TOTAL_LAMBDA_HIGH_STANDARD = 0.0419
RQC_CTRB1_PER_CTRB2_LOW_STANDARD = 0.274
RQC_CTRB1_PER_CTRB2_HIGH_STANDARD = 0.373
RQC_CTRB2_LOW_STANDARD = 2.09
RQC_CTRB2_HIGH_STANDARD = 2.98
PQC_CTRB1_PER_CTRB2_LOW_STANDARD = 0.096
PQC_CTRB1_PER_CTRB2_HIGH_STANDARD = 0.217
PQC_CTRB2_LOW_STANDARD = 0.7
PQC_CTRB2_HIGH_STANDARD = 1.38
MASTER_CSV_BASENAME = "all_raw_data_clinical_study.csv"


def result(
    fields_data_name: str,
    sample: str,
    total_wells: int,
    cluster_1: float | None,
    cluster_2: float | None,
    cluster_3: float | None,
    ctrb2_unit: float | None,
    ctrb1_per_ctrb2: float | None,
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
    red_mean: float | None,
    red_mean_standard: float,
    red_mean_result: str | None,
    total_lambda: float | None,
    total_lambda_low_standard: float,
    total_lambda_high_standard: float,
    total_lambda_result: str | None,
    rqc_ctrb1_per_ctrb2: float | None,
    rqc_ctrb2: float | None,
    pqc_ctrb1_per_ctrb2: float | None,
    pqc_ctrb2: float | None,
    qc_result: str | None,
    fields_image_analysis_method: str,
    analysis_time: str,
    script_dir: str,
):
    return pd.Series(
        {
            "data": fields_data_name,
            "sample_name": sample,
            "all_well": total_wells,
            "CTRB1": cluster_1,
            "CTRB2_lambda": cluster_2,
            "cluster_3": cluster_3,
            "CTRB2": ctrb2_unit,
            "CTRB1/CTRB2": ctrb1_per_ctrb2,
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
            "red_mean": red_mean,
            "red_mean_standard": red_mean_standard,
            "red_mean_result": red_mean_result,
            "total_lambda": total_lambda,
            "total_lambda_low_standard": total_lambda_low_standard,
            "total_lambda_high_standard": total_lambda_high_standard,
            "total_lambda_result": total_lambda_result,
            "rQC_CTRB1_PER_CTRB2": rqc_ctrb1_per_ctrb2,
            "rQC_CTRB1_PER_CTRB2_low_standard": RQC_CTRB1_PER_CTRB2_LOW_STANDARD,
            "rQC_CTRB1_PER_CTRB2_high_standard": RQC_CTRB1_PER_CTRB2_HIGH_STANDARD,
            "rQC_CTRB2": rqc_ctrb2,
            "rQC_CTRB2_low_standard": RQC_CTRB2_LOW_STANDARD,
            "rQC_CTRB2_high_standard": RQC_CTRB2_HIGH_STANDARD,
            "pQC_CTRB1_PER_CTRB2": pqc_ctrb1_per_ctrb2,
            "pQC_CTRB1_PER_CTRB2_low_standard": PQC_CTRB1_PER_CTRB2_LOW_STANDARD,
            "pQC_CTRB1_PER_CTRB2_high_standard": PQC_CTRB1_PER_CTRB2_HIGH_STANDARD,
            "pQC_CTRB2": pqc_ctrb2,
            "pQC_CTRB2_low_standard": PQC_CTRB2_LOW_STANDARD,
            "pQC_CTRB2_high_standard": PQC_CTRB2_HIGH_STANDARD,
            "QC_result": qc_result,
            "image_method": fields_image_analysis_method,
            "analysis_time": analysis_time,
            "analysis_method": script_dir,
        }
    )

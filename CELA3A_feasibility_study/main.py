import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from matplotlib import rcParams
import pandas as pd
import numpy as np
from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs
from analysisrun.interactive import scan_model_input, FilePath
from analysisrun.helper import read_dict
from analysisrun.cleansing import filter_by_entity
from pydantic import BaseModel, Field

from clustering import (
    cela3a_filtering,
    scatter_image_output,
    calculate_cela3a,
    calculate_required_number_of_measurements,
)
from qc_conductor import lane_qc_conductor, QC_sample_qc_conductor
from cela3a_output import (
    result,
)


class Input(BaseModel):
    input_csv: FilePath = Field(
        description="画像解析結果CSVファイル（ファイルのあるフォルダーに解析結果が出力されます）",
    )
    filter_csv: FilePath = Field(
        description="フィルター用のCSVファイル",
    )
    sample_csv: FilePath = Field(
        description="サンプル名CSVファイル（サンプル名とレーン番号の対応表）",
    )


@dataclass
class Context:
    sample_names: dict[str, str]


def analyze(args: AnalyzeArgs[Context]) -> pd.Series:
    rcParams["font.family"] = "Arial"
    rcParams["axes.linewidth"] = 2.0

    analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script_dir = os.path.basename(
        os.path.dirname(os.path.abspath(__file__))
    )  # スクリプトの入っているフォルダ名が現在のスクリプトのバージョンを反映

    ctx, fields, output = args.ctx, args.fields, args.output
    sample = ctx.sample_names.get(fields.data_name, fields.data_name)

    well_filtered_fields = [v[v["All_well_count"] > 5000] for v in fields]
    sufficient_well_fields = [v for v in well_filtered_fields if len(v) > 0]

    total_wells = np.sum([v["All_well_count"].iat[0] for v in sufficient_well_fields])

    # TODO: sufficient_well_fieldsの要素数が0の場合例外が発生する。
    if len(sufficient_well_fields) == 0:  # 当該データがない場合の処理
        return result(
            fields_data_name=fields.data_name,
            sample=sample,
            total_wells=total_wells,
            cela3a=None,
            cela3a_unit=None,
            required_number_of_measurements=None,
            lane_result="NG",
            lambda_cv=None,
            lambda_cv_result=None,
            insufficient_well_field_count=12,
            insufficient_well_field_count_result="NG",
            green_cv=None,
            green_cv_result=None,
            red_cv=None,
            red_cv_result=None,
            total_lambda=None,
            total_lambda_result=None,
            rqc_lambda=None,
            pqc_high_red_intensity_lambda=None,
            qc_result=None,
            fields_image_analysis_method=fields.image_analysis_method,
            analysis_time=analysis_time,
            script_dir=script_dir,
        )

    df_new = pd.concat(sufficient_well_fields)
    df_filter = args.data_for_enhancement[0].data

    total_lambda = len(df_new) / total_wells
    green = (df_new.FITC_Sum - (df_new.FITC_Max + df_new.FITC_Min)).to_numpy()
    red = (df_new.mCherry_Sum - (df_new.mCherry_Max + df_new.mCherry_Min)).to_numpy()

    df_cela3a, green_cela3a, red_cela3a = cela3a_filtering(df_new, df_filter)
    scatter_image_output(green_cela3a, red_cela3a, fields, sample, output)
    cela3a, cela3a_unit = calculate_cela3a(green_cela3a, total_wells)
    required_number_of_measurements = calculate_required_number_of_measurements(
        green_cela3a, total_wells
    )

    (
        lane_result,
        lambda_cv,
        lambda_cv_result,
        insufficient_well_field_count,
        insufficient_well_field_count_result,
        green_cv,
        green_cv_result,
        red_cv,
        red_cv_result,
        total_lambda_result,
    ) = lane_qc_conductor(
        fields,
        well_filtered_fields,
        green_cela3a,
        red_cela3a,
        green,
        red,
        total_lambda,
        output,
    )
    rqc_lambda, pqc_high_red_intensity_lambda, qc_result = QC_sample_qc_conductor(
        sample, total_lambda, red, total_wells
    )

    return result(
        fields_data_name=fields.data_name,
        sample=sample,
        total_wells=total_wells,
        cela3a=cela3a,
        cela3a_unit=cela3a_unit,
        required_number_of_measurements=required_number_of_measurements,
        lane_result=lane_result,
        lambda_cv=float(lambda_cv),
        lambda_cv_result=lambda_cv_result,
        insufficient_well_field_count=insufficient_well_field_count,
        insufficient_well_field_count_result=insufficient_well_field_count_result,
        green_cv=float(green_cv),
        green_cv_result=green_cv_result,
        red_cv=float(red_cv),
        red_cv_result=red_cv_result,
        total_lambda=total_lambda,
        total_lambda_result=total_lambda_result,
        rqc_lambda=rqc_lambda,
        pqc_high_red_intensity_lambda=pqc_high_red_intensity_lambda,
        qc_result=qc_result,
        fields_image_analysis_method=fields.image_analysis_method,
        analysis_time=analysis_time,
        script_dir=script_dir,
    )


def postprocess(args: PostprocessArgs[Context]) -> pd.DataFrame:
    analysis_results = args.analysis_results.copy()

    # デフォルトはOKで初期化
    analysis_results["result"] = "OK"

    # "sample"が"rQC", "pQCの行を抽出
    rqc_rows = analysis_results[analysis_results["sample_name"] == "rQC"]
    pqc_rows = analysis_results[analysis_results["sample_name"] == "pQC"]

    # まず「rQCかpQCのどちらかが存在しない」条件を判定
    if rqc_rows.empty or pqc_rows.empty:
        analysis_results["result"] = "disc_NG"

    # 次に「rQCとpQCが両方ともQC_result == NG」の条件
    elif (
        (rqc_rows["QC_result"] == "NG").all() or (rqc_rows["lane_result"] == "NG").all()
    ) and (
        (pqc_rows["QC_result"] == "NG").all() or (pqc_rows["lane_result"] == "NG").all()
    ):
        analysis_results["result"] = "disc_NG"

    else:
        # disc_NGでない行のみlane_resultをチェック
        lane_ng_mask = analysis_results["lane_result"] == "NG"
        analysis_results.loc[lane_ng_mask, "result"] = "lane_NG"

        lane_check_mask = analysis_results["lane_result"] == "check"
        analysis_results.loc[lane_check_mask, "result"] = "check"

        # rQC または pQC の行で QC_result が NG の場合は lane_NG にする
        qc_mask = analysis_results["sample_name"].isin(["rQC", "pQC"])
        qc_ng_mask = analysis_results["QC_result"] == "NG"
        combined_mask = qc_mask & qc_ng_mask
        analysis_results.loc[combined_mask, "result"] = "lane_NG"

    return analysis_results


def main():
    _in = scan_model_input(Input)

    input_csv_path = Path(_in.input_csv)
    df = pd.read_csv(input_csv_path)
    filter_csv_path = Path(_in.filter_csv)
    df_filter = pd.read_csv(filter_csv_path)
    sample_names = read_dict(_in.sample_csv, "data", "sample")
    os.chdir(
        input_csv_path.parent
    )  # 解析対象となる画像解析結果データが存在するディレクトリを作業ディレクトリとする。

    # 解析のコンテキスト（パラメータ）を組み立てる
    result = ParallelRunner(analyze, postprocess).run(
        ctx=Context(sample_names=sample_names),
        whole_data=filter_by_entity(df, entity="Activity Spots"),
        data_for_enhancement=[filter_by_entity(df_filter, entity="Surrounding Spots")],
        target_data=list(sample_names),
    )

    disc = input_csv_path.stem
    result.to_csv(f"{disc}_result.csv")
    df_NG = result[result.lane_result == "NG"]
    df_NG = df_NG.loc[:, ["data", "sample_name", "lane_result"]]
    df_NG.to_csv(f"{disc}_NG_list.csv")


if __name__ == "__main__":
    main()

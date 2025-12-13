import os
from datetime import datetime

from matplotlib import rcParams
import pandas as pd
import numpy as np
from pathlib import Path
from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs
from analysisrun.interactive import scan_model_input, FilePath
from analysisrun.helper import read_dict
from analysisrun.cleansing import filter_by_entity
from dataclasses import dataclass
from pydantic import BaseModel, Field

from qc_conductor import lane_qc_conductor, QC_sample_qc_conductor
from clustering import clustering_assign, scatter_image_output, paramater_calculator
from chymotrypsin_output import (
    result,
    RED_MEAN_INTENSITY_STANDARD,
    TOTAL_LAMBDA_LOW_STANDARD,
    TOTAL_LAMBDA_HIGH_STANDARD,
    MASTER_CSV_BASENAME,
)


class Input(BaseModel):
    input_csv: FilePath = Field(
        description="画像解析結果CSVファイル（ファイルのあるフォルダーに解析結果が出力されます）",
    )
    sample_csv: FilePath = Field(
        description="サンプル名CSVファイル（サンプル名とレーン番号の対応表）",
    )


@dataclass
class Context:
    sample_names: dict[str, str]
    master_csv: pd.DataFrame


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

    if len(sufficient_well_fields) == 0:  # 当該データがない場合の処理
        return result(
            fields_data_name=fields.data_name,
            sample=sample,
            total_wells=total_wells,
            cluster_1=None,
            cluster_2=None,
            cluster_3=None,
            ctrb2_unit=None,
            ctrb1_per_ctrb2=None,
            lane_result="NG",
            lambda_cv=None,
            lambda_cv_result=None,
            insufficient_well_field_count=12,
            insufficient_well_field_count_result="NG",
            green_cv=None,
            green_cv_result=None,
            red_cv=None,
            red_cv_result=None,
            green_mean=None,
            green_mean_result=None,
            red_mean=None,
            red_mean_standard=RED_MEAN_INTENSITY_STANDARD,
            red_mean_result=None,
            total_lambda=None,
            total_lambda_result=None,
            total_lambda_low_standard=TOTAL_LAMBDA_LOW_STANDARD,
            total_lambda_high_standard=TOTAL_LAMBDA_HIGH_STANDARD,
            rqc_ctrb1_per_ctrb2=None,
            rqc_ctrb2=None,
            pqc_ctrb1_per_ctrb2=None,
            pqc_ctrb2=None,
            qc_result=None,
            fields_image_analysis_method=fields.image_analysis_method,
            analysis_time=analysis_time,
            script_dir=script_dir,
        )

    df_new = pd.concat(sufficient_well_fields)
    total_lambda = len(df_new) / total_wells

    df_labeled_sample, df_cluster, new_labels = clustering_assign(
        df_new, ctx.master_csv
    )
    scatter_image_output(df_labeled_sample, fields, sample, output)
    cluster_1, cluster_2, cluster_3, ctrb2_unit, ctrb1_per_ctrb2 = paramater_calculator(
        df_labeled_sample, total_wells
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
        green_mean,
        green_mean_result,
        red_mean,
        red_mean_standard,
        red_mean_result,
        total_lambda_low_standard,
        total_lambda_high_standard,
        total_lambda_result,
    ) = lane_qc_conductor(
        fields,
        well_filtered_fields,
        df_labeled_sample,
        sample,
        new_labels,
        df_cluster,
        total_lambda,
        output,
    )
    rqc_ctrb1_per_ctrb2, rqc_ctrb2, pqc_ctrb1_per_ctrb2, pqc_ctrb2, qc_result = (
        QC_sample_qc_conductor(sample, ctrb1_per_ctrb2, ctrb2_unit)
    )

    return result(
        fields_data_name=fields.data_name,
        sample=sample,
        total_wells=total_wells,
        cluster_1=cluster_1,
        cluster_2=cluster_2,
        cluster_3=cluster_3,
        ctrb2_unit=ctrb2_unit,
        ctrb1_per_ctrb2=ctrb1_per_ctrb2,
        lane_result=lane_result,
        lambda_cv=float(lambda_cv),
        lambda_cv_result=lambda_cv_result,
        insufficient_well_field_count=insufficient_well_field_count,
        insufficient_well_field_count_result=insufficient_well_field_count_result,
        green_cv=float(green_cv),
        green_cv_result=green_cv_result,
        red_cv=float(red_cv),
        red_cv_result=red_cv_result,
        green_mean=green_mean,
        green_mean_result=green_mean_result,
        red_mean=red_mean,
        red_mean_standard=red_mean_standard,
        red_mean_result=red_mean_result,
        total_lambda=total_lambda,
        total_lambda_low_standard=total_lambda_low_standard,
        total_lambda_high_standard=total_lambda_high_standard,
        total_lambda_result=total_lambda_result,
        rqc_ctrb1_per_ctrb2=rqc_ctrb1_per_ctrb2,
        rqc_ctrb2=rqc_ctrb2,
        pqc_ctrb1_per_ctrb2=pqc_ctrb1_per_ctrb2,
        pqc_ctrb2=pqc_ctrb2,
        qc_result=qc_result,
        fields_image_analysis_method=fields.image_analysis_method,
        analysis_time=analysis_time,
        script_dir=script_dir,
    )


def postprocess(args: PostprocessArgs[Context]) -> pd.DataFrame:
    analysis_results = args.analysis_results.copy()

    # デフォルトはOKで初期化
    analysis_results["result"] = "OK"

    # "sample"が"QC"の行を抽出
    rqc_rows = analysis_results[analysis_results["sample_name"] == "rQC"]
    pqc_rows = analysis_results[analysis_results["sample_name"] == "pQC"]

    # まず「rQCかpQCのどちらかが存在しない」条件を判定
    if rqc_rows.empty or pqc_rows.empty:
        analysis_results["result"] = "disc_NG"

    # 次に「rQCとpQCが両方ともQC_result == NG」の条件
    elif (
        (rqc_rows["QC_result"] == "NG").all()
        or (rqc_rows["lane_result"] == "NG").all()
        or (rqc_rows["lane_result"] == "check").all()
    ) and (
        (pqc_rows["QC_result"] == "NG").all()
        or (pqc_rows["lane_result"] == "NG").all()
        or (pqc_rows["lane_result"] == "check").all()
    ):
        analysis_results["result"] = "disc_NG"

    else:
        # disc_NGでない行のみlane_resultをチェック
        lane_ng_mask = analysis_results["lane_result"] == "NG"
        analysis_results.loc[lane_ng_mask, "result"] = "lane_NG"

        lane_check_mask = analysis_results["lane_result"] == "check"
        analysis_results.loc[lane_check_mask, "result"] = "check"

    return analysis_results


def main():
    _in = scan_model_input(Input)

    input_csv_path = Path(_in.input_csv)
    df = pd.read_csv(input_csv_path)
    sample_names = read_dict(_in.sample_csv, "data", "sample")

    # === ここを修正：この .py と同じフォルダにある master.csv を読む ===
    script_dir = Path(__file__).resolve().parent
    master_csv_path = script_dir / MASTER_CSV_BASENAME
    if not master_csv_path.exists():
        raise FileNotFoundError(
            f"Master CSV not found: {master_csv_path}\n"
            "同じフォルダに 'master.csv' を置くか、MASTER_CSV_BASENAME を変更してください。"
        )
    master_csv = pd.read_csv(master_csv_path)

    os.chdir(
        input_csv_path.parent
    )  # 解析対象となる画像解析結果データが存在するディレクトリを作業ディレクトリとする。

    # 解析のコンテキスト（パラメータ）を組み立てる

    result = ParallelRunner(analyze, postprocess).run(
        ctx=Context(sample_names=sample_names, master_csv=master_csv),
        whole_data=filter_by_entity(df, entity="Activity Spots"),
        target_data=list(sample_names),
    )

    disc = input_csv_path.stem
    result.to_csv(f"{disc}_result.csv")
    df_NG = result[result.lane_result == "NG"]
    df_NG = df_NG.loc[:, ["data", "sample_name", "lane_result"]]
    df_NG.to_csv(f"{disc}_NG_list.csv")


if __name__ == "__main__":
    main()

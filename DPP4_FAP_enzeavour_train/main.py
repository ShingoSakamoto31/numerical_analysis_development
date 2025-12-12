import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from analysisrun.cleansing import filter_by_entity
from analysisrun.helper import read_dict
from analysisrun.interactive import FilePath, scan_model_input
from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs
from matplotlib import rcParams
from pydantic import BaseModel, Field

from dpp4_fap_output import (
    result,
)
from qc_conductor import QC_sample_qc_conductor, lane_qc_conductor
from scatter_fitting import paramater_calculator, scatter_fitting, scatter_image_output


def get_user_paths():
    """ユーザーに2つのパスの入力を要求"""
    print("=" * 50)
    print("パス入力")
    print("=" * 50)
    
    path1 = input("1つ目のパスを入力してください (画像解析結果CSVファイル): ").strip()
    if not path1:
        raise ValueError("1つ目のパスが入力されていません")
    
    path2 = input("2つ目のパスを入力してください (サンプル名CSVファイル): ").strip()
    if not path2:
        raise ValueError("2つ目のパスが入力されていません")
    
    print("=" * 50)
    
    return Path(path1), Path(path2)


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

    if len(sufficient_well_fields) == 0:  # 当該データがない場合の処理
        return result(
            fields_data_name=fields.data_name,
            sample=sample,
            total_wells=total_wells,
            dpp4_homodimer=None,
            dpp4_fap_heterodimer=None,
            fap_homodimer=None,
            dpp4_homodimer_ratio=None,
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
            total_lambda=None,
            total_lambda_result=None,
            qc_fap=None,
            qc_dpp4_homodimer_ratio=None,
            qc_result=None,
            fields_image_analysis_method=fields.image_analysis_method,
            analysis_time=analysis_time,
            script_dir=script_dir,
        )

    df_new = pd.concat(sufficient_well_fields)
    total_lambda = len(df_new) / total_wells

    green = (df_new.FITC_Sum - (df_new.FITC_Max + df_new.FITC_Min)).to_numpy()
    red = (df_new.mCherry_Sum - (df_new.mCherry_Max + df_new.mCherry_Min)).to_numpy()

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
        total_lambda_result,
    ) = lane_qc_conductor(
        fields, well_filtered_fields, green, red, total_lambda, output
    )

    if lane_result == "NG" or sample == "blank":
        return result(
            fields_data_name=fields.data_name,
            sample=sample,
            total_wells=total_wells,
            dpp4_homodimer=None,
            dpp4_fap_heterodimer=None,
            fap_homodimer=None,
            dpp4_homodimer_ratio=None,
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
            total_lambda=None,
            total_lambda_result=None,
            qc_fap=None,
            qc_dpp4_homodimer_ratio=None,
            qc_result=None,
            fields_image_analysis_method=fields.image_analysis_method,
            analysis_time=analysis_time,
            script_dir=script_dir,
        )

    g1, g2, g3, r1, r2, r3, r4, r5 = scatter_fitting(green, red, sample, fields, output)
    scatter_image_output(
        green, red, g1, g2, g3, r1, r2, r3, r4, r5, sample, fields, output
    )
    dpp4_homodimer, dpp4_fap_heterodimer, fap_homodimer, dpp4_homodimer_ratio = (
        paramater_calculator(green, red, g1, g2, g3, r1, r3, r5, total_wells)
    )
    qc_fap, qc_dpp4_homodimer_ratio, qc_result = QC_sample_qc_conductor(
        sample, fap_homodimer, dpp4_homodimer_ratio
    )

    return result(
        fields_data_name=fields.data_name,
        sample=sample,
        total_wells=total_wells,
        dpp4_homodimer=dpp4_homodimer,
        dpp4_fap_heterodimer=dpp4_fap_heterodimer,
        fap_homodimer=fap_homodimer,
        dpp4_homodimer_ratio=dpp4_homodimer_ratio,
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
        total_lambda=total_lambda,
        total_lambda_result=total_lambda_result,
        qc_fap=qc_fap,
        qc_dpp4_homodimer_ratio=qc_dpp4_homodimer_ratio,
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
    qc_rows = analysis_results[analysis_results["sample_name"] == "QC"]

    # "QC_result"がNGか、QC行が存在しない場合は全体をdisc_NGに
    if (
        qc_rows.empty
        or (qc_rows["QC_result"] == "NG").any()
        or (qc_rows["lane_result"] == "NG").any()
        or (qc_rows["lane_result"] == "check").any()
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
    input_csv_path, sample_csv_path = get_user_paths()
    
    # パスが存在するか確認
    if not input_csv_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {input_csv_path}")
    if not sample_csv_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {sample_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    sample_names = read_dict(str(sample_csv_path), "data", "sample")
    os.chdir(
        input_csv_path.parent
    )  # 解析対象となる画像解析結果データが存在するディレクトリを作業ディレクトリとする。

    # 解析のコンテキスト（パラメータ）を組み立てる
    result = ParallelRunner(analyze, postprocess).run(
        ctx=Context(sample_names=sample_names),
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

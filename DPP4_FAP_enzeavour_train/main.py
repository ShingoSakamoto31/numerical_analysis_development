import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib import rcParams

from analysisrun.cleansing import filter_by_entity
from analysisrun.helper import read_dict
from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs

from dpp4_fap_output import result
from qc_conductor import QC_sample_qc_conductor, lane_qc_conductor
from scatter_fitting import paramater_calculator, scatter_fitting, scatter_image_output


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


def get_user_paths():
    """ユーザーに2つのパスの入力を要求"""
    print("=" * 50)
    print("パス入力")
    print("=" * 50)

    path1 = (
        input("1つ目のパスを入力してください (画像解析結果CSVの格納フォルダ): ")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not path1:
        raise ValueError("1つ目のパスが入力されていません")

    path2 = (
        input("2つ目のパスを入力してください (結果を保存するフォルダ): ")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not path2:
        raise ValueError("2つ目のパスが入力されていません")

    print("=" * 50)

    return Path(path1), Path(path2)


def copy_path_contents(source_path: Path, destination_base_path: Path) -> Path:
    """1つ目のパスの中身を、同じフォルダ名で2つ目のパスの中にコピー"""
    source_path = Path(source_path)
    destination_base_path = Path(destination_base_path)

    # ソースパスが存在するか確認
    if not source_path.exists():
        raise FileNotFoundError(f"ソースパスが見つかりません: {source_path}")

    # コピー先の親ディレクトリが存在するか確認
    if not destination_base_path.exists():
        raise FileNotFoundError(
            f"コピー先の親ディレクトリが見つかりません: {destination_base_path}"
        )

    # ソースパスがファイルか、ディレクトリか判定
    if source_path.is_file():
        # ファイルの場合
        destination_path = destination_base_path / source_path.name
        print(f"ファイルをコピー中: {source_path} -> {destination_path}")
        shutil.copy2(source_path, destination_path)
        return destination_path

    elif source_path.is_dir():
        # ディレクトリの場合は、同じ名前のフォルダを作成してコピー
        destination_path = destination_base_path / source_path.name
        if destination_path.exists():
            print(f"警告: コピー先が既に存在します: {destination_path}")
            response = input("上書きしますか? (y/n): ").strip().lower()
            if response == "y":
                shutil.rmtree(destination_path)
            else:
                print("コピーをキャンセルしました")
                return destination_path

        print(f"ディレクトリをコピー中: {source_path} -> {destination_path}")
        shutil.copytree(source_path, destination_path)
        print(f"コピー完了: {destination_path}")
        return destination_path

    else:
        raise ValueError(
            f"ソースパスがファイルでもディレクトリでもありません: {source_path}"
        )


def process_copied_folder(copied_folder_path: Path) -> None:
    """
    コピー先フォルダのサブフォルダについて、
    フォルダ名と同じ名前のCSVファイルを見つけ、
    そのCSVファイルの行数がタイトルの、空のテキストファイルを各サブフォルダの中に作成
    """
    copied_folder_path = Path(copied_folder_path)

    if not copied_folder_path.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {copied_folder_path}")

    if not copied_folder_path.is_dir():
        raise ValueError(f"フォルダではありません: {copied_folder_path}")

    # サブフォルダを処理
    subfolders = [p for p in copied_folder_path.iterdir() if p.is_dir()]
    total = len(subfolders)

    def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 40) -> None:
        """シンプルなコンソール進捗バーを表示するヘルパー
        iteration: 現在のインデックス (1-based)
        total: 合計数
        prefix/suffix: 表示用テキスト
        length: バーの長さ
        """
        if total == 0:
            return
        percent = iteration / total
        filled_length = int(length * percent)
        bar = '█' * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total} {int(percent*100)}% {suffix}")
        sys.stdout.flush()

    for idx, subfolder in enumerate(subfolders, start=1):
        subfolder_name = subfolder.name
        print_progress_bar(idx, total, prefix='Processing', suffix=subfolder_name)

        # フォルダと同じ名前のCSVファイルを探す
        input_csv_path = subfolder / f"{subfolder_name}.csv"
        sample_csv_path = subfolder / f"{subfolder_name}_sample.csv"

        if not input_csv_path.exists():
            print(f"\n警告: CSVファイルが見つかりません: {input_csv_path}")
            continue

        if not sample_csv_path.exists():
            print(f"\n警告: サンプルCSVファイルが見つかりません: {sample_csv_path}")
            continue

        try:
            df = pd.read_csv(input_csv_path)
            # read_dict は文字列パスを期待することが多いため明示的に str() を渡す
            sample_names = read_dict(str(sample_csv_path), "data", "sample")
            os.chdir(input_csv_path.parent)  # 必要に応じて作業ディレクトリを変更

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

        except Exception as e:
            print(f"\nエラー: {input_csv_path}の処理中にエラーが発生しました: {e}")

    # 完了表示
    if total > 0:
        print_progress_bar(total, total, prefix='Processing', suffix='done')
        print()  # 改行


if __name__ == "__main__":
    # 対話でパスを取得 -> コピー -> コピー先を処理
    p1, p2 = get_user_paths()
    copied_path = copy_path_contents(p1, p2)

    if copied_path.is_dir():
        process_copied_folder(copied_path)
    else:
        print(f"コピー先はファイルです。処理対象のフォルダが必要です: {copied_path}")

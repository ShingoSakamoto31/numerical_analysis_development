import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from matplotlib import rcParams

from analysisrun.cleansing import filter_by_entity
from analysisrun.helper import read_dict
from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs

from qc_conductor import lane_qc_conductor, QC_sample_qc_conductor
from clustering import clustering_assign, scatter_image_output, paramater_calculator
from chymotrypsin_output import (
    result,
    RED_MEAN_INTENSITY_STANDARD,
    TOTAL_LAMBDA_LOW_STANDARD,
    TOTAL_LAMBDA_HIGH_STANDARD,
    MASTER_CSV_BASENAME,
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

    def print_progress_bar(
        iteration: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 40,
        elapsed_time: float = 0.0,
    ) -> None:
        """シンプルなコンソール進捗バーを表示するヘルパー
        iteration: 現在のインデックス (1-based)
        total: 合計数
        prefix/suffix: 表示用テキスト
        length: バーの長さ
        elapsed_time: 経過時間（秒）
        """
        if total == 0:
            return
        percent = iteration / total
        filled_length = int(length * percent)
        bar = "█" * filled_length + "-" * (length - filled_length)

        # 経過時間をHH:MM:SS形式に変換
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        sys.stdout.write(
            f"\r{prefix} |{bar}| {iteration}/{total} {int(percent * 100)}% [{time_str}] {suffix}"
        )
        sys.stdout.flush()

    start_time = time.time()

    for idx, subfolder in enumerate(subfolders, start=1):
        subfolder_name = subfolder.name
        elapsed_time = time.time() - start_time
        print_progress_bar(
            idx,
            total,
            prefix="Processing",
            suffix=subfolder_name,
            elapsed_time=elapsed_time,
        )

        # フォルダ名と同じCSVファイルを探す
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
            sample_names = read_dict(str(sample_csv_path), "data", "sample")

            # この.pyと同じフォルダにあるmaster.csvを読む
            script_dir = Path(__file__).resolve().parent
            master_csv_path = script_dir / MASTER_CSV_BASENAME
            if not master_csv_path.exists():
                raise FileNotFoundError(
                    f"Master CSV not found: {master_csv_path}\n"
                    "同じフォルダに 'master.csv' を置くか、MASTER_CSV_BASENAME を変更してください。"
                )
            master_csv = pd.read_csv(master_csv_path)
            os.chdir(input_csv_path.parent)  # 必要に応じて作業ディレクトリを変更

            # 解析のコンテキスト（パラメータ）を組み立てる
            result = ParallelRunner(analyze, postprocess).run(
                ctx=Context(sample_names=sample_names, master_csv=master_csv),
                whole_data=filter_by_entity(df, entity="Activity Spots"),
                target_data=list(sample_names),
            )

            disc = input_csv_path.stem
            result.to_csv(subfolder / f"{disc}_result.csv")
            df_NG = result[result.lane_result == "NG"]
            df_NG = df_NG.loc[:, ["data", "sample_name", "lane_result"]]
            df_NG.to_csv(subfolder / f"{disc}_NG_list.csv")

        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            print(traceback.format_exc())

    # 完了表示
    if total > 0:
        elapsed_time = time.time() - start_time
        print_progress_bar(
            total,
            total,
            prefix="Processing",
            suffix="Completed",
            elapsed_time=elapsed_time,
        )
        print()  # 改行


if __name__ == "__main__":
    # 対話でパスを取得 -> コピー -> コピー先フォルダを処理
    p1, p2 = get_user_paths()
    copied_path = copy_path_contents(p1, p2)

    if copied_path.is_dir():
        process_copied_folder(copied_path)
    else:
        print(f"コピー先はファイルです。処理対象のフォルダが必要です: {copied_path}")

from pathlib import Path
import pandas as pd
import shutil


def get_user_paths():
    """ユーザーにパスの入力を要求"""
    print("=" * 50)
    print("パス入力")
    print("=" * 50)

    path = (
        input("パスを入力してください (処理対象のフォルダ): ")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not path:
        raise ValueError("パスが入力されていません")

    print("=" * 50)

    return Path(path)


def get_result_paths():
    """ユーザーに2つのパスの入力を要求（CSV統合用）"""
    print("=" * 50)
    print("パス入力")
    print("=" * 50)

    path1 = (
        input("1つ目のパスを入力してください (CSVが格納されているフォルダ): ")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not path1:
        raise ValueError("1つ目のパスが入力されていません")

    path2 = (
        input("2つ目のパスを入力してください (統合結果を保存するフォルダ): ")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not path2:
        raise ValueError("2つ目のパスが入力されていません")

    print("=" * 50)

    return Path(path1), Path(path2)


def find_result_csv_files(root_path: Path) -> list[Path]:
    """'_result'で終わるCSVファイルをすべて見つける（サブフォルダも含む）

    Args:
        root_path: 検索対象のルートパス

    Returns:
        見つかったCSVファイルのパスリスト
    """
    if not root_path.exists():
        raise FileNotFoundError(f"パスが見つかりません: {root_path}")

    if not root_path.is_dir():
        raise ValueError(f"ディレクトリではありません: {root_path}")

    result_files = list(root_path.glob("**/*_result.csv"))
    return sorted(result_files)


def merge_csv_files(
    csv_files: list[Path], output_path: Path, output_filename: str = "merged_result.csv"
) -> Path:
    """複数のCSVファイルを統合して1つのファイルに出力

    Args:
        csv_files: 統合するCSVファイルのパスリスト
        output_path: 出力先ディレクトリ
        output_filename: 出力ファイル名

    Returns:
        出力されたファイルのパス
    """
    if not csv_files:
        raise ValueError("統合するCSVファイルが見つかりません")

    # 出力ディレクトリが存在しない場合は作成
    output_path.mkdir(parents=True, exist_ok=True)

    # CSVファイルを読み込んで統合
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"✓ {csv_file.name} を読み込みました")
        except Exception as e:
            print(f"✗ {csv_file.name} の読み込みに失敗しました: {e}")

    if not dfs:
        raise ValueError("有効なCSVファイルが見つかりません")

    # DataFrameを統合
    merged_df = pd.concat(dfs, ignore_index=True)

    # 出力
    output_file = output_path / output_filename
    merged_df.to_csv(output_file, index=False)

    return output_file


def find_scatter_png_files(root_path: Path) -> list[Path]:
    """'_scatter'で終わるPNGファイルをすべて見つける（サブフォルダも含む）

    Args:
        root_path: 検索対象のルートパス

    Returns:
        見つかったPNGファイルのパスリスト
    """
    if not root_path.exists():
        raise FileNotFoundError(f"パスが見つかりません: {root_path}")

    if not root_path.is_dir():
        raise ValueError(f"ディレクトリではありません: {root_path}")

    scatter_files = list(root_path.glob("**/*_scatter.png"))
    return sorted(scatter_files)


def copy_scatter_files(
    png_files: list[Path], output_path: Path, folder_name: str = "scatters"
) -> Path:
    """複数のPNGファイルをコピーして指定フォルダに格納

    Args:
        png_files: コピーするPNGファイルのパスリスト
        output_path: 出力先ディレクトリ
        folder_name: 作成するフォルダ名

    Returns:
        ファイルがコピーされたフォルダのパス
    """
    if not png_files:
        raise ValueError("コピーするPNGファイルが見つかりません")

    # 出力フォルダを作成
    scatter_folder = output_path / folder_name
    scatter_folder.mkdir(parents=True, exist_ok=True)

    # PNGファイルをコピー
    for png_file in png_files:
        try:
            dest_file = scatter_folder / png_file.name
            shutil.copy2(png_file, dest_file)
            print(f"✓ {png_file.name} をコピーしました")
        except Exception as e:
            print(f"✗ {png_file.name} のコピーに失敗しました: {e}")

    return scatter_folder


if __name__ == "__main__":
    try:
        # パス入力
        source_path, output_path = get_result_paths()
        
        print("\n" + "=" * 50)
        print("処理を開始します")
        print("=" * 50 + "\n")
        
        # CSV統合処理
        print("【CSV統合処理】")
        csv_files = find_result_csv_files(source_path)
        print(f"\n見つかったCSVファイル: {len(csv_files)} 個\n")
        
        if csv_files:
            merged_csv = merge_csv_files(csv_files, output_path)
            print(f"\n✓ CSV統合完了: {merged_csv}\n")
        else:
            print("※ '_result' で終わるCSVファイルが見つかりません\n")
        
        # PNG画像コピー処理
        print("【PNG画像コピー処理】")
        png_files = find_scatter_png_files(source_path)
        print(f"\n見つかったPNGファイル: {len(png_files)} 個\n")
        
        if png_files:
            scatter_folder = copy_scatter_files(png_files, output_path)
            print(f"\n✓ PNG画像コピー完了: {scatter_folder}\n")
        else:
            print("※ '_scatter' で終わるPNGファイルが見つかりません\n")
        
        print("=" * 50)
        print("すべての処理が完了しました")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

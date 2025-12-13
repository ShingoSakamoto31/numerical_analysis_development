import shutil
import pandas as pd
from pathlib import Path


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


def create_text_files_from_csv(copied_folder_path: Path) -> None:
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
    for subfolder in copied_folder_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        subfolder_name = subfolder.name
        
        # フォルダと同じ名前のCSVファイルを探す
        csv_file = subfolder / f"{subfolder_name}.csv"
        
        if not csv_file.exists():
            print(f"警告: CSVファイルが見つかりません: {csv_file}")
            continue
        
        try:
            # CSVファイルを読み込んで行数を取得
            df = pd.read_csv(csv_file)
            csv_length = len(df)
            
            # 空のテキストファイルを作成（ファイル名は行数）
            text_file_name = f"{csv_length}.txt"
            text_file_path = subfolder / text_file_name
            
            # ファイルを作成
            text_file_path.touch()
            print(f"作成: {text_file_path} (CSVの行数: {csv_length})")
            
        except Exception as e:
            print(f"エラー: {csv_file}の処理中にエラーが発生しました: {e}")


if __name__ == "__main__":
    p1, p2 = get_user_paths()
    print("path1 =", p1)
    print("path2 =", p2)
    copied_path = copy_path_contents(p1, p2)
    print("\nテキストファイル作成処理を開始します...")
    create_text_files_from_csv(copied_path)

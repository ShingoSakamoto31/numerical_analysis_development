import numpy as np
from scipy.ndimage import gaussian_filter1d


def make(data, num_bins=200, sigma=2):
    # 対数スケールでビンのエッジを設定
    bin_edges = np.logspace(
        np.log10(data.min()), np.log10(np.percentile(data, 99.9)), num=num_bins + 1
    )

    # ヒストグラムを計算
    hist, bin_edges = np.histogram(
        data, bins=bin_edges, range=(0, np.percentile(data, 99.9))
    )

    # ビンの中心を計算
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # ガウシアンフィルタを使ってスムージング
    hist_smoothed = gaussian_filter1d(hist, sigma)

    return bin_centers, hist_smoothed


def visualize(data, num_bins=200, sigma=2):
    # 対数スケールでビンのエッジを設定
    bin_edges = np.logspace(
        np.log10(data.min()), np.log10(data.max()), num=num_bins + 1
    )

    # ヒストグラムを計算
    hist, bin_edges = np.histogram(
        data, bins=bin_edges, range=(0, np.percentile(data, 99.9)), density=True
    )

    # ビンの中心を計算
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # ガウシアンフィルタを使ってスムージング
    hist_smoothed = gaussian_filter1d(hist, sigma)

    return bin_centers, hist_smoothed


if __name__ == "__main__":
    print("OK")

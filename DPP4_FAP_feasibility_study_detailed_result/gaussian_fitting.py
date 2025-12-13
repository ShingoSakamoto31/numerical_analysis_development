from typing import Optional

import pandas as pd
import numpy as np
import itertools
from scipy.optimize import least_squares
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm
import enhanced_histogram as eh


# 初期値の組み合わせ（for green）
def green_init():
    lm1 = [3000, 5000, 7000]  # １つ目の山と２つ目の山の間の極小値の候補
    sigma1 = [500, 1000, 1500]
    sigma2 = [1000, 2000, 3000]

    # すべての組み合わせを生成
    combinations = itertools.product(lm1, sigma1, sigma2)

    # DataFrameに格納
    df_init_g = pd.DataFrame(combinations, columns=["lm1", "sigma1", "sigma2"])
    df_init_g = df_init_g.sample(frac=1).reset_index(drop=True)

    return df_init_g


# 初期値の組み合わせ (for red, ２山)
def red_init_2():
    lm1 = [35000, 40000, 45000, 50000]  # １つ目の山と２つ目の山の間の極小値の候補
    sigma1 = [2000, 3000, 4000]
    sigma2 = [3000, 4000, 5000]

    # すべての組み合わせを生成
    combinations = itertools.product(lm1, sigma1, sigma2)

    # DataFrameに格納
    df_init_r2 = pd.DataFrame(combinations, columns=["lm1", "sigma1", "sigma2"])
    df_init_r2 = df_init_r2.sample(frac=1).reset_index(drop=True)

    return df_init_r2


# 初期値の組み合わせ (for red, ３山)
def red_init_3():
    lm1 = [35000, 40000, 45000, 50000]  # １つ目の山と２つ目の山の間の極小値の候補
    lm2 = [55000, 60000, 65000, 70000]  # ２つ目の山と３つ目の山の間の極小値の候補
    sigma1 = [2000, 3000, 4000]
    sigma2 = [3000, 4000, 5000]
    sigma3 = [3000, 4000, 5000]

    # すべての組み合わせを生成
    combinations = itertools.product(lm1, lm2, sigma1, sigma2, sigma3)

    # DataFrameに格納
    df_init_r3 = pd.DataFrame(
        combinations, columns=["lm1", "lm2", "sigma1", "sigma2", "sigma3"]
    )
    df_init_r3 = df_init_r3.sample(frac=1).reset_index(drop=True)

    return df_init_r3


# 正規分布フィッティングの初期値設定の関数（for １山)
def set_init_1(v):  # vが入力するデータ、lm1は１つ目の極小値候補
    x, y = eh.make(v)

    mu1_0 = x[np.argmax(y)]  # １つ目の山の中心値の初期値
    A1_0 = y[np.argmax(y)]  # １つ目の山の高さの初期値

    return mu1_0, A1_0


# 正規分布フィッティングの初期値設定の関数（for ２山）
def set_init_2(
    v, lm1
) -> Optional[
    tuple[float, float, float, float]
]:  # vが入力するデータ、lm1は１つ目の極小値候補
    x, y = eh.make(v)

    x1 = x[(x < lm1)]
    y1 = y[(x < lm1)]
    x2 = x[(x > lm1)]
    y2 = y[(x > lm1)]

    if all(arr.size for arr in (x1, y1, x2, y2)):
        mu1_0 = x1[np.argmax(y1)]  # １つ目の山の中心値の初期値
        A1_0 = y1[np.argmax(y1)]  # １つ目の山の高さの初期値
        mu2_0 = x2[np.argmax(y2)]
        A2_0 = y2[np.argmax(y2)]

    else:
        return None

    return mu1_0, A1_0, mu2_0, A2_0


# 正規分布フィッティングの初期値設定の関数（for ３山）
def set_init_3(
    v, lm1, lm2
) -> Optional[
    tuple[float, float, float, float, float, float]
]:  # vが入力するデータ、lm1は１つ目の極小値候補、lm2は２つ目の極小値候補
    x, y = eh.make(v)

    x1 = x[(x < lm1)]
    y1 = y[(x < lm1)]
    x2 = x[(x > lm1) & (x < lm2)]
    y2 = y[(x > lm1) & (x < lm2)]
    x3 = x[(x > lm2)]
    y3 = y[(x > lm2)]

    if all(arr.size for arr in (x1, y1, x2, y2, x3, y3)):
        mu1_0 = x1[np.argmax(y1)]  # １つ目の山の中心値の初期値
        A1_0 = y1[np.argmax(y1)]  # １つ目の山の高さの初期値
        mu2_0 = x2[np.argmax(y2)]
        A2_0 = y2[np.argmax(y2)]
        mu3_0 = x3[np.argmax(y3)]
        A3_0 = y3[np.argmax(y3)]

    else:
        return None

    return mu1_0, A1_0, mu2_0, A2_0, mu3_0, A3_0


# １山の正規分布の関数
def gaussian(x, mu1, sigma1, A1):
    return A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))


# ２山の正規分布の関数
def double_gaussian(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2)) + A2 * np.exp(
        -((x - mu2) ** 2) / (2 * sigma2**2)
    )


# ３山の正規分布の関数
def triple_gaussian(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    return (
        A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
        + A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
        + A3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3**2))
    )


# １つの正規分布へのフィッティング
def gaussian_fitting(data, x, y):
    # 目的関数
    def residuals(params, x, y):
        model = gaussian(x, *params)
        return y - model

    # フィッティング
    mu1_0, A1_0 = set_init_1(data)
    sigma_0 = np.std(data)
    initial_guess = [mu1_0, sigma_0, A1_0]
    res = least_squares(residuals, initial_guess, args=(x, y))
    popt = res.x
    fitted_y = gaussian(x, *popt)
    SSR = np.sum(((y - fitted_y) ** 2) * x)
    SSR = np.sqrt(SSR)

    return popt, SSR, fitted_y


# ２つの正規分布へのフィッティング
def double_gaussian_fitting(data, x, y, df_init):
    # 目的関数
    def residuals(params, x, y):
        model = double_gaussian(x, *params)
        return y - model

    # フィッティングした結果、元データからの乖離が最小となるパラメータを選択する
    params = []
    fitted_y = None
    error = 1e15

    for i in range(len(df_init)):
        initial_params = set_init_2(data, df_init.iloc[i, 0])
        if initial_params is None:
            pass
        else:
            mu1_0, A1_0, mu2_0, A2_0 = initial_params
            initial_guess = [
                mu1_0,
                df_init.iloc[i, 1],
                A1_0,
                mu2_0,
                df_init.iloc[i, 2],
                A2_0,
            ]
            res = least_squares(residuals, initial_guess, args=(x, y))
            popt = res.x
            fitted_y = double_gaussian(x, *popt)
            SSR = np.sum(((y - fitted_y) ** 2) * x)
            SSR = np.sqrt(SSR)

            if SSR < error:
                error = SSR
                params = popt

    return params, error, fitted_y


# ３つの正規分布へのフィッティング
def triple_gaussian_fitting(data, x, y, df_init):
    # 目的関数
    def residuals(params, x, y):
        model = triple_gaussian(x, *params)
        return y - model

    # フィッティングした結果、元データからの乖離が最小となるパラメータを選択する
    params = []
    fitted_y = None
    error = 1e15

    for i in range(len(df_init)):
        initial_params = set_init_3(data, df_init.iloc[i, 0], df_init.iloc[i, 1])
        if initial_params is None:
            pass
        else:
            mu1_0, A1_0, mu2_0, A2_0, mu3_0, A3_0 = initial_params
            initial_guess = [
                mu1_0,
                df_init.iloc[i, 2],
                A1_0,
                mu2_0,
                df_init.iloc[i, 3],
                A2_0,
                mu3_0,
                df_init.iloc[i, 4],
                A3_0,
            ]
            res = least_squares(residuals, initial_guess, args=(x, y))
            popt = res.x
            fitted_y = triple_gaussian(x, *popt)
            SSR = np.sum(((y - fitted_y) ** 2) * x)
            SSR = np.sqrt(SSR)

            if SSR < error:
                error = SSR
                params = popt

    return params, error, fitted_y


# パラメータを並び変える関数 (for ２山)
def sort_params_2(params):
    # mu1, sigma1, A1, mu2, sigma2, A2を分割
    mu_values = np.array([params[0], params[3]])  # mu1, mu2
    sigma_values = np.array([params[1], params[4]])  # sigma1, sigma2
    A_values = np.array([params[2], params[5]])  # A1, A2

    # mu_values に従ってインデックスを取得し、mu, sigma, A を一緒に並べ替え
    sorted_indices = np.argsort(mu_values)  # mu_valuesを昇順に並べ替えたインデックス

    mu_sorted = mu_values[sorted_indices]
    sigma_sorted = sigma_values[sorted_indices]
    A_sorted = A_values[sorted_indices]

    # 並べ替えた結果を結合して返す
    sorted_params = np.concatenate([mu_sorted, sigma_sorted, A_sorted])
    return sorted_params


# パラメータを並び変える関数 (for ３山)
def sort_params_3(params):
    # mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3 を分割
    mu_values = np.array([params[0], params[3], params[6]])  # mu1, mu2, mu3
    sigma_values = np.array([params[1], params[4], params[7]])  # sigma1, sigma2, sigma3
    A_values = np.array([params[2], params[5], params[8]])  # A1, A2, A3

    # mu_values に従ってインデックスを取得し、mu, sigma, A を一緒に並べ替え
    sorted_indices = np.argsort(mu_values)  # mu_valuesを昇順に並べ替えたインデックス

    mu_sorted = mu_values[sorted_indices]
    sigma_sorted = sigma_values[sorted_indices]
    A_sorted = A_values[sorted_indices]

    # 並べ替えた結果を結合して返す
    sorted_params = np.concatenate([mu_sorted, sigma_sorted, A_sorted])
    return sorted_params


# ２つの山の距離を計算する関数（簡略化したマハラノビス距離））
def Mahalanobis_distance(mu1, mu2, sigma1, sigma2):
    Dm = (mu2 - mu1) / np.sqrt(sigma2**2 + sigma1**2)
    Dm = np.abs(Dm)
    return Dm


# Red方向にFAP成分を抽出する関数
def FAP_R(data, sample_name, data_name, ax):
    x_0, y_0 = eh.make(data)

    # データ全体のうち赤の輝度が低いものについて解析
    x = x_0[x_0 < 10000]
    y = y_0[x_0 < 10000]

    # データ点の生成（各ビンの中心値をy回繰り返す）
    X = np.repeat(x, y)

    # VBGMMモデルの設定と学習
    n_components = 1  # 最大コンポーネント数
    vbgmm = BayesianGaussianMixture(n_components=n_components, random_state=42)
    vbgmm.fit(X.reshape(-1, 1))

    # パラメータの抽出
    mu_r1 = vbgmm.means_[0, 0]
    sigma_r1 = np.sqrt(vbgmm.covariances_[0, 0, 0])
    civ = 3.29
    r1 = mu_r1 + civ * sigma_r1

    # プロット
    counts, bins, patches = ax.hist(
        X, bins=len(x), density=True, alpha=1, color="skyblue"
    )
    x_range = np.linspace(0, 20000, 1000)
    pdf = 1 * norm.pdf(x_range, mu_r1, sigma_r1)
    ax.plot(x_range, pdf, linewidth=3)

    # 閾値ライン
    ax.vlines(
        r1,
        0,
        np.max(counts),
        color="red",
        linestyles="--",
        linewidth=3,
        label="threshold",
    )

    # 軸設定
    ax.set_xlim([0, 20000])
    ax.set_xlabel("green fluorescence intensity", fontsize=12)
    ax.set_title(f"{sample_name}_{data_name}_FAP_red_histogram", fontsize=12)

    return mu_r1, sigma_r1, r1


def FAP_G(data, sample_name, data_name, ax):
    x, y = eh.make(data, num_bins=100, sigma=3)
    X = np.repeat(x, y)

    n_components = 2
    vbgmm = BayesianGaussianMixture(n_components=n_components, random_state=42)
    vbgmm.fit(X.reshape(-1, 1))

    means = vbgmm.means_.flatten()
    covariances = vbgmm.covariances_.flatten()
    weights = vbgmm.weights_.flatten()

    covs = np.sqrt(covariances)  # ★ 標準偏差（sigma）を作る

    def mahalanobis_distance(mu1, mu2, sigma1, sigma2):
        Dm = (mu2 - mu1) / np.sqrt(sigma2**2 + sigma1**2)
        return np.abs(Dm)

    # ★ ここは covariances の sqrt を二重にしない
    FAP_Dm = mahalanobis_distance(means[0], means[1], covs[0], covs[1])

    if FAP_Dm < 0.5:
        vbgmm = BayesianGaussianMixture(n_components=1, random_state=42)
        vbgmm.fit(X.reshape(-1, 1))
        mu_g1 = vbgmm.means_[0, 0]
        sigma_g1 = np.sqrt(vbgmm.covariances_[0, 0, 0])
        means, covs, weights = [mu_g1], [sigma_g1], [1.0]
    else:
        top_components = sorted(
            [(i, w, m, s) for i, (w, m, s) in enumerate(zip(weights, means, covs))],
            key=lambda x: x[2],
        )
        mu_g1 = top_components[1][2]
        sigma_g1 = top_components[1][3]

    civ = 2.58
    g1 = mu_g1 - civ * sigma_g1
    g0 = 7500 if g1 > 7500 else g1 - 1

    hist = ax.hist(
        data, bins=100, density=True, color="black", alpha=0.3, label="original"
    )

    x_range = np.linspace(0, 20000, 1000)
    vbgmm_fits = []
    for i, (mean, sigma, weight) in enumerate(zip(means, covs, weights), start=1):
        pdf = weight * norm.pdf(x_range, mean, sigma)  # ★ sigma(標準偏差)を渡す
        (vgbmm_fit,) = ax.plot(
            x_range, pdf, linewidth=3, label=f"VGBMM Fit {i}", color="green"
        )
        vbgmm_fits.append(vgbmm_fit)

    threshold = ax.vlines(
        [g0, g1],
        0,
        np.max(ax.hist(data, bins=100)[0]),
        color="red",
        linestyles="--",
        linewidth=3,
        label="threshold",
    )
    ax.legend(handles=[hist[2][0], *vbgmm_fits, threshold], fontsize=12)
    ax.set_xlim([0, 20000])
    ax.set_xlabel("green fluorescence intensity", fontsize=16)
    ax.set_title("FAP green Histogram with VBGMM Fit", fontsize=16)

    return mu_g1, sigma_g1, g0, g1, FAP_Dm


# Green方向にDPP4成分をフィッティングする関数
def DPP4_G(data, sample_name, data_name, ax):
    x, y = eh.make(data)

    params, _, _ = double_gaussian_fitting(data, x, y, green_init())
    params = [abs(i) for i in params]
    params_sorted = sort_params_2(params)

    mu1 = params_sorted[0]
    mu2 = params_sorted[1]
    sigma1 = params_sorted[2]
    sigma2 = params_sorted[3]

    civ = 3.29
    g2 = mu1 + civ * sigma1
    g3 = mu2 - civ * sigma2
    g4 = mu2 + civ * sigma2

    g_Dm = Mahalanobis_distance(mu1, mu2, sigma1, sigma2)

    # フィット曲線の生成
    x_range = np.linspace(0, np.max(x) * 1.1, 10000)
    fit = double_gaussian(x_range, *params)

    # 描画（axベース）
    ax.plot(x, y, "b.", label="Data")
    ax.plot(x_range, fit, "r-", label="Fit")
    ax.set_xlim([0, 20000])
    ax.set_ylim([0, np.max(y) * 1.1])
    ax.vlines(
        [g2, g3, g4], 0, np.max(y) * 1.1, colors="black", linestyles="dashed", alpha=0.5
    )

    ax.set_xlabel("green fluorescence intensity", fontsize=12)
    ax.set_title(f"{sample_name}_{data_name}_DPP4_green_histogram", fontsize=12)

    return mu1, sigma1, mu2, sigma2, g2, g3, g4, g_Dm


# Red方向にDPP4成分をフィッティングする関数
def DPP4_R(data, green, g1, g2, g3, sample_name, data_name, ax):
    CIV1 = 2.58
    CIV2 = 3.29

    def plot_model_with_lines(x, y, model_func, params, cut_points):
        """2山 or 3山フィット結果 + 閾値ラインを描画"""
        x_range = np.linspace(0, np.max(x) * 1.1, 10000)
        fit = model_func(x_range, *params)

        ax.plot(x, y, "b.", label="Data")
        ax.plot(x_range, fit, "r-", label="Fit")
        ax.vlines(
            cut_points,
            0,
            np.max(y) * 1.1,
            colors="black",
            linestyles="dashed",
            alpha=0.5,
        )

        ax.set_xlim([0, 100000])
        ax.set_ylim([0, np.max(y) * 1.1])
        ax.set_xlabel("red fluorescence intensity", fontsize=12)
        ax.set_title(f"{sample_name}_{data_name}_DPP4_red_histogram", fontsize=12)

    def plot_hist_with_lines(x, y, cut_points):
        """ヒストグラム + 閾値ラインのみ描画（フォールバック用）"""
        ax.plot(x, y, "b.", label="Data")
        ax.vlines(
            cut_points,
            0,
            np.max(y) * 1.1,
            colors="black",
            linestyles="dashed",
            alpha=0.5,
        )

        ax.set_xlim([0, 100000])
        ax.set_ylim([0, np.max(y) * 1.1])
        ax.set_xlabel("red fluorescence intensity", fontsize=12)
        ax.set_title(f"{sample_name}_{data_name}_DPP4_red_histogram", fontsize=12)

    try:
        x, y = eh.make(data)

        # まず2山フィット
        params, err_double, _ = double_gaussian_fitting(data, x, y, red_init_2())
        params = [abs(i) for i in params]
        params_sorted = sort_params_2(params)

        mu1 = params_sorted[0]
        mu2 = params_sorted[1]
        sigma1 = params_sorted[2]
        sigma2 = params_sorted[3]

        r_Dm = Mahalanobis_distance(mu1, mu2, sigma1, sigma2)
        error = err_double  # 現時点での「最後のフィット誤差」

        # Dm < 1 の場合は3山フィットを試す
        if r_Dm < 1:
            params, err_triple, _ = triple_gaussian_fitting(data, x, y, red_init_3())
            error = err_triple  # 3山フィットが「最後」の誤差に更新
            params = [abs(i) for i in params]
            params_sorted = sort_params_3(params)

            mu1 = params_sorted[0]
            mu2 = params_sorted[1]
            sigma1 = params_sorted[3]
            sigma2 = params_sorted[4]

            r2 = mu1 - CIV2 * sigma1
            r3 = mu1 + 15000
            r4 = mu2 - CIV1 * sigma2
            r5 = mu2 + CIV2 * sigma2

            # ２つの山の距離（3山フィット後のもの）
            r_Dm = Mahalanobis_distance(mu1, mu2, sigma1, sigma2)

            # ★ ここでの条件は元コードと同じく Dm < 2
            if r_Dm < 2:
                # green 方向で分割して1山フィットするフォールバック
                # 左側（green < g1）
                data_1 = data[green < g1]
                x1, y1 = eh.make(data_1)
                params_1, err_g1, _ = gaussian_fitting(data_1, x1, y1)
                error = err_g1  # ひとまず更新
                params_1 = [abs(i) for i in params_1]
                mu1, sigma1, _ = params_1
                r5 = mu1 + CIV1 * sigma1

                # 中央（g2 < green < g3）
                data_2 = data[(g2 < green) & (green < g3)]
                x2, y2 = eh.make(data_2)
                params_2, err_g2, _ = gaussian_fitting(data_2, x2, y2)
                params_2 = [abs(i) for i in params_2]
                mu2, sigma2, _ = params_2

                error = err_g2  # ★ 最後に実行したフィット（data_2）の誤差を最終errorに
                r3 = mu2 + CIV1 * sigma2

                r2 = r4 = r_Dm = 0  # 境界扱い不能なので 0

                # ★ Dm < 2 のフォールバックでもプロットを行う
                #    ここでは「全データのヒストグラム + r3, r5 ライン」を描画
                x_all, y_all = x, y  # すでに eh.make(data) 済
                plot_hist_with_lines(x_all, y_all, [r3, r5])

            else:
                # 3山フィットの結果を描画
                plot_model_with_lines(x, y, triple_gaussian, params, [r2, r3, r4, r5])

        else:
            # Dm >= 1 の場合は2山フィットの結果を使用
            r2 = mu1 - CIV2 * sigma1
            r3 = mu1 + 15000
            r4 = mu2 - CIV1 * sigma2
            r5 = mu2 + CIV2 * sigma2

            r_Dm = Mahalanobis_distance(mu1, mu2, sigma1, sigma2)

            # 2山フィット結果を描画
            plot_model_with_lines(x, y, double_gaussian, params, [r2, r3, r4, r5])

        return mu1, sigma1, mu2, sigma2, r2, r3, r4, r5

    except Exception:
        # フォールバック：green で2分して1山フィット×2
        data_1 = data[green < g1]
        x1, y1 = eh.make(data_1)
        params_1, err_g1, _ = gaussian_fitting(data_1, x1, y1)
        params_1 = [abs(i) for i in params_1]
        mu1, sigma1, _ = params_1
        r5 = mu1 + CIV1 * sigma1

        data_2 = data[(g2 < green) & (green < g3)]
        x2, y2 = eh.make(data_2)
        params_2, err_g2, _ = gaussian_fitting(data_2, x2, y2)
        params_2 = [abs(i) for i in params_2]
        mu2, sigma2, _ = params_2
        r3 = mu2 + CIV1 * sigma2

        # 最後に実行したフィット(data_2)の誤差を採用
        error = err_g2
        r2 = r4 = r_Dm = 0

        # 例外時も一応ヒスト + 閾値ラインを描画しておく
        x_all, y_all = eh.make(data)
        plot_hist_with_lines(x_all, y_all, [r3, r5])

        return mu1, sigma1, mu2, sigma2, r2, r3, r4, r5

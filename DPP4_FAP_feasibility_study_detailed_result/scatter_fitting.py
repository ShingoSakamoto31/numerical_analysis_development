import matplotlib.pyplot as plt
import numpy as np

import gaussian_fitting as gf


def scatter_fitting(green, red, sample, fields, output) -> tuple:
    fitting_fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    _, _, r1 = gf.FAP_R(red, sample, fields.data_name, axes[0])
    _, _, g0, g1, _ = gf.FAP_G(green[red < r1], sample, fields.data_name, axes[0])
    _, _, _, _, g2, g3, g4, _ = gf.DPP4_G(
        green[r1 < red], sample, fields.data_name, axes[1]
    )
    _, _, _, _, r2, r3, r4, r5 = gf.DPP4_R(
        red[r1 < red],
        green[r1 < red],
        g1,
        g2,
        g3,
        sample,
        fields.data_name,
        axes[2],
    )
    output(fitting_fig, f"{fields.data_name}_fit.png", "fit", bbox_inches="tight")

    return g0, g1, g2, g3, g4, r1, r2, r3, r4, r5


def scatter_image_output(
    green, red, g0, g1, g2, g3, g4, r1, r2, r3, r4, r5, sample, fields, output
) -> None:
    scatter_fig = plt.figure(figsize=(8.7, 8))
    axes = scatter_fig.gca()
    axes.scatter(green, red, color="black", s=3)

    axes.vlines(g0, 0, r1, colors="red", linestyles="dashed", linewidth=3)
    axes.vlines(g1, 0, r1, colors="red", linestyles="dashed", linewidth=3)
    axes.vlines(g2, r1, r5, colors="red", linestyles="dashed", linewidth=3)
    axes.vlines(g3, r1, r3, colors="red", linestyles="dashed", linewidth=3)
    axes.vlines(g4, r1, r3, colors="red", linestyles="dashed", linewidth=3)

    axes.hlines(r1, 0, 25000, colors="red", linestyles="dashed", linewidth=3)
    axes.hlines(r2, 0, g2, colors="red", linestyles="dashed", linewidth=3)
    axes.hlines(r2, g3, g4, colors="red", linestyles="dashed", linewidth=3)
    axes.hlines(r3, g3, g4, colors="red", linestyles="dashed", linewidth=3)
    axes.hlines(r4, 0, g2, colors="red", linestyles="dashed", linewidth=3)
    axes.hlines(r5, 0, g2, colors="red", linestyles="dashed", linewidth=3)

    axes.set_xlabel("Green fluorescence intensity (a.u.)", fontsize=24)
    axes.set_ylabel("Red fluorescence intensity (a.u.)", fontsize=24)
    axes.set_xlim(0, 25000)
    axes.set_ylim(0, 100000)
    axes.tick_params(labelsize=24)

    scatter_fig.tight_layout()
    axes.set_title(f"{sample}_{fields.data_name}", fontsize=16)
    output(
        scatter_fig,
        f"{sample}_{fields.data_name}_scatter.png",
        "scatter",
        bbox_inches="tight",
    )


def paramater_calculator(
    green, red, g0, g1, g2, g3, g4, r1, r2, r3, r4, r5, total_wells
) -> tuple:
    if r2 == r4 == 0:
        area_1 = area_2 = area_3 = area_4 = area_5 = 0
    else:
        area_1 = np.size(green[(green < g2) & (r4 < red) & (red < r5)]) / total_wells
        area_2 = np.size(green[(green < g2) & (r2 < red) & (red < r4)]) / total_wells
        area_3 = np.size(green[(green < g2) & (r1 < red) & (red < r2)]) / total_wells
        area_4 = (
            np.size(green[(g3 < green) & (green < g4) & (r2 < red) & (red < r3)])
            / total_wells
        )
        area_5 = (
            np.size(green[(g3 < green) & (green < g4) & (r1 < red) & (red < r2)])
            / total_wells
        )

    area_6 = np.size(green[(green < g0) & (red < r1)]) / total_wells
    area_7 = np.size(green[(g0 < green) & (green < g1) & (red < r1)]) / total_wells
    area_8 = np.size(green[(g1 < green) & (red < r1)]) / total_wells

    dpp4_homodimer = (
        np.size(green[(green < g2) & (r1 < red) & (red < r5)]) / total_wells
    )
    dpp4_fap_heterodimer = (
        np.size(green[(g3 < green) & (green < g4) & (r1 < red) & (red < r3)])
        / total_wells
    )
    fap_homodimer = np.size(green[red < r1]) / total_wells

    if dpp4_homodimer + dpp4_fap_heterodimer == 0:
        dpp4_homodimer_ratio = 0
    else:
        dpp4_homodimer_ratio = (dpp4_homodimer) / (
            dpp4_homodimer + dpp4_fap_heterodimer
        )

    return (
        area_1,
        area_2,
        area_3,
        area_4,
        area_5,
        area_6,
        area_7,
        area_8,
        dpp4_homodimer,
        dpp4_fap_heterodimer,
        fap_homodimer,
        dpp4_homodimer_ratio,
    )

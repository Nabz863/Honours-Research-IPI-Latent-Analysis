#!/usr/bin/env python3
"""analysis/make_figures.py -- every data figure in the paper, drawn from the
analysis outputs (no numbers typed into LaTeX). CPU only, read-only; writes one
PDF per figure. Run anywhere matplotlib is installed (not the login node).

  python3 analysis/make_figures.py --analysis_out analysis_out --out paper/figures

Figures and their inputs (all under --analysis_out):
  fig_tradeoff.pdf      agentdojo_behaviour_gemma4_qwen3vl/numbers.json
  fig_compliance.pdf    compliance_bootstrap.json
  fig_rq1_layers.pdf    geometry_v2_agentdojo_{gemma4,qwen3vl}/displacement_payload_turn.csv
  fig_rq1_topology.pdf  directions_agentdojo_{gemma4,qwen3vl}.json   (topology)
  fig_rq2_layers.pdf    rq2_agentdojo_{gemma4,qwen3vl}.json         (layer_sweep)
  fig_rq2_vwa.pdf       rq2_vwa_{gemma4,qwen3vl}.json               (per_defense)
Every figure fails loudly if an input or an expected cell is missing.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

COL_W, TEXT_W = 3.5, 7.16                     # IEEE conference column / text width (in)
BLUE, ORANGE, GREY = "#2A78D6", "#EB6834", "#8a8a8a"
MODELS = (("gemma4", "Gemma-4-12B"), ("qwen3vl", "Qwen3-VL-8B"))
N_LAYERS = {"gemma4": 48, "qwen3vl": 36}
ANALYSIS_LAYER = {"gemma4": 42, "qwen3vl": 31}
# sequential, white -> dark red: darker = more of the attack survives the defence
REMAIN = LinearSegmentedColormap.from_list("remain", ["#ffffff", "#fcbba1", "#ef3b2c", "#99000d"])

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def need(path):
    if not os.path.exists(path):
        sys.exit(f"FATAL: {path} not found")
    return path


def load(path):
    return json.load(open(need(path)))


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p}")


def frac(kn):
    k, n = kn
    if not n:
        sys.exit(f"FATAL: empty cell {kn}")
    return 100.0 * k / n


# ------------------------------------------------------------------ tradeoff
TRADEOFF_ROWS = (("none", "none"), ("spotlighting_delimiting", "delimiting"),
                 ("spotlighting_datamarking", "datamarking"), ("spotlighting_encoding", "encoding"),
                 ("icon", "ICON"), ("attention_tracker", "AT"))


def tradeoff_values(numbers, attack="important_instructions"):
    """{model_key: [(label, asr %, defended-clean utility %)]} in TRADEOFF_ROWS order."""
    out = {}
    for mk, _ in MODELS:
        hits = [m for m in numbers["models"] if ("gemma" if mk == "gemma4" else "qwen") in m.lower()]
        if len(hits) != 1:
            sys.exit(f"FATAL: model {mk}: {hits} in numbers.json models {list(numbers['models'])}")
        cells = numbers["models"][hits[0]]["cells"]
        rows = []
        for dn, lab in TRADEOFF_ROWS:
            c = cells.get(f"{attack}|{dn}")
            if c is None:
                sys.exit(f"FATAL: {mk}: no cell {attack}|{dn}")
            rows.append((lab, frac(c["asr"]), frac(c["dclean_ua"])))
        out[mk] = rows
    return out


def fig_tradeoff(numbers, out):
    vals = tradeoff_values(numbers)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True)
    for ax, (mk, title), col in zip(axes, MODELS, (BLUE, ORANGE)):
        rows = vals[mk]
        y = np.arange(len(rows))[::-1]
        for yi, (_, a, u) in zip(y, rows):
            ax.plot([a, u], [yi, yi], color="#cfcfcf", lw=1.2, zorder=1)
        ax.scatter([r[1] for r in rows], y, s=14, color=col, zorder=3, label="attack success")
        ax.scatter([r[2] for r in rows], y, s=14, facecolor="white", edgecolor=col, marker="s",
                   linewidth=0.9, zorder=3, label="clean utility under defence")
        ax.set_title(title, pad=3)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("%", labelpad=1)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_yticks(np.arange(len(TRADEOFF_ROWS))[::-1], [lab for _, lab in TRADEOFF_ROWS])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.03))
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "fig_tradeoff.pdf")
    return vals


# ------------------------------------------------------------------ compliance
def compliance_values(comp):
    """[(label, model_key, lower %, upper %, point %)] top to bottom."""
    out = []
    for mk, mname in MODELS:
        for atk, short in (("important_instructions", "II"), ("template_ipi", "T")):
            key = f"final_agentdojo_{mk}|{atk}"
            c = comp["cells"].get(key)
            if not c or "lower" not in c:
                sys.exit(f"FATAL: compliance cell {key} missing or undefined")
            out.append((f"{mname.split('-')[0]}-{mname.split('-')[1]}, {short}", mk,
                        100 * c["lower"], 100 * c["upper"], 100 * c["asr_e"] / c["c"]))
    return out


def fig_compliance(comp, out):
    vals = compliance_values(comp)
    fig, ax = plt.subplots(figsize=(COL_W * 0.95, 1.45))
    y = np.arange(len(vals))[::-1]
    for yi, (_, mk, lo, hi, pt) in zip(y, vals):
        ax.plot([lo, hi], [yi, yi], color=BLUE if mk == "gemma4" else ORANGE, lw=4.5,
                solid_capstyle="butt", zorder=2)
        ax.scatter([pt], [yi], s=16, facecolor="white", edgecolor="black", linewidth=0.9, zorder=3)
    ax.set_yticks(y, [v[0] for v in vals])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xlabel(r"probability of complying when exposed, $P(K\mid E)$ (%)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    save(fig, out, "fig_compliance.pdf")
    return vals


# ------------------------------------------------------------------ RQ1 layers
def rq1_layer_values(csv_paths):
    """{model_key: {(attack, outcome): [(relative depth, delta_norm_mean)]}}"""
    out = {}
    for mk, p in csv_paths.items():
        rows = list(csv.DictReader(open(need(p))))
        d = {}
        for r in rows:
            d.setdefault((r["attack"], r["outcome"]), []).append(
                (int(r["layer"]) / (N_LAYERS[mk] - 1), float(r["delta_norm_mean"])))
        for k in (("important_instructions", "success"), ("important_instructions", "failure"),
                  ("template_ipi", "success"), ("template_ipi", "failure")):
            if len(d.get(k, [])) != N_LAYERS[mk]:
                sys.exit(f"FATAL: {p}: {k} has {len(d.get(k, []))} layers, expected {N_LAYERS[mk]}")
        out[mk] = {k: sorted(v) for k, v in d.items()}
    return out


def fig_rq1_layers(csv_paths, out):
    vals = rq1_layer_values(csv_paths)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    style = {("important_instructions", "success"): (BLUE, "-", "II succeeded"),
             ("important_instructions", "failure"): (BLUE, "--", "II failed"),
             ("template_ipi", "success"): (ORANGE, "-", "T succeeded"),
             ("template_ipi", "failure"): (ORANGE, "--", "T failed")}
    for ax, (mk, title) in zip(axes, MODELS):
        ax.axvline(ANALYSIS_LAYER[mk] / (N_LAYERS[mk] - 1), color="#bdbdbd", lw=0.8, ls=(0, (2, 2)))
        for k, (c, ls, lab) in style.items():
            xs, ys = zip(*vals[mk][k])
            ax.plot(xs, ys, color=c, ls=ls, lw=1.1, label=lab)
        ax.set_title(title, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
        ax.set_xlabel("relative depth", labelpad=1)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$\Vert\Delta h\Vert / \Vert h_{\mathrm{clean}}\Vert$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, -0.05),
               handlelength=1.8, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.07)
    save(fig, out, "fig_rq1_layers.pdf")
    return vals


# ------------------------------------------------------------------ RQ1 topology
def topology_values(directions, attack="important_instructions", dim="0"):
    out = {}
    for mk, d in directions.items():
        t = (d.get("topology") or {}).get(attack)
        if not t or dim not in t["curves"]:
            sys.exit(f"FATAL: {mk}: no topology curve {attack} H{dim}")
        c = t["curves"][dim]
        n = len(c["clean"])
        out[mk] = {"x": [i / (n - 1) for i in range(n)], **{k: c[k] for k in
                   ("clean", "attacked", "swap_lo", "swap_hi")}}
    return out


def fig_rq1_topology(directions, out, xmax=0.4):
    vals = topology_values(directions)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True)
    for ax, (mk, title) in zip(axes, MODELS):
        v = vals[mk]
        x = np.array(v["x"])
        ax.fill_between(x, v["swap_lo"], v["swap_hi"], color="#dcdcdc", lw=0, label="swap null 95%",
                        step=None)
        ax.plot(x, v["clean"], color="#444444", ls="--", lw=1.1, label="clean")
        ax.plot(x, v["attacked"], color=BLUE, lw=1.2, label="attacked")
        ax.set_title(title, pad=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
        ax.set_xlabel("radius (quantile of distances)", labelpad=1)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"components ($\beta_0$)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.05))
    fig.subplots_adjust(wspace=0.07)
    save(fig, out, "fig_rq1_topology.pdf")
    return vals


# ------------------------------------------------------------------ RQ2 heatmaps
RQ2_LAYER_ROWS = (("spotlighting_delimiting", "delimiting"), ("spotlighting_datamarking", "datamarking"),
                  ("spotlighting_encoding", "encoding"), ("icon", "ICON"),
                  ("attention_tracker", r"AT $\tau{=}0.5$"), ("attention_tracker_t30", r"AT $\tau{=}0.3$"))
RQ2_LAYER_RANGE = {"gemma4": range(24, 43), "qwen3vl": range(18, 32)}
VMAX = 1.2


def rq2_layer_values(rq2):
    """{model_key: matrix [rows x layers] of along_frac_median (NaN where absent)}"""
    out = {}
    for mk, d in rq2.items():
        sw = ((d.get("per_attack") or {}).get("important_instructions") or {}).get("layer_sweep") or {}
        if not sw:
            sys.exit(f"FATAL: {mk}: no layer_sweep for important_instructions")
        L = list(RQ2_LAYER_RANGE[mk])
        M = np.full((len(RQ2_LAYER_ROWS), len(L)), np.nan)
        for i, (dn, _) in enumerate(RQ2_LAYER_ROWS):
            if dn not in sw:
                sys.exit(f"FATAL: {mk}: no layer sweep for {dn}")
            for e in sw[dn]:
                if e["layer"] in L and e["along_frac_median"] is not None:
                    M[i, L.index(e["layer"])] = e["along_frac_median"]
        out[mk] = M
    return out


def _cells(ax, M, extent):
    Mc = np.clip(M, 0, VMAX)
    return ax.imshow(np.ma.masked_invalid(Mc), cmap=REMAIN, vmin=0, vmax=VMAX, aspect="auto",
                     interpolation="nearest", extent=extent)


def _colorbar(fig, im, ax, width=0.018):
    """Colour bar in its own axes, right of `ax` (never drawn over a panel)."""
    b = ax.get_position()
    cax = fig.add_axes([b.x1 + 0.015, b.y0, width, b.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label(r"$a_\parallel / \Vert\Delta h\Vert$", labelpad=2)


def fig_rq2_layers(rq2, out):
    vals = rq2_layer_values(rq2)
    widths = [len(RQ2_LAYER_RANGE[mk]) for mk, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55), gridspec_kw={"width_ratios": widths})
    for ax, (mk, title) in zip(axes, MODELS):
        L = list(RQ2_LAYER_RANGE[mk])
        im = _cells(ax, vals[mk], (L[0] - 0.5, L[-1] + 0.5, len(RQ2_LAYER_ROWS) - 0.5, -0.5))
        ax.set_title(title, pad=3)
        ax.set_xticks(L[::4])
        ax.set_xlabel("layer")
        ax.set_yticks(range(len(RQ2_LAYER_ROWS)), [lab for _, lab in RQ2_LAYER_ROWS] if ax is axes[0] else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.05, right=0.88)
    _colorbar(fig, im, axes[-1])
    save(fig, out, "fig_rq2_layers.pdf")
    return vals


VWA_ROWS = (("template_ipi", "Template IPI"), ("cross_modal_injection", "CrossInject"), ("comet", "COMET"),
            ("typographic", "Typographic"), ("steganographic", "Steganographic"),
            ("pixel_perturbation", "Pixel"), ("adversarial_patch", "Patch"),
            ("bim_caption_transfer", "BIM transfer"), ("clip_attack_transfer", "CLIP transfer"))
VWA_COLS = (("spotlighting_delimiting", "delim."), ("spotlighting_datamarking", "datamark."),
            ("spotlighting_encoding", "encoding"), ("icon", "ICON"), ("gaussian_blur_1.5", "blur 1.5"),
            ("gaussian_blur_3.0", "blur 3"), ("ecso", "ECSO"))


def rq2_vwa_values(rq2):
    """{model_key: (matrix of along_frac_median, boolean matrix: verdict RESTORATION)}"""
    out = {}
    for mk, d in rq2.items():
        M = np.full((len(VWA_ROWS), len(VWA_COLS)), np.nan)
        B = np.zeros(M.shape, bool)
        for i, (atk, _) in enumerate(VWA_ROWS):
            pa = (d.get("per_attack") or {}).get(atk)
            if not pa:
                sys.exit(f"FATAL: {mk}: no RQ2 rows for {atk}")
            by = {e["defense"]: e for e in pa["per_defense"]}
            for j, (dn, _) in enumerate(VWA_COLS):
                e = by.get(dn)
                if e is None or e.get("along_frac_median") is None:
                    sys.exit(f"FATAL: {mk}: no along_frac_median for {atk} x {dn}")
                M[i, j] = e["along_frac_median"]
                B[i, j] = str(e.get("verdict", "")).startswith("RESTORATION")
        out[mk] = (M, B)
    return out


def fig_rq2_vwa(rq2, out):
    vals = rq2_vwa_values(rq2)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W * 0.92, 2.9), sharey=True)
    for ax, (mk, title) in zip(axes, MODELS):
        M, B = vals[mk]
        im = _cells(ax, M, (-0.5, M.shape[1] - 0.5, M.shape[0] - 0.5, -0.5))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if min(M[i, j], VMAX) > 0.75 else "black")
                if B[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.4))
        ax.set_title(title, pad=3)
        ax.set_xticks(range(len(VWA_COLS)), [lab for _, lab in VWA_COLS], rotation=35, ha="right")
        ax.set_yticks(range(len(VWA_ROWS)), [lab for _, lab in VWA_ROWS])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.04, right=0.92)
    _colorbar(fig, im, axes[-1], width=0.01)
    save(fig, out, "fig_rq2_vwa.pdf")
    return vals


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_out", default="analysis_out")
    ap.add_argument("--behaviour", default="agentdojo_behaviour_gemma4_qwen3vl/numbers.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", help="also write every plotted number to this JSON (for checking)")
    a = ap.parse_args()
    A = lambda *p: os.path.join(a.analysis_out, *p)
    os.makedirs(a.out, exist_ok=True)
    dump = {
        "tradeoff": fig_tradeoff(load(A(a.behaviour)), a.out),
        "compliance": fig_compliance(load(A("compliance_bootstrap.json")), a.out),
        "rq1_layers": {mk: {"|".join(k): v for k, v in d.items()} for mk, d in fig_rq1_layers(
            {mk: A(f"geometry_v2_agentdojo_{mk}", "displacement_payload_turn.csv") for mk, _ in MODELS},
            a.out).items()},
        "rq1_topology": fig_rq1_topology({mk: load(A(f"directions_agentdojo_{mk}.json")) for mk, _ in MODELS},
                                         a.out),
        "rq2_layers": {mk: m.tolist() for mk, m in fig_rq2_layers(
            {mk: load(A(f"rq2_agentdojo_{mk}.json")) for mk, _ in MODELS}, a.out).items()},
        "rq2_vwa": {mk: {"along": m.tolist(), "restoration": b.tolist()} for mk, (m, b) in fig_rq2_vwa(
            {mk: load(A(f"rq2_vwa_{mk}.json")) for mk, _ in MODELS}, a.out).items()},
    }
    if a.dump:
        json.dump(dump, open(a.dump, "w"), indent=1)
        print(f"wrote {a.dump}")


if __name__ == "__main__":
    main()

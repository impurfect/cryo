#!/usr/bin/env python3
"""
analyze.py - turn the measurement tables into plots, statistics and conclusions.

WHAT THIS SCRIPT DOES, IN PLAIN LANGUAGE
----------------------------------------
run_workflow.py did the heavy work on the GPU and left behind a handful of small
CSV tables. This script reads only those tables. It needs no GPU, no cryo-ET
software and about one second, so you can re-run it on a laptop as often as you
like while you think about how to present the result.

It answers two questions.

  TASK 1 - does it matter which alignment method you use?
    Both alignment branches were pushed all the way through to particle picking,
    so we can compare them on things that mean the same for both:
      * how sharp the resulting tomograms are
      * how many molecules the picker finds in them
      * how confidently it finds them
      * whether the two branches agree about where the molecules are
      * how long each took
    We deliberately do NOT compare the two aligners' own internally reported
    residuals against each other. IMOD reports the average distance in pixels
    between predicted and observed positions of tracked patches; AreTomo reports
    an error from a completely different projection-matching calculation. They
    are different quantities. Declaring the smaller number the winner would be
    like saying a golfer beat a basketball player because they scored fewer
    points. Those numbers are reported, per method, as a diagnostic only.

  TASK 2 - does it matter which particle picker you use?
    The tomograms are identical for both pickers, so any difference is the
    picker. We compare counts, where the picks are, how much they agree, how the
    agreement depends on how strict we are about "same position", and whether
    the picks each tool finds alone really are its least confident ones.

Everything written to results/ is computed from the tables. Nothing is
hard-coded: if the data said the opposite tomorrow, the conclusions would say
the opposite too.

HOW TO RUN IT

    python analyze.py               # analyse the real results in results/tables/
    python analyze.py --selftest    # exercise every code path on obviously fake
                                    # data, written to results/selftest/ and
                                    # stamped SYNTHETIC on every single output
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # draw to files, never to a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import stats

import config

BRANCH_COLOR = {"etomo": "#1f77b4", "aretomo": "#ff7f0e"}
PICKER_COLOR = {"warp": "#1f77b4", "pytom": "#2ca02c"}

# Set to True by --selftest. Every plot title and every conclusion gets stamped.
SYNTHETIC = False
STAMP = "SYNTHETIC SELF-TEST DATA - NOT A SCIENTIFIC RESULT"


# ===========================================================================
#  SECTION A - loading, and the one geometric routine everything else uses
# ===========================================================================

def load(name, tables_dir, required=True):
    """Read one table, or stop with a message a human can act on."""
    path = tables_dir / name
    if not path.exists():
        if required:
            sys.exit(f"Missing {path}\nRun:  python run_workflow.py --collect")
        return pd.DataFrame()
    return pd.read_csv(path)


def match_one_to_one(a_xyz, b_xyz, radius_a):
    """Pair up two lists of 3D positions, each point used at most once.

    THE PROBLEM. Two programs each hand us a list of positions. We want to know
    which entries in list A and list B are the same physical molecule seen
    twice, and which are unique to one list.

    THE NAIVE APPROACH AND WHY IT IS WRONG. The obvious thing is to give every
    point in A its nearest neighbour in B. But two different points in A can
    both be nearest to the same point in B, so you get double counting; and if
    you fix that by claiming B-points first-come-first-served, then whether a
    given pair matches depends on the arbitrary order the file was written in.

    WHAT WE DO INSTEAD. We solve it properly, as an assignment problem: out of
    all the possible ways of pairing A-points with B-points using each point at
    most once, find the one with the smallest total distance. That is the
    Hungarian algorithm, and scipy has it as linear_sum_assignment. The answer
    does not depend on file order, and no point is ever used twice.

    The distance cutoff is applied by making any pair further apart than the
    radius astronomically expensive, then discarding whatever survives past the
    cutoff at the end.

    Returns (indices into A, indices into B, distances) for the matched pairs.
    """
    if len(a_xyz) == 0 or len(b_xyz) == 0:
        return np.array([], int), np.array([], int), np.array([], float)

    d = cdist(a_xyz, b_xyz)
    forbidden = radius_a * 1e6
    cost = np.where(d <= radius_a, d, forbidden)
    rows, cols = linear_sum_assignment(cost)
    dist = d[rows, cols]
    keep = dist <= radius_a
    return rows[keep], cols[keep], dist[keep]


def paired_test(values_a, values_b, label_a, label_b):
    """Compare two methods measured on the SAME five tilt series.

    Because both methods saw the same five samples, the right comparison is a
    PAIRED one: for each tilt series, did method A beat method B? That removes
    the (large) variation between tilt series and asks only about the method.

    With five pairs the smallest possible p-value from a Wilcoxon signed-rank
    test is about 0.06, so a "significant" result is essentially unobtainable
    and we should not pretend otherwise. What five pairs CAN tell you is whether
    the effect is consistent: 5 out of 5 in the same direction is meaningful
    evidence even without a small p-value, and that is what we report first.
    """
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    diff = a - b
    wins = int(np.sum(diff > 0))
    result = {
        "metric_higher_is": "",
        f"mean_{label_a}": round(float(a.mean()), 5),
        f"mean_{label_b}": round(float(b.mean()), 5),
        "mean_difference": round(float(diff.mean()), 5),
        "relative_difference_pct": round(float(100 * diff.mean() / (abs(b.mean()) + 1e-12)), 2),
        "n_pairs": len(a),
        f"{label_a}_higher_in": f"{wins}/{len(a)} tilt series",
    }
    if len(a) >= 3 and np.any(diff != 0):
        try:
            result["wilcoxon_p"] = round(float(stats.wilcoxon(a, b).pvalue), 4)
        except ValueError:
            result["wilcoxon_p"] = np.nan
    else:
        result["wilcoxon_p"] = np.nan
    return result


def finish(fig, path, title):
    """Add the synthetic-data stamp if needed, save, close."""
    if SYNTHETIC:
        fig.suptitle(STAMP, color="crimson", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    plot  {path.name}")


# ===========================================================================
#  SECTION B - TASK 1: does the alignment method matter?
# ===========================================================================

def analyse_alignment(tables, out_dir):
    """Compare the two alignment branches and write the Task 1 outputs."""
    print("\n--- Task 1: alignment comparison " + "-" * 38)
    tomo = tables["tomogram_stats"]
    picks = tables["warp_picks"]
    runtimes = tables["runtimes"]
    residuals = tables["native_residuals"]

    branches = [b for b in ("etomo", "aretomo") if b in set(tomo["branch"])]
    if len(branches) < 2:
        print("    Only one branch present - nothing to compare.")
        return {}

    # ---- 1. per-tilt-series measurements, side by side --------------------
    # For each tilt series we now have, for each branch: how sharp the tomogram
    # is, how many particles were found in it, and how strong those picks were.
    per_series = []
    for series in sorted(set(tomo["tilt_series"])):
        row = {"tilt_series": series}
        for b in branches:
            t = tomo[(tomo.branch == b) & (tomo.tilt_series == series)]
            p = picks[(picks.branch == b) & (picks.tilt_series == series)]
            strong = p[p.score >= config.PICK_THRESHOLD_SIGMA]
            row[f"contrast_{b}"] = float(t["contrast"].iloc[0]) if len(t) else np.nan
            row[f"sharpness_{b}"] = float(t["sharpness"].iloc[0]) if len(t) else np.nan
            row[f"n_picks_{b}"] = len(strong)
            row[f"median_score_{b}"] = round(float(strong["score"].median()), 3) if len(strong) else np.nan
            row[f"top50_score_{b}"] = (round(float(strong["score"].nlargest(50).mean()), 3)
                                       if len(strong) >= 1 else np.nan)
        per_series.append(row)
    per_series = pd.DataFrame(per_series)
    per_series.to_csv(out_dir / "task1_per_tilt_series.csv", index=False)
    print(f"    table task1_per_tilt_series.csv ({len(per_series)} tilt series)")

    # ---- 2. paired statistics on the comparable metrics --------------------
    # "top50_score" - the mean correlation score of the 50 strongest picks - is
    # the single most informative number here. It is measured in standard
    # deviations above the tomogram's own background, so it is directly
    # comparable between branches, and it responds to exactly the thing an
    # alignment error damages: how crisply the molecule's fine detail survives
    # into the reconstruction.
    a, b = branches[0], branches[1]
    stat_rows = []
    for metric, higher_is in [("sharpness", "better (crisper edges)"),
                              ("contrast", "better (more structure)"),
                              ("n_picks", "more particles found"),
                              ("median_score", "better (more confident picks)"),
                              ("top50_score", "better (strongest evidence)")]:
        col_a, col_b = f"{metric}_{a}", f"{metric}_{b}"
        sub = per_series[[col_a, col_b]].dropna()
        if len(sub) < 2:
            continue
        r = paired_test(sub[col_a], sub[col_b], a, b)
        r["metric"] = metric
        r["metric_higher_is"] = higher_is
        stat_rows.append(r)
    if not stat_rows:
        print("    No metric could be compared - are both branches complete?")
        return {}
    stats_df = pd.DataFrame(stat_rows)
    cols = ["metric", "metric_higher_is", f"mean_{a}", f"mean_{b}", "mean_difference",
            "relative_difference_pct", f"{a}_higher_in", "n_pairs", "wilcoxon_p"]
    stats_df = stats_df[cols]
    stats_df.to_csv(out_dir / "task1_summary.csv", index=False)
    print(f"    table task1_summary.csv")

    # ---- 3. do the two branches find the SAME molecules? -------------------
    # This is a purely geometric check that needs no notion of "quality". If
    # both alignments are sound they should place the same molecules in nearly
    # the same places. Large disagreement means at least one of them has the
    # geometry wrong.
    agree_rows = []
    for series in sorted(set(picks["tilt_series"])):
        pa = picks[(picks.branch == a) & (picks.tilt_series == series)]
        pb = picks[(picks.branch == b) & (picks.tilt_series == series)]
        pa = pa[pa.score >= config.PICK_THRESHOLD_SIGMA]
        pb = pb[pb.score >= config.PICK_THRESHOLD_SIGMA]
        ia, ib, dist = match_one_to_one(pa[["x_A", "y_A", "z_A"]].to_numpy(),
                                        pb[["x_A", "y_A", "z_A"]].to_numpy(),
                                        config.MATCH_RADIUS_A)
        agree_rows.append({
            "tilt_series": series,
            f"n_{a}": len(pa), f"n_{b}": len(pb),
            "n_matched": len(ia),
            "fraction_of_smaller_set": round(len(ia) / max(min(len(pa), len(pb)), 1), 3),
            "median_displacement_A": round(float(np.median(dist)), 2) if len(dist) else np.nan,
        })
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(out_dir / "task1_branch_agreement.csv", index=False)
    print(f"    table task1_branch_agreement.csv")

    # ---- 4. plots ----------------------------------------------------------
    _plot_task1_quality(per_series, branches, out_dir)
    _plot_task1_yield(per_series, picks, branches, out_dir)
    _plot_task1_scores(picks, branches, out_dir)
    _plot_task1_runtime(runtimes, out_dir)

    return {"per_series": per_series, "stats": stats_df, "agreement": agree,
            "residuals": residuals, "runtimes": runtimes, "branches": branches}


def _plot_task1_quality(ps, branches, out_dir):
    """Paired plot: one line per tilt series, connecting its two branch values.

    A paired plot is the honest way to show five samples. A bar chart of two
    means would hide whether the difference is consistent or driven by one
    outlier; here you can see every tilt series individually and whether the
    lines all slope the same way.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(axes, ["sharpness", "contrast"],
                                 ["Tomogram sharpness\n(variance of Laplacian, higher = crisper)",
                                  "Tomogram contrast\n(std / mean |value|, higher = more structure)"]):
        for _, r in ps.iterrows():
            ys = [r[f"{metric}_{b}"] for b in branches]
            ax.plot([0, 1], ys, "-o", color="grey", alpha=0.7, markersize=5)
            ax.annotate(r["tilt_series"], (1.02, ys[1]), fontsize=7, va="center")
        for i, b in enumerate(branches):
            ax.scatter([i] * len(ps), ps[f"{metric}_{b}"], zorder=3, s=60,
                       color=BRANCH_COLOR[b], label=config.BRANCH_LABELS[b])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([config.BRANCH_LABELS[b] for b in branches], fontsize=8)
        ax.set_xlim(-0.4, 1.5)
        ax.set_title(title, fontsize=9)
    finish(fig, out_dir / "task1_tomogram_quality.png", "")


def _plot_task1_yield(ps, picks, branches, out_dir):
    """How many particles each branch yields, and how that depends on strictness.

    The right-hand panel is important: the number of particles you get depends
    entirely on where you put the threshold, so showing a single count invites
    the reader to over-interpret it. The curve shows the whole picture, and if
    one branch is above the other everywhere, that conclusion does not depend on
    the threshold at all.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(ps))
    width = 0.38
    for i, b in enumerate(branches):
        axes[0].bar(x + (i - 0.5) * width, ps[f"n_picks_{b}"], width,
                    color=BRANCH_COLOR[b], label=config.BRANCH_LABELS[b])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ps["tilt_series"], fontsize=8)
    axes[0].set_ylabel(f"particles at >= {config.PICK_THRESHOLD_SIGMA} sigma")
    axes[0].set_title("Particles found per tilt series", fontsize=10)
    axes[0].legend(fontsize=8)

    thresholds = np.linspace(3, 12, 40)
    for b in branches:
        sub = picks[picks.branch == b]["score"].to_numpy()
        counts = [(sub >= t).sum() for t in thresholds]
        axes[1].plot(thresholds, counts, color=BRANCH_COLOR[b],
                     label=config.BRANCH_LABELS[b], linewidth=2)
    axes[1].axvline(config.PICK_THRESHOLD_SIGMA, color="k", ls=":", linewidth=1)
    axes[1].annotate("cutoff used", (config.PICK_THRESHOLD_SIGMA, 0),
                     fontsize=7, rotation=90, va="bottom", xytext=(3, 5),
                     textcoords="offset points")
    axes[1].set_xlabel("score cutoff (sigma above background)")
    axes[1].set_ylabel("particles retained (all 5 series)")
    axes[1].set_title("Yield versus how strict you are", fontsize=10)
    axes[1].legend(fontsize=8)
    finish(fig, out_dir / "task1_particle_yield.png", "")


def _plot_task1_scores(picks, branches, out_dir):
    """Distribution of template-matching scores from both branches.

    These two histograms CAN share an axis, because both were produced by the
    same program with the same template and the same normalisation. The score is
    "how many standard deviations above this volume's own background", which is
    a physically meaningful, tomogram-independent quantity. A distribution
    pushed further right means the molecules stood out more clearly, which means
    the tomogram preserved their structure better, which means the alignment was
    better. This is the plot that answers Task 1.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(3, max(12, float(picks["score"].max())), 60)
    for b in branches:
        s = picks[picks.branch == b]["score"]
        ax.hist(s, bins=bins, alpha=0.55, label=f"{config.BRANCH_LABELS[b]} (n={len(s)})",
                color=BRANCH_COLOR[b])
    ax.axvline(config.PICK_THRESHOLD_SIGMA, color="k", ls=":", linewidth=1)
    ax.set_xlabel("template-matching score (sigma above background)")
    ax.set_ylabel("number of picks")
    ax.set_title("How strongly the molecules stood out, per alignment branch", fontsize=10)
    ax.legend(fontsize=8)
    finish(fig, out_dir / "task1_score_distributions.png", "")


def _plot_task1_runtime(runtimes, out_dir):
    """Wall-clock time per stage. Stacked, so the total cost of each route is
    visible as well as where the time went."""
    if runtimes.empty:
        return
    df = runtimes[runtimes.method.isin(["etomo", "aretomo"])]
    if df.empty:
        return
    stages = list(dict.fromkeys(df["stage"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottoms = {m: 0.0 for m in ("etomo", "aretomo")}
    for stage in stages:
        vals, labels = [], []
        for m in ("etomo", "aretomo"):
            v = df[(df.stage == stage) & (df.method == m)]["seconds"].sum()
            vals.append(v)
            labels.append(m)
        ax.bar(labels, vals, bottom=[bottoms[m] for m in labels], label=stage)
        for m, v in zip(labels, vals):
            bottoms[m] += v
    ax.set_ylabel("wall-clock seconds (5 tilt series, 1 GPU)")
    ax.set_title("Where the time goes, per alignment route", fontsize=10)
    ax.legend(fontsize=7)
    finish(fig, out_dir / "task1_runtime.png", "")


# ===========================================================================
#  SECTION C - TASK 2: does the particle picker matter?
# ===========================================================================

def analyse_picking(tables, out_dir):
    """Compare Warp's picker with PyTom's on identical tomograms."""
    print("\n--- Task 2: particle-picking comparison " + "-" * 31)
    warp = tables["warp_picks"]
    # .copy() because we add a "matched" column below; operating on a slice
    # of the caller's table would be ambiguous.
    warp = warp[(warp.branch == "etomo")
                & (warp.score >= config.PICK_THRESHOLD_SIGMA)].copy()
    pytom = tables["pytom_picks"].copy()
    warp["matched"] = False
    pytom["matched"] = False
    runtimes = tables["runtimes"]

    if warp.empty or pytom.empty:
        print("    One of the two pick sets is empty - nothing to compare.")
        return {}

    tomos = sorted(set(warp.tilt_series) & set(pytom.tilt_series))
    if not tomos:
        print("    No tilt series in common - check the tilt_series names.")
        return {}

    # ---- 1. per-tomogram counts and agreement at the chosen tolerance ------
    rows, matched_pairs = [], []
    for t in tomos:
        w = warp[warp.tilt_series == t].reset_index(drop=True)
        p = pytom[pytom.tilt_series == t].reset_index(drop=True)
        iw, ip, dist = match_one_to_one(w[["x_A", "y_A", "z_A"]].to_numpy(),
                                        p[["x_A", "y_A", "z_A"]].to_numpy(),
                                        config.MATCH_RADIUS_A)
        rows.append({
            "tilt_series": t, "n_warp": len(w), "n_pytom": len(p),
            "n_matched": len(iw),
            "jaccard": round(len(iw) / max(len(w) + len(p) - len(iw), 1), 3),
            "recall_of_warp": round(len(iw) / max(len(w), 1), 3),
            "recall_of_pytom": round(len(iw) / max(len(p), 1), 3),
            "median_displacement_A": round(float(np.median(dist)), 2) if len(dist) else np.nan,
        })
        # Remember which picks matched, so we can look at their scores later.
        w["matched"] = False
        p["matched"] = False
        w.loc[iw, "matched"] = True
        p.loc[ip, "matched"] = True
        matched_pairs.append(pd.DataFrame({
            "tilt_series": t,
            "warp_score": w.loc[iw, "score"].to_numpy(),
            "pytom_score": p.loc[ip, "score"].to_numpy(),
            "displacement_A": dist}))
        warp.loc[warp.tilt_series == t, "matched"] = w["matched"].to_numpy()
        pytom.loc[pytom.tilt_series == t, "matched"] = p["matched"].to_numpy()

    per_tomo = pd.DataFrame(rows)
    per_tomo.to_csv(out_dir / "task2_per_tomogram.csv", index=False)
    print(f"    table task2_per_tomogram.csv")
    pairs = pd.concat(matched_pairs, ignore_index=True) if matched_pairs else pd.DataFrame()

    # ---- 2. how much does the answer depend on the tolerance? --------------
    # The single most manipulable number in this whole analysis is "how close
    # counts as the same particle". Rather than pick one and hope, we sweep it
    # and publish the curve. If the qualitative story is the same across a wide
    # band of tolerances, it is real; if it flips, we say so.
    sweep = []
    for r_a in config.MATCH_RADIUS_SWEEP_A:
        total_m = 0
        for t in tomos:
            w = warp[warp.tilt_series == t]
            p = pytom[pytom.tilt_series == t]
            iw, _, _ = match_one_to_one(w[["x_A", "y_A", "z_A"]].to_numpy(),
                                        p[["x_A", "y_A", "z_A"]].to_numpy(), r_a)
            total_m += len(iw)
        n_w, n_p = len(warp), len(pytom)
        sweep.append({"match_radius_A": r_a,
                      "match_radius_voxels": round(r_a / config.TOMO_ANGPIX, 2),
                      "n_matched": total_m,
                      "jaccard": round(total_m / max(n_w + n_p - total_m, 1), 3),
                      "fraction_of_warp": round(total_m / max(n_w, 1), 3),
                      "fraction_of_pytom": round(total_m / max(n_p, 1), 3)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out_dir / "task2_radius_sweep.csv", index=False)
    print(f"    table task2_radius_sweep.csv")

    # ---- 3. are the unique picks really the weak ones? ---------------------
    # A very common claim is "the picks only one tool finds are its false
    # positives". That is a testable statement, not something to assert. We test
    # it: for each tool, compare the scores of its matched picks against its
    # unmatched ones with a Mann-Whitney U test (which compares rankings, so it
    # does not care that the two tools' scores are on different scales).
    unique_rows = []
    for name, df in [("warp", warp), ("pytom", pytom)]:
        m = df[df["matched"] == True]["score"].to_numpy()
        u = df[df["matched"] == False]["score"].to_numpy()
        row = {"picker": name, "n_matched": len(m), "n_unique": len(u),
               "median_score_matched": round(float(np.median(m)), 4) if len(m) else np.nan,
               "median_score_unique": round(float(np.median(u)), 4) if len(u) else np.nan}
        if len(m) > 5 and len(u) > 5:
            res = stats.mannwhitneyu(m, u, alternative="greater")
            row["p_matched_score_higher"] = round(float(res.pvalue), 6)
            # Common-language effect size: the chance that a randomly chosen
            # matched pick scores higher than a randomly chosen unique one.
            row["prob_matched_beats_unique"] = round(float(res.statistic / (len(m) * len(u))), 3)
        else:
            row["p_matched_score_higher"] = np.nan
            row["prob_matched_beats_unique"] = np.nan
        unique_rows.append(row)
    unique_df = pd.DataFrame(unique_rows)

    # ---- 4. do the two tools agree on WHICH particles are best? ------------
    # Their raw scores mean different things, so we compare rankings with
    # Spearman's correlation. A high value means that where both tools found a
    # particle, they agree about how convincing it is - which is a much stronger
    # form of agreement than merely picking the same coordinates.
    rank_rho = rank_p = np.nan
    if len(pairs) > 10:
        rho, pval = stats.spearmanr(pairs["warp_score"], pairs["pytom_score"])
        rank_rho, rank_p = round(float(rho), 3), float(pval)

    # ---- 5. headline summary ----------------------------------------------
    rt = {r["method"]: r for _, r in runtimes.iterrows()} if not runtimes.empty else {}
    warp_time = runtimes[runtimes.method.str.startswith("warp_etomo")]["seconds"].sum() \
        if not runtimes.empty else np.nan
    pytom_time = runtimes[runtimes.method == "pytom"]["seconds"].sum() \
        if not runtimes.empty else np.nan

    n_w, n_p, n_m = len(warp), len(pytom), int(per_tomo["n_matched"].sum())
    summary = pd.DataFrame([
        {"metric": "particles found by Warp", "value": n_w},
        {"metric": "particles found by PyTom", "value": n_p},
        {"metric": f"agreeing within {config.MATCH_RADIUS_A:.0f} A", "value": n_m},
        {"metric": "Jaccard overlap", "value": round(n_m / max(n_w + n_p - n_m, 1), 3)},
        {"metric": "fraction of Warp picks confirmed by PyTom", "value": round(n_m / max(n_w, 1), 3)},
        {"metric": "fraction of PyTom picks confirmed by Warp", "value": round(n_m / max(n_p, 1), 3)},
        {"metric": "median distance between agreeing picks (A)",
         "value": round(float(pairs["displacement_A"].median()), 2) if len(pairs) else np.nan},
        {"metric": "Spearman rank correlation of scores (matched picks)", "value": rank_rho},
        {"metric": "Warp picking wall-clock (s, 5 tomograms)", "value": round(float(warp_time), 1)},
        {"metric": "PyTom picking wall-clock (s, 5 tomograms)", "value": round(float(pytom_time), 1)},
    ])
    summary.to_csv(out_dir / "task2_summary.csv", index=False)
    unique_df.to_csv(out_dir / "task2_unique_vs_matched.csv", index=False)
    print(f"    table task2_summary.csv, task2_unique_vs_matched.csv")

    # ---- 6. plots ----------------------------------------------------------
    _plot_task2_counts(per_tomo, out_dir)
    _plot_task2_sweep(sweep, out_dir)
    _plot_task2_scores(warp, pytom, out_dir)
    _plot_task2_rank(pairs, rank_rho, out_dir)
    _plot_task2_xy(warp, pytom, tomos[0], out_dir)

    return {"per_tomo": per_tomo, "sweep": sweep, "summary": summary,
            "unique": unique_df, "pairs": pairs, "rank_rho": rank_rho,
            "rank_p": rank_p, "n_warp": n_w, "n_pytom": n_p, "n_matched": n_m,
            "warp_time": warp_time, "pytom_time": pytom_time}


def _plot_task2_counts(per_tomo, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(per_tomo))
    ax.bar(x - 0.27, per_tomo["n_warp"], 0.25, label="Warp", color=PICKER_COLOR["warp"])
    ax.bar(x, per_tomo["n_matched"], 0.25, label="agreeing (both)", color="#9467bd")
    ax.bar(x + 0.27, per_tomo["n_pytom"], 0.25, label="PyTom", color=PICKER_COLOR["pytom"])
    ax.set_xticks(x)
    ax.set_xticklabels(per_tomo["tilt_series"], fontsize=8)
    ax.set_ylabel("particles")
    ax.set_title(f"Picks per tomogram (agreement within {config.MATCH_RADIUS_A:.0f} A)",
                 fontsize=10)
    ax.legend(fontsize=8)
    finish(fig, out_dir / "task2_counts.png", "")


def _plot_task2_sweep(sweep, out_dir):
    """How the measured agreement depends on the tolerance we chose.

    The dotted line marks one particle radius, which is the physically
    defensible operating point: beyond that, two "matching" centres describe
    spheres that barely overlap.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sweep["match_radius_A"], sweep["jaccard"], "-o", color="#9467bd",
            label="Jaccard overlap")
    ax.plot(sweep["match_radius_A"], sweep["fraction_of_warp"], "-s",
            color=PICKER_COLOR["warp"], label="fraction of Warp picks confirmed")
    ax.plot(sweep["match_radius_A"], sweep["fraction_of_pytom"], "-^",
            color=PICKER_COLOR["pytom"], label="fraction of PyTom picks confirmed")
    ax.axvline(config.MATCH_RADIUS_A, color="k", ls=":", linewidth=1)
    ax.annotate(f"one particle radius\n({config.MATCH_RADIUS_A:.0f} A)",
                (config.MATCH_RADIUS_A, 0.05), fontsize=7,
                xytext=(6, 0), textcoords="offset points")
    ax.set_xlabel("how close two picks must be to count as the same particle (A)")
    ax.set_ylabel("agreement")
    ax.set_ylim(0, 1)
    ax.set_title("Sensitivity of the agreement to the matching tolerance", fontsize=10)
    ax.legend(fontsize=8)
    finish(fig, out_dir / "task2_radius_sweep.png", "")


def _plot_task2_scores(warp, pytom, out_dir):
    """Score distributions - in SEPARATE panels, on purpose.

    Warp's score counts standard deviations above background; PyTom's is a
    normalised correlation coefficient. Overlaying them on one axis, or
    rescaling both to 0-1 to make them "comparable", would invent a relationship
    that does not exist. Each panel is split into the picks the other tool
    confirmed and the picks it did not, which is the actual question of
    interest.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, name, df, unit in [
            (axes[0], "Warp", warp, "sigma above background"),
            (axes[1], "PyTom", pytom, "normalised cross-correlation")]:
        m = df[df["matched"] == True]["score"]
        u = df[df["matched"] == False]["score"]
        bins = np.linspace(float(df["score"].min()), float(df["score"].max()), 45)
        ax.hist(m, bins=bins, alpha=0.65, label=f"confirmed by the other tool (n={len(m)})",
                color=PICKER_COLOR[name.lower()])
        ax.hist(u, bins=bins, alpha=0.65, label=f"found by {name} only (n={len(u)})",
                color="grey")
        ax.set_xlabel(f"{name} score ({unit})")
        ax.set_ylabel("number of picks")
        ax.set_title(f"{name}: are the unique picks the weak ones?", fontsize=10)
        ax.legend(fontsize=7)
    finish(fig, out_dir / "task2_score_distributions.png", "")


def _plot_task2_rank(pairs, rho, out_dir):
    if pairs.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pairs["warp_score"], pairs["pytom_score"], s=8, alpha=0.35,
               color="#9467bd")
    ax.set_xlabel("Warp score (sigma above background)")
    ax.set_ylabel("PyTom score (normalised cross-correlation)")
    ax.set_title(f"Do the tools agree on which picks are best?\n"
                 f"Spearman rank correlation = {rho}", fontsize=10)
    finish(fig, out_dir / "task2_rank_agreement.png", "")


def _plot_task2_xy(warp, pytom, tomo, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    w = warp[warp.tilt_series == tomo]
    p = pytom[pytom.tilt_series == tomo]
    ax.scatter(w.x_A, w.y_A, s=22, facecolors="none", edgecolors=PICKER_COLOR["warp"],
               linewidths=0.9, label=f"Warp (n={len(w)})")
    ax.scatter(p.x_A, p.y_A, s=10, marker="x", color=PICKER_COLOR["pytom"],
               linewidths=0.9, label=f"PyTom (n={len(p)})")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Where the picks are, looking down the beam - {tomo}", fontsize=10)
    ax.legend(fontsize=8)
    finish(fig, out_dir / "task2_xy_example.png", "")


# ===========================================================================
#  SECTION D - writing the conclusions, entirely from the numbers above
# ===========================================================================

def write_report(t1, t2, tables, out_dir):
    """Compose results/conclusions.md from the computed results.

    Every claim below is produced from a variable, and the wording is chosen by
    comparing numbers. If the data changed, the text would change with it. This
    is the difference between a report and a template with numbers dropped in.
    """
    L = []
    add = L.append

    add("# Cryo-ET workflow comparison - results and conclusions\n")
    if SYNTHETIC:
        add(f"> **{STAMP}**\n>\n> Every number on this page was generated by "
            "`analyze.py --selftest` to exercise the code. It describes nothing "
            "about any real specimen.\n")
    add("```")
    add(config.describe())
    add("```\n")

    sw = tables.get("software", pd.DataFrame())
    if not sw.empty:
        add("## Software used\n")
        add("| tool | version |")
        add("|---|---|")
        for _, r in sw.iterrows():
            add(f"| {r['tool']} | {r['version']} |")
        add("")

    # ---------------- Task 1 ------------------------------------------------
    add("## Task 1 - does the alignment method change the science?\n")
    if not t1:
        add("_Not enough data: both alignment branches must be processed._\n")
    else:
        a, b = t1["branches"]
        la, lb = config.BRANCH_LABELS[a], config.BRANCH_LABELS[b]
        st = t1["stats"].set_index("metric")

        add(f"Both branches were processed identically apart from the alignment "
            f"step, and both were carried through reconstruction **and** template "
            f"matching, so they can be judged on outcomes rather than on each "
            f"program's own internal error measure.\n")

        add("| what was measured | " + la + " | " + lb + " | difference | consistent? |")
        add("|---|---|---|---|---|")
        for metric in st.index:
            r = st.loc[metric]
            add(f"| {metric} ({r['metric_higher_is']}) | {r[f'mean_{a}']} | "
                f"{r[f'mean_{b}']} | {r['relative_difference_pct']:+.1f}% | "
                f"{r[f'{a}_higher_in']} favour {la} |")
        add("")

        # Decide the headline from the strongest common metric available.
        key = "top50_score" if "top50_score" in st.index else st.index[0]
        r = st.loc[key]
        diff_pct = float(r["relative_difference_pct"])
        wins = str(r[f"{a}_higher_in"])
        n_pairs = int(r["n_pairs"])
        consistent = wins.startswith(f"{n_pairs}/") or wins.startswith("0/")
        winner, loser = (la, lb) if diff_pct > 0 else (lb, la)

        if abs(diff_pct) < 2:
            add(f"**Headline: the two alignments are equivalent for this dataset.** "
                f"On the strongest common metric (`{key}`) they differ by only "
                f"{abs(diff_pct):.1f}%, which is smaller than the spread between "
                f"tilt series. Choose on runtime and convenience, not accuracy.\n")
        elif consistent:
            add(f"**Headline: {winner} is better on this dataset.** On the "
                f"strongest common metric (`{key}`) it leads by {abs(diff_pct):.1f}%, "
                f"and the direction is the same in every one of the {n_pairs} tilt "
                f"series, which is what makes it believable with so few samples. "
                f"With n={n_pairs} a formal significance test cannot reach p<0.05 "
                f"even in the best case, so consistency of direction is the "
                f"evidence, not the p-value.\n")
        else:
            add(f"**Headline: {winner} leads on average ({abs(diff_pct):.1f}% on "
                f"`{key}`) but not in every tilt series ({wins}).** With only "
                f"{n_pairs} series that is weak evidence; treat it as a hint, not "
                f"a finding, and re-run on more data before acting on it.\n")

        ag = t1["agreement"]
        if not ag.empty and ag["median_displacement_A"].notna().any():
            disp = float(ag["median_displacement_A"].median())
            voxels = disp / config.TOMO_ANGPIX
            verdict = ("far smaller than the particle itself, so this is just "
                       "localisation jitter"
                       if disp < config.TEMPLATE_RADIUS_A / 2 else
                       "a noticeable fraction of the particle radius, so the two "
                       "geometries are not interchangeable at the sub-voxel level")
            add(f"Independently of quality, the two branches place the same "
                f"molecules in the same places: they agree on "
                f"{ag['fraction_of_smaller_set'].mean():.0%} of picks on average, "
                f"with a median disagreement of {disp:.1f} A "
                f"({voxels:.1f} voxels of {config.TOMO_ANGPIX:.0f} A) - {verdict}. "
                f"Neither alignment has the geometry grossly wrong.\n")

        rt = t1["runtimes"]
        al = rt[rt.stage == "alignment"] if not rt.empty else pd.DataFrame()
        if len(al) == 2:
            fast = al.loc[al["seconds"].idxmin()]
            slow = al.loc[al["seconds"].idxmax()]
            ratio = slow["seconds"] / max(fast["seconds"], 1e-9)
            add(f"On speed, **{config.BRANCH_LABELS[fast['method']]}** aligned all "
                f"{int(fast['n_tilt_series'])} tilt series in "
                f"{fast['seconds']:.0f} s versus {slow['seconds']:.0f} s "
                f"({ratio:.1f}x) - measured as total wall clock on one GPU with "
                f"{config.PERDEVICE_WORKERS} workers. Warp aligns all series inside "
                f"a single call, so per-series timings were not measured and are "
                f"not claimed.\n")

        res = t1["residuals"]
        add("### A note on the aligners' own residuals\n")
        if res.empty or (res["metric"] == "parse_failed").all() or res["value"].isna().all():
            add("Neither program's log yielded a residual we could parse "
                "confidently, and we would rather report that than guess. It "
                "changes nothing: the comparison above never depended on them.\n")
        else:
            add("| branch | tilt series | metric | value | unit |")
            add("|---|---|---|---|---|")
            for _, r in res.dropna(subset=["value"]).iterrows():
                add(f"| {r['branch']} | {r['tilt_series']} | {r['metric']} | "
                    f"{r['value']} | {r['unit']} |")
            add("")
            add("**These two columns must not be compared with each other.** IMOD "
                "reports the average distance between predicted and observed "
                "positions of tracked patches; AreTomo reports an error from its "
                "own projection-matching objective. They measure different things "
                "on different scales. They are useful for spotting a tilt series "
                "that went wrong within one method, and for nothing else.\n")

    # ---------------- Task 2 ------------------------------------------------
    add("## Task 2 - does the particle picker change the science?\n")
    if not t2:
        add("_Not enough data: both pickers must have produced particles._\n")
    else:
        n_w, n_p, n_m = t2["n_warp"], t2["n_pytom"], t2["n_matched"]
        jac = n_m / max(n_w + n_p - n_m, 1)
        add(f"Both pickers ran on the **same tomograms** (the etomo branch), with "
            f"the same reference map, the same particle diameter and the same "
            f"{config.ANGULAR_STEP_DEG} degree angular step, so the only variable "
            f"is the program.\n")
        add(f"- Warp found **{n_w}** particles; PyTom found **{n_p}**.")
        add(f"- **{n_m}** of them are the same molecule found twice "
            f"(within {config.MATCH_RADIUS_A:.0f} A, one particle radius), "
            f"a Jaccard overlap of **{jac:.2f}**.")
        add(f"- {n_m / max(n_w, 1):.0%} of Warp's picks were confirmed by PyTom; "
            f"{n_m / max(n_p, 1):.0%} of PyTom's were confirmed by Warp.")
        if len(t2["pairs"]):
            add(f"- Where they agree they agree closely: median separation "
                f"{t2['pairs']['displacement_A'].median():.1f} A, i.e. "
                f"{t2['pairs']['displacement_A'].median() / config.TOMO_ANGPIX:.1f} "
                f"voxels.")
        if not np.isnan(t2["rank_rho"]):
            rho = t2["rank_rho"]
            strength = ("strongly" if rho > 0.6 else "moderately" if rho > 0.3
                        else "only weakly")
            add(f"- On the particles both found, their confidence rankings agree "
                f"{strength} (Spearman rho = {rho}). Their raw scores are "
                f"different quantities and are never compared directly.")
        add("")

        add("### Is the count difference meaningful?\n")
        gap = abs(n_w - n_p) / max(n_w, n_p, 1)
        if gap < 0.1:
            add(f"The two totals differ by only {gap:.0%}, so neither tool is "
                f"obviously more or less permissive at these settings.\n")
        else:
            more = "Warp" if n_w > n_p else "PyTom"
            add(f"{more} returned {gap:.0%} more particles. **A higher count is "
                f"not by itself a better result** - a picker can produce more "
                f"picks purely by producing more false positives. Each tool used "
                f"its own native thresholding scheme (Warp: a fixed "
                f"{config.PICK_THRESHOLD_SIGMA} sigma cutoff; PyTom: its automatic "
                f"false-alarm estimate), so part of this gap is a threshold "
                f"choice rather than a detection difference.\n")

        add("### Are the picks each tool finds alone its weakest ones?\n")
        add("This is a claim worth testing rather than assuming, so we tested it.\n")
        add("| picker | median score, confirmed | median score, unique | "
            "chance a confirmed pick outscores a unique one | p |")
        add("|---|---|---|---|---|")
        for _, r in t2["unique"].iterrows():
            prob = r["prob_matched_beats_unique"]
            add(f"| {r['picker']} | {r['median_score_matched']} | "
                f"{r['median_score_unique']} | "
                f"{'n/a' if pd.isna(prob) else f'{prob:.0%}'} | "
                f"{'n/a' if pd.isna(r['p_matched_score_higher']) else r['p_matched_score_higher']} |")
        add("")
        probs = t2["unique"]["prob_matched_beats_unique"].dropna()
        if len(probs) and (probs > 0.65).all():
            add("Confirmed. For both tools the picks the other tool missed are "
                "systematically the lower-scoring ones, which is what you would "
                "expect if they are enriched in false positives.\n")
        elif len(probs) and (probs > 0.55).all():
            add("Partly confirmed: the unique picks do score lower, but the "
                "distributions overlap heavily, so discarding them would throw "
                "away real particles as well as junk.\n")
        elif len(probs):
            add("**Not confirmed.** The unique picks are not reliably the "
                "low-scoring ones, so the common assumption that method-unique "
                "picks are junk does not hold here. Do not throw them away "
                "without looking at them.\n")

        add("### How much does the matching tolerance matter?\n")
        sw = t2["sweep"]
        lo = sw[sw.match_radius_A == sw.match_radius_A.min()].iloc[0]
        hi = sw[sw.match_radius_A == sw.match_radius_A.max()].iloc[0]
        at = sw.iloc[(sw["match_radius_A"] - config.MATCH_RADIUS_A).abs().argmin()]
        add(f"Agreement is not a fixed property of the data - it depends on how "
            f"close two picks have to be before you call them the same molecule. "
            f"Across {lo['match_radius_A']:.0f}-{hi['match_radius_A']:.0f} A the "
            f"Jaccard overlap runs from {lo['jaccard']:.2f} to {hi['jaccard']:.2f}. "
            f"We quote {config.MATCH_RADIUS_A:.0f} A (one apoferritin radius, "
            f"{config.MATCH_RADIUS_A / config.TOMO_ANGPIX:.1f} voxels) because "
            f"beyond that the two spheres barely overlap and calling them the same "
            f"particle stops being physically meaningful. At that point the "
            f"overlap is {at['jaccard']:.2f}. The full curve is in "
            f"`task2_radius_sweep.csv` so a reader can see the whole picture "
            f"rather than one chosen number.\n")

        if not np.isnan(t2["warp_time"]) and not np.isnan(t2["pytom_time"]):
            faster, slower = (("Warp", t2["warp_time"]), ("PyTom", t2["pytom_time"])) \
                if t2["warp_time"] < t2["pytom_time"] else \
                (("PyTom", t2["pytom_time"]), ("Warp", t2["warp_time"]))
            add(f"### Runtime\n")
            add(f"{faster[0]} took {faster[1]:.0f} s and {slower[0]} took "
                f"{slower[1]:.0f} s for the same 5 tomograms ("
                f"{slower[1] / max(faster[1], 1e-9):.1f}x). Both numbers cover "
                f"search plus peak extraction. One caveat that belongs with any "
                f"such number: Warp exploited the full octahedral symmetry of "
                f"apoferritin while PyTom can only use the 4-fold axis about z, "
                f"so PyTom searched roughly six times more orientations for the "
                f"same result. That is a capability difference, not inefficiency.\n")

    # ---------------- Recommendations ---------------------------------------
    add("## Recommendations\n")
    recs = []
    if t1:
        a, b = t1["branches"]
        st = t1["stats"].set_index("metric")
        key = "top50_score" if "top50_score" in st.index else st.index[0]
        diff_pct = float(st.loc[key, "relative_difference_pct"])
        rt = t1["runtimes"]
        al = rt[rt.stage == "alignment"] if not rt.empty else pd.DataFrame()
        fastest = config.BRANCH_LABELS[al.loc[al["seconds"].idxmin(), "method"]] \
            if len(al) == 2 else None
        best = config.BRANCH_LABELS[a] if diff_pct > 0 else config.BRANCH_LABELS[b]
        if abs(diff_pct) < 2 and fastest:
            recs.append(f"**Alignment:** use **{fastest}**. Quality is "
                        f"indistinguishable here ({abs(diff_pct):.1f}% apart), so "
                        f"take the faster route.")
        elif fastest and fastest != best:
            recs.append(f"**Alignment:** **{best}** gives the better particles, "
                        f"**{fastest}** is faster. Use {fastest} for on-the-fly "
                        f"screening while data is being collected, and {best} for "
                        f"the tilt series you intend to average.")
        else:
            recs.append(f"**Alignment:** use **{best}** - it is both better and "
                        f"not slower here.")
    if t2:
        n_w, n_p, n_m = t2["n_warp"], t2["n_pytom"], t2["n_matched"]
        recs.append(f"**Picking:** the {n_m} picks both tools agree on are the "
                    f"safest starting set for subtomogram averaging. Taking the "
                    f"union instead ({n_w + n_p - n_m} picks) trades precision for "
                    f"recall and needs a 3D classification step to clean it up.")
        recs.append("**Thresholds:** calibrate each tool's cutoff separately. "
                    "Warp's score is in standard deviations above background; "
                    "PyTom's is a correlation coefficient. There is no conversion "
                    "between them.")
    recs.append("**Validation:** the strongest test available and not done here "
                "is to run the agreed picks through subtomogram averaging and "
                "compare the resolution reached. That converts every proxy on "
                "this page into the number that actually matters. It is the "
                "obvious next step, and it needs RELION plus M rather than "
                "another analysis script.")
    recs.append("**Sample size:** every comparison here rests on 5 tilt series "
                "from one dataset of one very well-behaved test specimen. Nothing "
                "here should be generalised to thick cellular samples without "
                "repeating it on thick cellular samples.")
    for r in recs:
        add(f"- {r}")
    add("")

    path = out_dir / "conclusions.md"
    path.write_text("\n".join(L))
    print(f"\n    report {path.name}")


# ===========================================================================
#  SECTION E - the self-test: obviously fake data, loudly labelled
# ===========================================================================

def write_selftest_tables(tables_dir):
    """Fabricate tables in the real schema so every code path can be exercised
    on a machine with no GPU.

    This exists to test the SOFTWARE, not to stand in for a result. Everything
    it produces goes into results/selftest/, and every plot and every paragraph
    generated from it carries a red SYNTHETIC banner. It is never mixed with
    real output and is never committed as if it were a finding.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    series = config.TILT_SERIES

    # runtimes
    rt = []
    for stage, m, s in [("alignment", "etomo", 900), ("alignment", "aretomo", 260),
                        ("ctf_estimation", "etomo", 150), ("ctf_estimation", "aretomo", 150),
                        ("reconstruction", "etomo", 320), ("reconstruction", "aretomo", 320),
                        ("template_matching", "warp_etomo", 1400),
                        ("template_matching", "warp_aretomo", 1400),
                        ("peak_extraction", "warp_etomo", 40),
                        ("peak_extraction", "warp_aretomo", 40),
                        ("template_matching", "pytom", 2600),
                        ("peak_extraction", "pytom", 90)]:
        rt.append({"stage": stage, "method": m, "seconds": s, "n_tilt_series": 5,
                   "seconds_per_tilt_series": s / 5, "note": "SYNTHETIC"})
    pd.DataFrame(rt).to_csv(tables_dir / "runtimes.csv", index=False)

    # tomograms + picks
    tomo_rows, warp_rows, pytom_rows = [], [], []
    shape_a = np.array([4400, 6000, 1000]) * config.PIXEL_SIZE_A / 1.0
    shape_a = np.array([4400, 6000, 1000])          # in 10 A voxels -> A below
    for s in series:
        truth = rng.uniform([200, 200, 200], [4200, 5800, 800], (240, 3))
        for branch, quality in [("etomo", 1.0), ("aretomo", 0.93)]:
            tomo_rows.append({"branch": branch, "tilt_series": s,
                              "nx": 440, "ny": 600, "nz": 100,
                              "contrast": round(0.42 * quality + rng.normal(0, 0.01), 5),
                              "sharpness": round(1.8e-3 * quality**3 + rng.normal(0, 3e-5), 6)})
            n = int(len(truth) * (0.92 * quality))
            idx = rng.choice(len(truth), n, replace=False)
            pos = truth[idx] + rng.normal(0, 12, (n, 3))
            sc = np.clip(rng.normal(7.4 * quality, 1.3, n), 5.0, None)
            for (x, y, z), v in zip(pos, sc):
                warp_rows.append({"branch": branch, "tilt_series": s,
                                  "x_A": round(x, 2), "y_A": round(y, 2), "z_A": round(z, 2),
                                  "score": round(v, 4),
                                  "score_units": "sigma_above_background",
                                  "coords_were_normalised": True})
        n = int(len(truth) * 0.88)
        idx = rng.choice(len(truth), n, replace=False)
        pos = truth[idx] + rng.normal(0, 18, (n, 3))
        sc = np.clip(rng.normal(0.33, 0.06, n), 0.15, 1.0)
        extra = rng.uniform([200, 200, 200], [4200, 5800, 800], (35, 3))
        for (x, y, z), v in zip(pos, sc):
            pytom_rows.append({"tilt_series": s, "x_A": round(x, 2), "y_A": round(y, 2),
                               "z_A": round(z, 2), "score": round(v, 5),
                               "score_units": "normalised_cross_correlation",
                               "cutoff": 0.2})
        for x, y, z in extra:
            pytom_rows.append({"tilt_series": s, "x_A": round(x, 2), "y_A": round(y, 2),
                               "z_A": round(z, 2),
                               "score": round(float(np.clip(rng.normal(0.22, 0.03), 0.15, 1)), 5),
                               "score_units": "normalised_cross_correlation",
                               "cutoff": 0.2})

    pd.DataFrame(tomo_rows).to_csv(tables_dir / "tomogram_stats.csv", index=False)
    pd.DataFrame(warp_rows).to_csv(tables_dir / "warp_picks.csv", index=False)
    pd.DataFrame(pytom_rows).to_csv(tables_dir / "pytom_picks.csv", index=False)
    pd.DataFrame([{"branch": "etomo", "tilt_series": s, "metric": "imod_residual_mean",
                   "value": round(0.4 + rng.normal(0, 0.05), 4), "unit": "pixels",
                   "source": "SYNTHETIC", "comparable_across_methods": False}
                  for s in series]
                 + [{"branch": "aretomo", "tilt_series": s, "metric": "aretomo_reported_error",
                     "value": round(1.2 + rng.normal(0, 0.1), 4), "unit": "arbitrary",
                     "source": "SYNTHETIC", "comparable_across_methods": False}
                    for s in series]).to_csv(tables_dir / "native_residuals.csv", index=False)
    pd.DataFrame([{"scope": "SYNTHETIC", "parameter": "-", "warp_or_etomo": "-",
                   "aretomo_or_pytom": "-", "note": "self-test only"}]
                 ).to_csv(tables_dir / "parameters.csv", index=False)
    pd.DataFrame([{"tool": "(self-test)", "found": True, "path": "",
                   "version": "no real software was run"}]
                 ).to_csv(tables_dir / "software_versions.csv", index=False)
    print(f"    wrote synthetic tables to {tables_dir}")


# ===========================================================================
#  main
# ===========================================================================

def main():
    global SYNTHETIC
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the whole analysis on obviously fake data, into "
                         "results/selftest/, with every output stamped SYNTHETIC")
    args = ap.parse_args()

    if args.selftest:
        SYNTHETIC = True
        out_dir = config.RESULTS_DIR / "selftest"
        tables_dir = out_dir / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"SELF-TEST MODE - {STAMP}")
        write_selftest_tables(tables_dir)
    else:
        out_dir = config.RESULTS_DIR
        tables_dir = config.TABLES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading tables from {tables_dir}")
    tables = {
        "tomogram_stats": load("tomogram_stats.csv", tables_dir),
        "warp_picks": load("warp_picks.csv", tables_dir),
        "pytom_picks": load("pytom_picks.csv", tables_dir, required=False),
        "runtimes": load("runtimes.csv", tables_dir, required=False),
        "native_residuals": load("native_residuals.csv", tables_dir, required=False),
        "parameters": load("parameters.csv", tables_dir, required=False),
        "software": load("software_versions.csv", tables_dir, required=False),
    }

    t1 = analyse_alignment(tables, out_dir)
    t2 = analyse_picking(tables, out_dir)
    write_report(t1, t2, tables, out_dir)

    print(f"\nEverything written to {out_dir}")
    print("Next step:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dashboard.py - Task 3: a small interactive report of everything above.

WHAT THIS IS
------------
A Streamlit web page that reads results/ and shows the alignment comparison, the
particle-picking comparison, and the conclusions. Run it with:

    streamlit run dashboard.py

WHAT MAKES IT DIFFERENT FROM A PDF
-----------------------------------
Two things, and they are the two things that make a dashboard worth building
rather than just exporting a document:

  1. Nothing on this page is written down in advance. Every number, every
     verdict and every recommendation comes from the CSV tables that analyze.py
     produced. If tomorrow's data reversed the result, this page would say the
     opposite tomorrow, with no code change. A dashboard that hard-codes its own
     conclusions is a poster with extra steps, and it will eventually tell a
     confident lie.

  2. The reader can move the two knobs that the conclusions are most sensitive
     to - the score cutoff for calling something a particle, and how close two
     picks must be to count as the same molecule - and watch the answer move.
     That is the honest way to present a result that depends on a threshold:
     let the reader see the dependence instead of hiding it behind one number.

There is also a banner at the top that states, unmissably, whether you are
looking at real results or at self-test data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config
from analyze import match_one_to_one

st.set_page_config(page_title="Cryo-ET workflow comparison", layout="wide")


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_results(results_dir: str):
    """Read every CSV in a results folder into a dictionary of DataFrames.

    Cached, so moving a slider does not re-read the disk. Missing files come
    back as empty DataFrames rather than exceptions, so a partially-complete run
    still renders whatever it does have.
    """
    d = Path(results_dir)
    out = {}
    for name in ["task1_summary", "task1_per_tilt_series", "task1_branch_agreement",
                 "task2_summary", "task2_per_tomogram", "task2_radius_sweep",
                 "task2_unique_vs_matched"]:
        f = d / f"{name}.csv"
        out[name] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    for name in ["warp_picks", "pytom_picks", "runtimes", "parameters",
                 "native_residuals", "software_versions", "tomogram_stats"]:
        f = d / "tables" / f"{name}.csv"
        out[name] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    md = d / "conclusions.md"
    out["conclusions"] = md.read_text() if md.exists() else ""
    return out


def pick_results_dir():
    """Let the user choose between the real results and the self-test output,
    and make it impossible to confuse the two."""
    real = config.RESULTS_DIR
    selftest = config.RESULTS_DIR / "selftest"
    choices = {}
    if (real / "conclusions.md").exists():
        choices["Real results"] = real
    if (selftest / "conclusions.md").exists():
        choices["SELF-TEST (synthetic, not a result)"] = selftest
    if not choices:
        st.error("No results found. Run `python analyze.py` first "
                 "(or `python analyze.py --selftest` to see the layout).")
        st.stop()
    label = st.sidebar.radio("Data source", list(choices)) if len(choices) > 1 \
        else list(choices)[0]
    return label, choices[label]


# ---------------------------------------------------------------------------
#  Page
# ---------------------------------------------------------------------------

label, results_dir = pick_results_dir()
R = load_results(str(results_dir))
is_synthetic = "SELF-TEST" in label

st.title("Cryo-ET workflow comparison")

if is_synthetic:
    st.error("**SYNTHETIC SELF-TEST DATA — NOT A SCIENTIFIC RESULT.** "
             "These numbers were fabricated by `analyze.py --selftest` purely to "
             "exercise the code. They describe no real specimen.")
else:
    st.success("Real processing results.")

st.code(config.describe())

with st.sidebar:
    st.header("Provenance")
    if not R["software_versions"].empty:
        for _, r in R["software_versions"].iterrows():
            st.caption(f"**{r['tool']}** — {r['version']}")
    st.divider()
    st.header("Explore the thresholds")
    st.caption("The two numbers the conclusions depend on most. Move them and "
               "watch the answer move.")
    sigma = st.slider("Particle score cutoff (sigma above background)",
                      3.0, 12.0, float(config.PICK_THRESHOLD_SIGMA), 0.25)
    radius = st.slider("How close two picks must be to be the same molecule (A)",
                       10, 200, int(config.MATCH_RADIUS_A), 5)
    st.caption(f"{radius} A = {radius / config.TOMO_ANGPIX:.1f} voxels. "
               f"One apoferritin radius is {config.TEMPLATE_RADIUS_A:.0f} A.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1 · Alignment (etomo vs AreTomo2)",
    "2 · Picking (Warp vs PyTom)",
    "3 · Conclusions",
    "4 · Parameters & provenance"])


# ------------------------------------------------------------------ Tab 1
with tab1:
    st.header("Does the alignment method change the science?")
    st.markdown(
        "Both branches share every processing step except alignment, and both "
        "were carried all the way through template matching. That lets us judge "
        "them on **outcomes** — how sharp the tomograms are and how confidently "
        "molecules can be found in them — rather than on each program's own "
        "internal error number, which are not on comparable scales.")

    if R["task1_summary"].empty:
        st.info("Run `python run_workflow.py --all` then `python analyze.py`.")
    else:
        st.subheader("Comparable metrics, paired across tilt series")
        st.dataframe(R["task1_summary"], width="stretch", hide_index=True)
        st.caption("Each row compares the two branches on the same five tilt "
                   "series. With n=5 the p-value cannot reach 0.05 even in the "
                   "best case — the column that carries the evidence is whether "
                   "the direction is the same in all five.")

        c1, c2 = st.columns(2)
        for col, img, cap in [
                (c1, "task1_tomogram_quality.png",
                 "Every tilt series shown individually; a line per series joins "
                 "its two branch values."),
                (c2, "task1_score_distributions.png",
                 "These two histograms may share an axis: same program, same "
                 "template, same normalisation."),
                (c1, "task1_particle_yield.png",
                 "Right panel: the yield curve, so no conclusion hangs on one "
                 "arbitrary cutoff."),
                (c2, "task1_runtime.png", "Wall clock, one GPU, five tilt series.")]:
            p = results_dir / img
            if p.exists():
                col.image(str(p), caption=cap, width="stretch")

        if not R["task1_branch_agreement"].empty:
            st.subheader("Do the two branches find the same molecules?")
            st.dataframe(R["task1_branch_agreement"], width="stretch",
                         hide_index=True)

        if not R["native_residuals"].empty:
            with st.expander("Each aligner's own residual (diagnostic only — "
                             "read the warning)"):
                st.warning(
                    "**Do not compare these two columns with each other.** IMOD "
                    "reports the mean distance between predicted and observed "
                    "positions of tracked patches. AreTomo reports an error from "
                    "a different projection-matching objective. They measure "
                    "different quantities on different scales, so a smaller "
                    "number does not mean a better alignment. They are useful "
                    "only for spotting a bad tilt series *within* one method.")
                st.dataframe(R["native_residuals"], width="stretch",
                             hide_index=True)


# ------------------------------------------------------------------ Tab 2
with tab2:
    st.header("Does the particle picker change the science?")
    st.markdown(
        "Both pickers ran on the **same tomograms**, with the same reference "
        "map, the same particle diameter and the same angular step. The only "
        "variable is the program.")

    warp = R["warp_picks"]
    pytom = R["pytom_picks"]

    if warp.empty or pytom.empty:
        st.info("Run `python run_workflow.py --all` then `python analyze.py`.")
    else:
        # Recompute live at the slider settings. This is the point of the
        # dashboard: the reader gets to see how much the answer moves.
        w = warp[(warp.branch == "etomo") & (warp.score >= sigma)]
        tomos = sorted(set(w.tilt_series) & set(pytom.tilt_series))
        rows = []
        for t in tomos:
            a = w[w.tilt_series == t][["x_A", "y_A", "z_A"]].to_numpy()
            b = pytom[pytom.tilt_series == t][["x_A", "y_A", "z_A"]].to_numpy()
            ia, _, dist = match_one_to_one(a, b, float(radius))
            rows.append({"tilt_series": t, "n_warp": len(a), "n_pytom": len(b),
                         "n_matched": len(ia),
                         "median_displacement_A": round(float(np.median(dist)), 1)
                         if len(dist) else np.nan})
        live = pd.DataFrame(rows)
        n_w, n_p = int(live.n_warp.sum()), int(live.n_pytom.sum())
        n_m = int(live.n_matched.sum())

        st.subheader(f"At {sigma:g} sigma and a {radius} A tolerance")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Warp picks", n_w)
        k2.metric("PyTom picks", n_p)
        k3.metric("Agree (both)", n_m)
        k4.metric("Jaccard overlap", f"{n_m / max(n_w + n_p - n_m, 1):.2f}")
        st.dataframe(live, width="stretch", hide_index=True)
        st.caption("Recomputed live from the pick tables using the same optimal "
                   "one-to-one matching the report uses — each pick is used at "
                   "most once, and the answer does not depend on file order.")

        st.subheader("Headline numbers at the settings used in the report")
        st.dataframe(R["task2_summary"], width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        for col, img, cap in [
                (c1, "task2_counts.png", "Per-tomogram counts and agreement."),
                (c2, "task2_radius_sweep.png",
                 "How much the agreement depends on the tolerance you choose."),
                (c1, "task2_score_distributions.png",
                 "Separate axes on purpose — the two scores are different "
                 "quantities and rescaling them to a shared 0–1 axis would "
                 "invent a relationship that does not exist."),
                (c2, "task2_rank_agreement.png",
                 "Rankings can be compared even when raw scores cannot."),
                (c1, "task2_xy_example.png", "Where the picks sit in one tomogram.")]:
            p = results_dir / img
            if p.exists():
                col.image(str(p), caption=cap, width="stretch")

        if not R["task2_unique_vs_matched"].empty:
            st.subheader("Are the picks each tool finds alone its weakest ones?")
            st.dataframe(R["task2_unique_vs_matched"], width="stretch",
                         hide_index=True)
            st.caption("A tested claim, not an assumed one. 'Chance a confirmed "
                       "pick outscores a unique one' is 50% if there is no "
                       "relationship at all.")


# ------------------------------------------------------------------ Tab 3
with tab3:
    st.header("Conclusions and recommendations")
    st.caption("Generated by analyze.py from the tables in this results folder. "
               "Nothing here is hard-coded — if the data changed, this text "
               "would change with it.")
    if R["conclusions"]:
        st.markdown(R["conclusions"])
    else:
        st.info("Run `python analyze.py` to generate the report.")


# ------------------------------------------------------------------ Tab 4
with tab4:
    st.header("Exactly what was run")
    st.markdown(
        "Reproducibility is a deliverable, not a nice-to-have. This tab records "
        "the settings each program was given and the version of each program "
        "that produced these numbers.")
    if not R["parameters"].empty:
        st.subheader("Settings, side by side")
        st.dataframe(R["parameters"], width="stretch", hide_index=True)
    if not R["software_versions"].empty:
        st.subheader("Software versions")
        st.dataframe(R["software_versions"], width="stretch", hide_index=True)
    if not R["runtimes"].empty:
        st.subheader("Runtime, by stage")
        st.dataframe(R["runtimes"], width="stretch", hide_index=True)
        st.caption("Warp processes all tilt series inside one call, so alignment "
                   "and matching times are honest totals for five series on one "
                   "GPU; they are not per-series measurements.")
    if not R["tomogram_stats"].empty:
        st.subheader("Reconstructed tomograms")
        st.dataframe(R["tomogram_stats"], width="stretch", hide_index=True)

#!/usr/bin/env python3
"""
run_workflow.py - Tasks 1 and 2: actually run the cryo-ET processing.

WHAT THIS SCRIPT DOES, IN PLAIN LANGUAGE
----------------------------------------
A cryo-electron tomography experiment works like a medical CT scan of a single
frozen protein sample. The microscope photographs the same tiny patch of ice
41 times while tilting it from -40 to +40 degrees. Each photograph is a
shadowgram: everything in the sample squashed flat into one 2D image, and
extremely noisy, because we deliberately use very few electrons so as not to
destroy the sample.

To turn those 41 flat shadows back into a 3D volume you must first work out
exactly where each shadow was taken from - the sample drifts and the stage is
not perfect, so the nominal angles are not good enough. That step is called
ALIGNMENT. Then you back-project the aligned images into a 3D volume, which is
called RECONSTRUCTION, and the volume is called a TOMOGRAM.

Once you have the tomogram you want to find the individual protein molecules
inside it. Because they are buried in noise you cannot just look for them - you
take a known 3D shape of the molecule and slide and rotate it everywhere in the
volume, recording how well it fits at each position and orientation. That is
TEMPLATE MATCHING, and it is how particles get "picked".

This script runs that whole chain twice over, changing exactly one thing each
time, so we can measure what that one change does:

  TASK 1 - two ways of doing ALIGNMENT
      branch "etomo"   : IMOD patch tracking
      branch "aretomo" : AreTomo2
    Everything before alignment (drift correction, defocus estimation) is done
    ONCE and shared, so the branches cannot differ for any other reason. Both
    branches then run all the way through reconstruction AND template matching,
    because the question we actually care about is not "which alignment reports
    a smaller internal error" but "which alignment lets you find better
    particles".

  TASK 2 - two ways of doing PARTICLE PICKING
      Warp's built-in template matching, versus PyTom's.
    Here the tomograms are held fixed (we use the etomo branch for both), so the
    only thing that changes is the picking program.

WHAT COMES OUT
--------------
Not tomograms - those stay on the GPU machine. What comes out is a handful of
small, tidy CSV tables in results/tables/. Those tables are the only thing
analyze.py and the dashboard ever read, so all the plotting and statistics can
be redone in a second on a laptop.

HOW TO RUN IT

    python run_workflow.py --all              # everything, start to finish
    python run_workflow.py --preprocess       # just the shared first stage
    python run_workflow.py --align            # just the two alignment branches
    python run_workflow.py --reconstruct      # CTF + tomograms for both branches
    python run_workflow.py --warp-pick        # Warp template matching, both branches
    python run_workflow.py --pytom-pick       # PyTom template matching
    python run_workflow.py --collect          # rebuild the CSV tables only (fast, no GPU)

Every stage is skipped if its output already exists, so you can stop and resume.
Requires a Linux machine with an NVIDIA GPU. See README.md.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config

# Set by --force: make every stage re-run instead of skipping completed work.
FORCE = False


# ===========================================================================
#  SECTION A - small helpers used by everything below
# ===========================================================================

def run(cmd, label, cwd=None, check=True):
    """Run one external command, print it, time it, and return how long it took.

    Why print the command? Because a scientific result is only reproducible if
    a reader can see the literal command that produced it. Every command this
    script issues is echoed to the terminal and, for the timed steps, recorded
    in results/tables/runtimes.csv.
    """
    printable = " ".join(str(c) for c in cmd)
    print(f"\n>>> [{label}]\n    {printable}\n", flush=True)

    start = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd,
                          capture_output=True, text=True)
    elapsed = time.time() - start

    # Keep the program's own output next to the data. When something goes wrong
    # six weeks later, this log is the only evidence of what happened.
    log_dir = config.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.log").write_text(
        f"$ {printable}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        msg = f"[{label}] failed with exit code {proc.returncode}"
        if check:
            sys.exit(msg)
        print("WARNING: " + msg)
    else:
        print(f"    done in {elapsed:.0f} s")
    return elapsed, proc.stdout + proc.stderr


def warp(subcommand, *args, label=None, settings=True, check=True):
    """Build and run a `WarpTools <subcommand> ...` call.

    All Warp calls go through here so that the settings file is never forgotten
    and every call is logged the same way.
    """
    cmd = ["WarpTools", subcommand]
    if settings:
        cmd += ["--settings", config.TILTSERIES_SETTINGS]
    cmd += list(args)
    return run(cmd, label or subcommand, cwd=config.DATA_DIR, check=check)


def record(rows, path, fieldnames):
    """Write a list of dictionaries to a CSV table. This is the only way any
    measurement leaves this script."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {path.name}  ({len(rows)} rows)")


def append_runtime(stage, method, seconds, n_series, note=""):
    """Runtimes accumulate across stages, so they go into a JSON scratch file
    and are turned into a CSV at the end by collect_tables()."""
    f = config.TABLES_DIR / "_runtimes.json"
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(f.read_text()) if f.exists() else []
    data = [d for d in data if not (d["stage"] == stage and d["method"] == method)]
    data.append({"stage": stage, "method": method, "seconds": round(seconds, 1),
                 "n_tilt_series": n_series,
                 "seconds_per_tilt_series": round(seconds / max(n_series, 1), 1),
                 "note": note})
    f.write_text(json.dumps(data, indent=2))


# ===========================================================================
#  SECTION B - STAGE 1: shared preprocessing (done once, used by both branches)
# ===========================================================================

def stage_preprocess():
    """Turn raw movies into drift-corrected images with known defocus, and group
    them into tilt series.

    Four steps:

    1. create_settings (frame series)
       Writes a small config file telling Warp where the movies are, how big a
       pixel is, and which gain reference to divide by.

    2. fs_motion_and_ctf
       "fs" = frame series. Two jobs at once:
       - MOTION: the sample creeps under the electron beam during the ~1 second
         exposure, so the frames of each movie are blurred relative to each
         other. Warp measures that creep and shifts the frames back into
         register before averaging them. Without this you lose all fine detail.
       - CTF: an electron microscope does not form a clean image; it applies an
         oscillating transfer function that depends on how far out of focus you
         are. To recover the true structure you must know the defocus of every
         image. Warp measures it from the Thon rings in the power spectrum.
       The grids "1x1x3" and "2x2x1" say how finely each model varies: motion is
       allowed to change over time (3 samples) but not across the image, and
       defocus is allowed to vary 2x2 across the image but not over time.

    3. ts_import
       Reads the .mdoc metadata and works out which of the 205 movies belong to
       which of the 5 tilt series, and at what angle and accumulated dose each
       was taken. It writes one .tomostar file per tilt series.

    4. create_settings (tilt series)
       A second config file, this time for the tilt-series stage, which also
       records how big the final tomograms should be.
    """
    print("\n" + "=" * 72)
    print("STAGE 1 - shared preprocessing (motion, CTF, tilt-series grouping)")
    print("=" * 72)

    gain_args = ["--gain_path", config.GAIN_FILE]
    if config.GAIN_FLIP_Y:
        gain_args.append("--gain_flip_y")

    if not config.FRAMESERIES_SETTINGS.exists():
        run(["WarpTools", "create_settings",
             "--folder_data", "frames",
             "--folder_processing", "warp_frameseries",
             "--output", "warp_frameseries.settings",
             "--extension", "*.tif",
             "--angpix", config.PIXEL_SIZE_A,
             *gain_args,
             "--exposure", config.DOSE_PER_TILT],
            "create_settings_frameseries", cwd=config.DATA_DIR)

    # Warp writes one XML of results per movie, named after the movie. Compare
    # those NAMES against the movies actually present, rather than counting
    # against an expected total: the frames folder may hold more movies than the
    # five tilt series need (a wildcard download leaves extras), and a run that
    # was interrupted part-way leaves a partial set. Counting alone cannot tell
    # "finished" from "stopped after enough files to look finished".
    movies = {m.stem for m in config.FRAMES_DIR.glob("*.tif")} \
        if config.FRAMES_DIR.exists() else set()
    fs_dir = config.DATA_DIR / "warp_frameseries"
    processed = {x.stem for x in fs_dir.glob("*.xml")} if fs_dir.exists() else set()
    outstanding = movies - processed

    if outstanding or FORCE:
        print(f"    {len(processed)}/{len(movies)} movies already done, "
              f"{len(outstanding)} to go")
        print("    (Warp skips movies it has already processed, so this resumes)")
        cmd = ["WarpTools", "fs_motion_and_ctf",
               "--settings", "warp_frameseries.settings",
               "--m_grid", "1x1x3",
               "--c_grid", "2x2x1",
               "--c_range_max", 7,
               "--c_defocus_max", 8,
               "--c_use_sum",
               "--out_averages",
               "--perdevice", config.PERDEVICE_WORKERS]
        # Half-set averages are only the input for Noise2Noise denoising, which
        # this project deliberately does not do (see config.WRITE_HALF_AVERAGES).
        # They cost ~19 GB and nothing downstream reads them.
        if config.WRITE_HALF_AVERAGES:
            cmd.insert(-2, "--out_average_halves")
        secs, _ = run(cmd, "fs_motion_and_ctf", cwd=config.DATA_DIR)
        append_runtime("frame_series_motion_ctf", "shared", secs,
                       len(config.TILT_SERIES), "shared by both branches")
    elif movies:
        print(f"    all {len(movies)} movies already processed - skipping")
    else:
        sys.exit(f"No movies found in {config.FRAMES_DIR}. "
                 f"Run: python setup_data.py --download")

    if FORCE or not config.TOMOSTAR_DIR.exists() \
            or not list(config.TOMOSTAR_DIR.glob("*.tomostar")):
        # --dont_invert sets the geometric handedness of the reconstruction.
        # It is a property of this microscope + acquisition software, and the
        # tutorial tells us this is the right choice for EMPIAR-10491.
        run(["WarpTools", "ts_import",
             "--mdocs", "mdoc",
             "--frameseries", "warp_frameseries",
             "--tilt_exposure", config.DOSE_PER_TILT,
             "--min_intensity", 0.3,
             "--dont_invert",
             "--output", "tomostar"],
            "ts_import", cwd=config.DATA_DIR)

    if not config.TILTSERIES_SETTINGS.exists():
        run(["WarpTools", "create_settings",
             "--output", "warp_tiltseries.settings",
             "--folder_processing", "warp_tiltseries",
             "--folder_data", "tomostar",
             "--extension", "*.tomostar",
             "--angpix", config.PIXEL_SIZE_A,
             *gain_args,
             "--exposure", config.DOSE_PER_TILT,
             "--tomo_dimensions", config.TOMO_DIMENSIONS],
            "create_settings_tiltseries", cwd=config.DATA_DIR)


# ===========================================================================
#  SECTION C - STAGE 2: the two alignment branches (Task 1)
# ===========================================================================

def stage_align():
    """Run both alignment methods on the identical preprocessed data.

    The key mechanism here is Warp's --output_processing flag. Normally Warp
    writes its tilt-series results into the one folder named in the settings
    file. --output_processing redirects the output somewhere else, which lets
    two alignment methods run from the same inputs into two separate folders
    without ever touching each other. This is exactly the use case Warp's own
    documentation recommends it for.

    Everything downstream then reads from a chosen branch with
    --input_processing.

    Note on the runtime numbers: Warp aligns all five tilt series inside one
    command, so what we can honestly measure is the total wall-clock time for
    five series, on one GPU, with the same number of worker processes. We record
    that, plus the per-series average, and we do NOT pretend to per-series
    timings we did not measure.
    """
    print("\n" + "=" * 72)
    print("STAGE 2 - alignment, two ways (Task 1)")
    print("=" * 72)

    # --- Branch A: IMOD patch tracking -------------------------------------
    # Chop each tilt image into 500 A squares, follow each square from one tilt
    # to the next by cross-correlation, and fit a projection model that explains
    # all of those tracks at once.
    if not _branch_aligned("etomo"):
        secs, _ = warp("ts_etomo_patches",
                       "--angpix", config.TOMO_ANGPIX,
                       "--patch_size", config.ETOMO_PATCH_SIZE_A,
                       "--initial_axis", config.INITIAL_TILT_AXIS,
                       "--perdevice", config.PERDEVICE_WORKERS,
                       "--output_processing", config.BRANCHES["etomo"],
                       label="align_etomo_patches")
        append_runtime("alignment", "etomo", secs, len(config.TILT_SERIES),
                       f"{config.PERDEVICE_WORKERS} workers, one GPU, all 5 series")
    else:
        print("    etomo branch already aligned - skipping")

    # --- Branch B: AreTomo2 -------------------------------------------------
    # No patches and no markers. AreTomo reconstructs a rough tomogram,
    # re-projects it back to 2D, compares with the real images, corrects the
    # geometry, and repeats. --alignz tells it how thick the sample is so it
    # knows what volume to reconstruct while iterating.
    if not _branch_aligned("aretomo"):
        secs, _ = warp("ts_aretomo",
                       "--angpix", config.TOMO_ANGPIX,
                       "--alignz", config.ARETOMO_ALIGN_Z,
                       "--axis_iter", config.ARETOMO_AXIS_ITER,
                       "--min_fov", config.ARETOMO_MIN_FOV,
                       "--perdevice", config.PERDEVICE_WORKERS,
                       "--output_processing", config.BRANCHES["aretomo"],
                       label="align_aretomo")
        append_runtime("alignment", "aretomo", secs, len(config.TILT_SERIES),
                       f"{config.PERDEVICE_WORKERS} workers, one GPU, all 5 series")
    else:
        print("    aretomo branch already aligned - skipping")


def _branch_aligned(branch):
    """A branch counts as aligned once Warp has written one XML per tilt series."""
    d = config.branch_dir(branch)
    return d.exists() and len(list(d.glob("*.xml"))) >= len(config.TILT_SERIES)


# ===========================================================================
#  SECTION D - STAGE 3: CTF refinement and tomogram reconstruction
# ===========================================================================

def stage_reconstruct():
    """For each alignment branch: fix the defocus handedness, refine the CTF
    using the now-known geometry, and reconstruct the 3D tomograms.

    Three steps per branch:

    1. ts_defocus_hand --check
       There is a sign ambiguity in how defocus changes across a tilted sample:
       is the left side of the image closer to focus, or the right? Getting it
       backwards quietly halves your resolution. Warp measures the correlation
       between predicted and observed defocus gradients. A POSITIVE correlation
       means the current setting is right; a NEGATIVE one means it must be
       flipped. We read the answer out of Warp's own output and act on it,
       rather than assuming - the two official Warp examples disagree about
       which way this dataset goes, so assuming would be a coin flip.

    2. ts_ctf
       Re-estimates the defocus of every tilt image, this time using the
       constraint that all images in a series share one physical geometry. Much
       more accurate than the per-image estimate from stage 1.

    3. ts_reconstruct
       Back-projects the aligned, CTF-corrected images into a 3D volume at
       10 A/pixel.
    """
    print("\n" + "=" * 72)
    print("STAGE 3 - defocus handedness, CTF, reconstruction (both branches)")
    print("=" * 72)

    hand_rows = []
    for branch in config.BRANCHES:
        proc_dir = config.BRANCHES[branch]
        print(f"\n--- branch: {branch} ({config.BRANCH_LABELS[branch]}) ---")

        # 1. handedness check, then conditional correction
        _, out = warp("ts_defocus_hand", "--input_processing", proc_dir, "--check",
                      label=f"defocus_hand_check_{branch}", check=False)
        corr = _parse_handedness(out)
        needs_flip = corr is not None and corr < 0
        if needs_flip:
            print(f"    correlation {corr:+.3f} is negative -> applying --set_flip")
            warp("ts_defocus_hand", "--input_processing", proc_dir, "--set_flip",
                 label=f"defocus_hand_flip_{branch}")
        else:
            print(f"    correlation {corr if corr is not None else 'n/a'} "
                  f"-> no flip needed")
        hand_rows.append({"branch": branch, "correlation": corr,
                          "flip_applied": needs_flip})

        # 2. tilt-series CTF estimation
        secs, _ = warp("ts_ctf",
                       "--input_processing", proc_dir,
                       "--range_high", 7,
                       "--defocus_max", 8,
                       "--perdevice", config.PERDEVICE_WORKERS,
                       label=f"ts_ctf_{branch}")
        append_runtime("ctf_estimation", branch, secs, len(config.TILT_SERIES))

        # 3. reconstruction
        recon_dir = config.branch_dir(branch) / "reconstruction"
        if recon_dir.exists() and len(list(recon_dir.glob("*.mrc"))) >= len(config.TILT_SERIES):
            print("    tomograms already reconstructed - skipping")
            continue
        secs, _ = warp("ts_reconstruct",
                       "--input_processing", proc_dir,
                       "--angpix", config.TOMO_ANGPIX,
                       "--perdevice", config.PERDEVICE_WORKERS,
                       label=f"ts_reconstruct_{branch}")
        append_runtime("reconstruction", branch, secs, len(config.TILT_SERIES))

    record(hand_rows, config.TABLES_DIR / "defocus_handedness.csv",
           ["branch", "correlation", "flip_applied"])


def _parse_handedness(text):
    """Pull the average correlation out of ts_defocus_hand's output.

    Warp prints a line like 'Average correlation: 0.932'. We look for that
    number; if the wording changes in a future version we return None and the
    caller reports 'n/a' instead of silently guessing.
    """
    m = re.search(r"[Aa]verage correlation[:\s]+(-?\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    if re.search(r"should be set to 'flip'", text):
        return -1.0
    if re.search(r"should be set to 'no flip'", text):
        return 1.0
    return None


# ===========================================================================
#  SECTION E - STAGE 4: Warp template matching on BOTH branches
# ===========================================================================

def stage_warp_pick():
    """Find apoferritin molecules in both sets of tomograms using Warp.

    Running the same picker on both alignment branches is what makes Task 1
    answerable. An alignment error blurs the tomogram; a blurred tomogram gives
    lower, broader correlation peaks; so the height of the peaks is a direct,
    common-currency measure of alignment quality. Comparing the two aligners'
    own internally-reported residuals would NOT be valid, because IMOD and
    AreTomo minimise different quantities - their numbers are not on the same
    scale and are not commensurable.

    Two steps per branch:

    1. ts_template_match
       Downloads EMD-15854, scales it to 10 A/pixel, and correlates it against
       the tomogram at every position and every orientation on a 7.5 degree
       grid, exploiting the molecule's octahedral symmetry to avoid searching
       orientations that are equivalent. --whiten boosts the high-resolution
       part of the template so that the huge low-resolution signal does not
       dominate the score. The result is a "correlation volume": the same shape
       as the tomogram, with a score at every voxel.
       Scores are expressed in standard deviations above the volume's own
       background, which is what makes them comparable between tomograms.

    2. threshold_picks
       Finds the local maxima in the correlation volume that are above the
       cutoff, and writes them out as a list of particle positions.
    """
    print("\n" + "=" * 72)
    print("STAGE 4 - Warp template matching on both branches (Tasks 1 and 2)")
    print("=" * 72)

    for branch in config.BRANCHES:
        proc_dir = config.BRANCHES[branch]
        matching = config.branch_dir(branch) / "matching"
        print(f"\n--- branch: {branch} ---")

        if not (matching.exists() and list(matching.glob(f"*{config.TEMPLATE_EMDB_ID}.star"))):
            secs, _ = warp("ts_template_match",
                           "--input_processing", proc_dir,
                           "--tomo_angpix", config.TOMO_ANGPIX,
                           "--subdivisions", config.WARP_SUBDIVISIONS,
                           "--template_emdb", config.TEMPLATE_EMDB_ID,
                           "--template_diameter", config.TEMPLATE_DIAMETER_A,
                           "--symmetry", config.TEMPLATE_SYMMETRY,
                           "--whiten",
                           "--perdevice", 1,
                           label=f"template_match_warp_{branch}")
            append_runtime("template_matching", f"warp_{branch}", secs,
                           len(config.TILT_SERIES),
                           "search only; peak extraction timed separately")
        else:
            print("    correlation volumes already present - skipping search")

        # Peak extraction is cheap but we time it anyway, so that the Warp and
        # PyTom totals cover the same set of operations (search + extraction).
        secs, _ = warp("threshold_picks",
                       "--input_processing", proc_dir,
                       "--in_suffix", config.TEMPLATE_EMDB_ID,
                       "--out_suffix", "clean",
                       "--minimum", config.PICK_THRESHOLD_SIGMA,
                       label=f"threshold_picks_{branch}")
        append_runtime("peak_extraction", f"warp_{branch}", secs,
                       len(config.TILT_SERIES))


# ===========================================================================
#  SECTION F - STAGE 5: PyTom template matching (Task 2)
# ===========================================================================

def stage_pytom_pick():
    """Find the same molecules in the SAME tomograms using PyTom instead.

    This is the controlled experiment for Task 2: identical tomograms (we use
    the etomo branch), identical reference map (EMD-15854), identical particle
    diameter, and an angular search set explicitly to 7.5 degrees so that both
    programs are searching the same number of orientations. The only thing that
    differs is the program.

    Three steps:

    1. pytom_create_template.py
       Warp fetches and prepares the template internally; PyTom needs it as a
       file. This resamples the deposited 1 A/voxel map down to 10 A/voxel to
       match the tomograms.

    2. pytom_create_mask.py
       A soft-edged sphere the size of the molecule. During matching, only what
       is inside the mask contributes to the correlation, so surrounding noise
       does not dilute the score. The radius is given in VOXELS, so a 65 A
       radius at 10 A/voxel is 6.5 voxels.

    3. pytom_match_template.py, then pytom_extract_candidates.py
       Search, then pull the peaks out. We feed PyTom the Warp XML for each
       tilt series so that it uses exactly the same tilt angles, accumulated
       dose and defocus values that Warp used - another confound removed.
    """
    print("\n" + "=" * 72)
    print("STAGE 5 - PyTom template matching (Task 2)")
    print("=" * 72)

    branch = "etomo"                      # tomograms held constant for Task 2
    recon_dir = config.branch_dir(branch) / "reconstruction"
    tomograms = sorted(recon_dir.glob("*.mrc"))
    if not tomograms:
        sys.exit(f"No tomograms in {recon_dir}. Run --reconstruct first.")

    config.PYTOM_DIR.mkdir(parents=True, exist_ok=True)
    template = config.PYTOM_DIR / "template_10A.mrc"
    mask = config.PYTOM_DIR / "mask_10A.mrc"

    # --- 1. template -------------------------------------------------------
    if not template.exists():
        cmd = ["pytom_create_template.py",
               "-i", config.TEMPLATE_MAP,
               "-o", template,
               "--input-voxel-size-angstrom", config.TEMPLATE_VOXEL_SIZE_A,
               "--output-voxel-size-angstrom", config.TOMO_ANGPIX,
               "--center"]
        if config.PYTOM_INVERT_TEMPLATE:
            cmd.append("--invert")
        run(cmd, "pytom_create_template")
        print("    NOTE: check the contrast matches. In Warp's tomogram preview\n"
              "    images the protein should look BRIGHT. If it looks dark, set\n"
              "    PYTOM_INVERT_TEMPLATE = True in config.py and re-run.")

    # --- 2. mask -----------------------------------------------------------
    if not mask.exists():
        # The box must be large enough to hold the molecule plus a margin; a
        # box of 2.5x the particle diameter (rounded to an even number) is the
        # usual rule of thumb.
        box = int(round(2.5 * config.TEMPLATE_DIAMETER_A / config.TOMO_ANGPIX / 2) * 2)
        radius_voxels = config.TEMPLATE_RADIUS_A / config.TOMO_ANGPIX
        run(["pytom_create_mask.py",
             "-b", box,
             "-o", mask,
             "--voxel-size", config.TOMO_ANGPIX,
             "-r", round(radius_voxels, 2),
             "-s", 1.0],
            "pytom_create_mask")

    # --- 3. match + extract, one tomogram at a time -------------------------
    extract_exe = _pytom_extract_executable()
    total_search = total_extract = 0.0
    per_tomo = []

    for tomo in tomograms:
        name = _series_name_from_path(tomo)

        def find_job():
            """PyTom names its job file after the tomogram, but the exact suffix
            has changed between versions. Look for whatever it actually wrote
            rather than hard-coding one spelling."""
            hits = sorted(config.PYTOM_DIR.glob(f"{tomo.stem}*job.json"))
            return hits[0] if hits else None

        job_json = find_job()
        if job_json is None:
            cmd = ["pytom_match_template.py",
                   "-t", template, "-m", mask, "-v", tomo,
                   "-d", config.PYTOM_DIR,
                   "--particle-diameter", config.TEMPLATE_DIAMETER_A,
                   "--angular-search", config.ANGULAR_STEP_DEG,
                   "--z-axis-rotational-symmetry", config.PYTOM_Z_SYMMETRY,
                   "--voxel-size-angstrom", config.TOMO_ANGPIX,
                   "--low-pass", config.PYTOM_LOW_PASS_A,
                   "--per-tilt-weighting",
                   "--spectral-whitening",
                   "--random-phase-correction",
                   "-g", *[str(g) for g in config.GPU_IDS]]

            # Preferred: hand PyTom the Warp metadata directly, so tilt angles,
            # dose and defocus are guaranteed identical between the two pickers.
            xml = config.branch_dir(branch) / f"{name}.xml"
            secs = 0.0
            if xml.exists():
                secs, out = run(cmd + ["--warp-xml-file", xml],
                                f"pytom_match_{name}", check=False)
            # Fallback for PyTom builds without --warp-xml-file: write plain
            # tilt-angle and dose text files out of the .tomostar and use those.
            job_json = find_job()
            if job_json is None:
                print("    --warp-xml-file route did not produce a job file; "
                      "falling back to explicit tilt/dose files")
                tlt, dose = _write_tilt_and_dose_files(name)
                secs, _ = run(cmd + ["--tilt-angles", tlt,
                                     "--dose-accumulation", dose],
                              f"pytom_match_{name}_fallback")
                job_json = find_job()
                if job_json is None:
                    sys.exit(f"PyTom produced no job file for {name}; see "
                             f"{config.DATA_DIR}/logs/pytom_match_{name}*.log")
            total_search += secs
        else:
            print(f"    {name}: search already done - skipping")

        secs, _ = run([extract_exe,
                       "-j", job_json,
                       "-n", config.PYTOM_MAX_PARTICLES,
                       "--particle-diameter", config.TEMPLATE_DIAMETER_A],
                      f"pytom_extract_{name}")
        total_extract += secs
        per_tomo.append(name)

    append_runtime("template_matching", "pytom", total_search, len(per_tomo),
                   "sum over tomograms, one GPU")
    append_runtime("peak_extraction", "pytom", total_extract, len(per_tomo))


def _pytom_extract_executable():
    """PyTom renamed this entry point between versions (candidates -> candidate).
    Rather than pinning one spelling and breaking on the other, find whichever
    is installed."""
    for name in ("pytom_extract_candidates.py", "pytom_extract_candidate.py"):
        if shutil.which(name):
            return name
    sys.exit("Neither pytom_extract_candidates.py nor pytom_extract_candidate.py "
             "is on PATH - is the pytom environment active?")


def _write_tilt_and_dose_files(series):
    """Extract tilt angles and accumulated dose from Warp's .tomostar file.

    A .tomostar is a small STAR-format table with one row per tilt image. The
    columns we need are _wrpAngleTilt (the stage angle) and _wrpDose (how much
    total radiation the sample had absorbed by that image). PyTom wants those as
    two plain one-number-per-line text files.
    """
    import starfile
    df = starfile.read(config.TOMOSTAR_DIR / f"{series}.tomostar")
    if isinstance(df, dict):
        df = next(iter(df.values()))
    df = df.sort_values("wrpAngleTilt")

    tlt = config.PYTOM_DIR / f"{series}.rawtlt"
    dose = config.PYTOM_DIR / f"{series}.dose"
    tlt.write_text("\n".join(f"{v:.2f}" for v in df["wrpAngleTilt"]) + "\n")
    dose.write_text("\n".join(f"{v:.2f}" for v in df["wrpDose"]) + "\n")
    return tlt, dose


def _series_name_from_path(path):
    """Warp names tomograms like 'TS_11_10.00Apx.mrc'. Recover 'TS_11'."""
    m = re.match(r"(TS_\d+)", path.stem)
    return m.group(1) if m else path.stem


# ===========================================================================
#  SECTION G - STAGE 6: turn everything into tidy CSV tables
# ===========================================================================

def collect_tables():
    """Gather every measurement into small CSVs. No GPU needed for this stage.

    Six tables come out of here:

      runtimes.csv            how long each stage took, per method
      tomogram_stats.csv      image-quality measures of each reconstructed volume
      warp_picks.csv          every particle Warp found, in both branches
      pytom_picks.csv         every particle PyTom found
      native_residuals.csv    each aligner's own error estimate (DIAGNOSTIC ONLY)
      parameters.csv          the settings each program was run with

    Coordinates in the two pick tables are converted to a single common unit -
    Angstroms from the corner of the tomogram - so that they can be compared
    directly. This matters more than it sounds: Warp writes coordinates
    NORMALISED to the range 0-1, while PyTom writes them in VOXELS. Comparing
    the two raw columns without converting would put every Warp particle within
    one voxel of the origin and produce a completely meaningless answer.
    """
    print("\n" + "=" * 72)
    print("STAGE 6 - collecting results into tidy tables")
    print("=" * 72)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    _collect_runtimes()
    _collect_tomogram_stats()
    _collect_warp_picks()
    _collect_pytom_picks()
    _collect_native_residuals()
    _collect_parameters()
    print(f"\nAll tables in {config.TABLES_DIR}")


def _collect_runtimes():
    f = config.TABLES_DIR / "_runtimes.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    record(rows, config.TABLES_DIR / "runtimes.csv",
           ["stage", "method", "seconds", "n_tilt_series",
            "seconds_per_tilt_series", "note"])


def _collect_tomogram_stats():
    """Measure the quality of each reconstructed tomogram.

    Both branches are reconstructed by the same code at the same pixel size, so
    these numbers ARE comparable between branches - unlike the aligners' own
    internal residuals.

    Two measures:

      contrast   = standard deviation of the voxel values in the central slab,
                   divided by the mean absolute value. A well-aligned tomogram
                   has crisp particles and therefore more spread in its values;
                   a misaligned one is smeared towards uniform grey.

      sharpness  = variance of the Laplacian of the central slice. The Laplacian
                   responds to edges, so its variance is large when edges are
                   crisp and small when they are blurred. This is the standard
                   autofocus metric from photography, used here for the same
                   reason: it measures blur without needing to know the answer.
    """
    import mrcfile
    from scipy import ndimage

    rows = []
    for branch in config.BRANCHES:
        recon = config.branch_dir(branch) / "reconstruction"
        for tomo in sorted(recon.glob("*.mrc")) if recon.exists() else []:
            name = _series_name_from_path(tomo)
            with mrcfile.mmap(tomo, permissive=True) as mrc:
                vol = mrc.data
                nz, ny, nx = vol.shape
                # Use the middle fifth of the volume in Z: that is where the
                # sample is, away from the empty ice above and below.
                z0, z1 = int(nz * 0.4), int(nz * 0.6)
                slab = np.asarray(vol[z0:z1], dtype=np.float32)
                mid = np.asarray(vol[nz // 2], dtype=np.float32)

            contrast = float(slab.std() / (np.abs(slab).mean() + 1e-9))
            sharpness = float(ndimage.laplace(mid).var())
            rows.append({"branch": branch, "tilt_series": name,
                         "nx": nx, "ny": ny, "nz": nz,
                         "contrast": round(contrast, 5),
                         "sharpness": round(sharpness, 6)})
    record(rows, config.TABLES_DIR / "tomogram_stats.csv",
           ["branch", "tilt_series", "nx", "ny", "nz", "contrast", "sharpness"])


def _tomogram_shape(branch, series):
    """Voxel dimensions (nx, ny, nz) of one reconstructed tomogram, read from the
    MRC header rather than assumed."""
    import mrcfile
    recon = config.branch_dir(branch) / "reconstruction"
    hits = list(recon.glob(f"{series}_*.mrc")) if recon.exists() else []
    if not hits:
        return None
    with mrcfile.mmap(hits[0], permissive=True) as mrc:
        nz, ny, nx = mrc.data.shape
    return nx, ny, nz


def _need_columns(df, columns, source):
    """Stop with a clear message if a STAR file lacks a column we depend on.

    Column names occasionally change between versions of these tools. Failing
    here with the file name and the missing column is far better than silently
    producing a table of NaNs that then quietly poisons every statistic.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(f"{source} is missing column(s) {missing}.\n"
                 f"Columns present: {list(df.columns)}\n"
                 f"The tool version may write different names - update "
                 f"_collect_warp_picks/_collect_pytom_picks in run_workflow.py.")


def _collect_warp_picks():
    """Read Warp's particle lists and convert them to Angstroms.

    Warp's threshold_picks writes one STAR file per tilt series, named like
    TS_1_10.00Apx_emd_15854_clean.star, with columns:

      rlnCoordinateX/Y/Z          position, NORMALISED to 0-1 across the volume
      rlnAngleRot/Tilt/Psi        the orientation the template best fitted in
      rlnAutopickFigureOfMerit    the correlation score, in units of standard
                                  deviations above the volume's background

    We multiply the normalised coordinates by the volume's real dimensions and
    then by the pixel size to land in Angstroms. The code checks the range first
    and passes values through unchanged if a future Warp version starts writing
    plain voxels instead, so it cannot silently mangle the numbers either way.
    """
    import starfile
    rows = []
    for branch in config.BRANCHES:
        matching = config.branch_dir(branch) / "matching"
        for star in sorted(matching.glob("*_clean.star")) if matching.exists() else []:
            series = _series_name_from_path(star)
            shape = _tomogram_shape(branch, series)
            if shape is None:
                print(f"    WARNING: no tomogram found for {series} in {branch}; skipped")
                continue
            df = starfile.read(star)
            if isinstance(df, dict):
                df = next(iter(df.values()))
            if len(df) == 0:
                continue
            _need_columns(df, ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
                               "rlnAutopickFigureOfMerit"], star.name)

            xyz = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
            normalised = xyz.max() <= 1.5      # 0-1 range means normalised
            scale = np.array(shape, dtype=float) if normalised else np.ones(3)
            xyz_a = xyz * scale * config.TOMO_ANGPIX

            score = df["rlnAutopickFigureOfMerit"].to_numpy(float)
            for (x, y, z), s in zip(xyz_a, score):
                rows.append({"branch": branch, "tilt_series": series,
                             "x_A": round(float(x), 2), "y_A": round(float(y), 2),
                             "z_A": round(float(z), 2), "score": round(float(s), 4),
                             "score_units": "sigma_above_background",
                             "coords_were_normalised": normalised})
    record(rows, config.TABLES_DIR / "warp_picks.csv",
           ["branch", "tilt_series", "x_A", "y_A", "z_A", "score",
            "score_units", "coords_were_normalised"])


def _collect_pytom_picks():
    """Read PyTom's particle lists and convert them to Angstroms.

    PyTom writes one STAR file per tomogram with columns:

      rlnCoordinateX/Y/Z   position in VOXELS (not normalised)
      rlnLCCmax            the correlation score - this is PyTom's score column,
                           NOT rlnAutopickFigureOfMerit
      rlnCutOff            the threshold PyTom chose for this tomogram
      rlnAngleRot/Tilt/Psi the best-fitting orientation

    PyTom's score is a locally-normalised cross-correlation coefficient, roughly
    on a 0-1 scale. Warp's score is in units of standard deviations above
    background. They are NOT the same quantity and must never be plotted on a
    shared axis as if they were - the analysis keeps them in separate panels and
    compares their RANKINGS instead of their raw values.
    """
    import starfile
    rows = []
    if not config.PYTOM_DIR.exists():
        record([], config.TABLES_DIR / "pytom_picks.csv",
               ["tilt_series", "x_A", "y_A", "z_A", "score", "score_units", "cutoff"])
        return

    for star in sorted(config.PYTOM_DIR.glob("*particles.star")):
        series = _series_name_from_path(star)
        df = starfile.read(star)
        if isinstance(df, dict):
            df = next(iter(df.values()))
        if len(df) == 0:
            continue
        _need_columns(df, ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
                           "rlnLCCmax"], star.name)
        xyz = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
        xyz_a = xyz * config.TOMO_ANGPIX
        score = df["rlnLCCmax"].to_numpy(float)
        cutoff = df["rlnCutOff"].to_numpy(float) if "rlnCutOff" in df else np.full(len(df), np.nan)
        for (x, y, z), s, c in zip(xyz_a, score, cutoff):
            rows.append({"tilt_series": series,
                         "x_A": round(float(x), 2), "y_A": round(float(y), 2),
                         "z_A": round(float(z), 2), "score": round(float(s), 5),
                         "score_units": "normalised_cross_correlation",
                         "cutoff": round(float(c), 5)})
    record(rows, config.TABLES_DIR / "pytom_picks.csv",
           ["tilt_series", "x_A", "y_A", "z_A", "score", "score_units", "cutoff"])


def _collect_native_residuals():
    """Collect each aligner's own reported error - as a DIAGNOSTIC, not a
    comparison.

    IMOD's tiltalign reports the root-mean-square distance, in pixels, between
    where it predicted a tracked patch should appear and where it actually
    appeared. AreTomo reports an error from its own projection-matching
    objective. These two numbers are computed from different quantities by
    different algorithms; a smaller IMOD residual does not mean a better
    alignment than a larger AreTomo one, any more than a lower golf score beats
    a higher basketball score.

    So we record them, clearly labelled with which program produced them, and
    the analysis reports them per method WITHOUT declaring a winner. The winner
    is decided on the common metrics instead.

    If a log cannot be parsed we write an explicit 'parse_failed' row. An empty
    file that quietly looks like "no error" is the worst possible outcome here.
    """
    rows = []
    for branch in config.BRANCHES:
        d = config.branch_dir(branch)
        logs = list(d.glob("tiltstack/*/taSolution.log")) if branch == "etomo" \
            else list(d.glob("tiltstack/*/*.log"))
        if not logs:
            rows.append({"branch": branch, "tilt_series": "", "metric": "",
                         "value": "", "unit": "", "source": "NO LOG FILES FOUND",
                         "comparable_across_methods": False})
            continue
        for log in sorted(logs):
            series = log.parent.name
            text = log.read_text(errors="ignore")
            found = _parse_residual_summary(text, branch)
            if found is None:
                rows.append({"branch": branch, "tilt_series": series,
                             "metric": "parse_failed", "value": "", "unit": "",
                             "source": str(log.relative_to(config.DATA_DIR)),
                             "comparable_across_methods": False})
            else:
                metric, value, unit = found
                rows.append({"branch": branch, "tilt_series": series,
                             "metric": metric, "value": round(value, 4), "unit": unit,
                             "source": str(log.relative_to(config.DATA_DIR)),
                             "comparable_across_methods": False})
    record(rows, config.TABLES_DIR / "native_residuals.csv",
           ["branch", "tilt_series", "metric", "value", "unit", "source",
            "comparable_across_methods"])


def _parse_residual_summary(text, branch):
    """Find the one summary error number a log reports, if it reports one.

    We deliberately look only for an explicit, labelled summary line, never for
    'any numeric row with five columns' - that kind of loose pattern will
    happily parse a completely different table and hand back a confident wrong
    answer.
    """
    if branch == "etomo":
        # IMOD tiltalign prints e.g.
        #   "Residual error mean and sd:   0.42   0.19"
        m = re.search(r"Residual error mean and sd:\s+(\d+\.?\d*)", text)
        if m:
            return "imod_residual_mean", float(m.group(1)), "pixels"
        m = re.search(r"Residual error weighted mean:\s+(\d+\.?\d*)", text)
        if m:
            return "imod_residual_weighted_mean", float(m.group(1)), "pixels"
    else:
        # AreTomo prints e.g. "Rotation align: ... error = 1.23"
        m = re.search(r"[Ee]rror\s*[=:]\s*(\d+\.?\d*)", text)
        if m:
            return "aretomo_reported_error", float(m.group(1)), "arbitrary"
    return None


def _collect_parameters():
    """Write down, side by side, exactly what each program was told to do.

    The assessment asks for 'key parameters' as one of the comparison metrics,
    and it is also the honest way to present the runtime numbers: a picker that
    searches more orientations is slower for a reason, not because it is worse
    software.
    """
    rows = [
        {"scope": "environment", "parameter": "CUDA runtime",
         "warp_or_etomo": "12.9 (Warp 2.0.0dev38+ build target)",
         "aretomo_or_pytom": "AreTomo2 Cuda12x binary, borrowing the same "
                             "runtime; PyTom/CuPy on CUDA 12",
         "note": "one conda environment, one CUDA runtime; the host supplies "
                 "only the driver"},
        {"scope": "shared", "parameter": "dataset", "warp_or_etomo": f"EMPIAR-{config.EMPIAR_ACCESSION}",
         "aretomo_or_pytom": f"EMPIAR-{config.EMPIAR_ACCESSION}",
         "note": "apoferritin, 5 tilt series"},
        {"scope": "shared", "parameter": "raw pixel size (A)", "warp_or_etomo": config.PIXEL_SIZE_A,
         "aretomo_or_pytom": config.PIXEL_SIZE_A, "note": ""},
        {"scope": "shared", "parameter": "processing pixel size (A)", "warp_or_etomo": config.TOMO_ANGPIX,
         "aretomo_or_pytom": config.TOMO_ANGPIX, "note": "tomograms and matching"},
        {"scope": "task1 alignment", "parameter": "algorithm", "warp_or_etomo": "IMOD patch tracking",
         "aretomo_or_pytom": "AreTomo2 projection matching", "note": ""},
        {"scope": "task1 alignment", "parameter": "patch size (A) / alignz (px)",
         "warp_or_etomo": config.ETOMO_PATCH_SIZE_A, "aretomo_or_pytom": config.ARETOMO_ALIGN_Z,
         "note": "not the same kind of parameter; each is its method's main setting"},
        {"scope": "task1 alignment", "parameter": "tilt-axis handling",
         "warp_or_etomo": f"seeded at {config.INITIAL_TILT_AXIS} deg, refined",
         "aretomo_or_pytom": f"{config.ARETOMO_AXIS_ITER} refinement iterations", "note": ""},
        {"scope": "task2 picking", "parameter": "template", "warp_or_etomo": f"EMD-{config.TEMPLATE_EMDB_ID}",
         "aretomo_or_pytom": f"EMD-{config.TEMPLATE_EMDB_ID}", "note": "identical source map"},
        {"scope": "task2 picking", "parameter": "particle diameter (A)",
         "warp_or_etomo": config.TEMPLATE_DIAMETER_A, "aretomo_or_pytom": config.TEMPLATE_DIAMETER_A,
         "note": "apoferritin"},
        {"scope": "task2 picking", "parameter": "angular step (deg)",
         "warp_or_etomo": f"{config.ANGULAR_STEP_DEG} (subdivisions={config.WARP_SUBDIVISIONS})",
         "aretomo_or_pytom": config.ANGULAR_STEP_DEG, "note": "matched deliberately"},
        {"scope": "task2 picking", "parameter": "symmetry used",
         "warp_or_etomo": f"{config.TEMPLATE_SYMMETRY} (full octahedral, 24-fold)",
         "aretomo_or_pytom": f"C{config.PYTOM_Z_SYMMETRY} about z",
         "note": "NOT equivalent - PyTom only supports z-axis symmetry, so it "
                 "searches ~6x more orientations. Affects runtime, not accuracy."},
        {"scope": "task2 picking", "parameter": "spectral whitening",
         "warp_or_etomo": "on (--whiten)", "aretomo_or_pytom": "on (--spectral-whitening)", "note": ""},
        {"scope": "task2 picking", "parameter": "score definition",
         "warp_or_etomo": "sigma above volume background",
         "aretomo_or_pytom": "normalised cross-correlation (LCCmax)",
         "note": "different quantities - compared by ranking, never by raw value"},
        {"scope": "task2 picking", "parameter": "peak cutoff",
         "warp_or_etomo": f"{config.PICK_THRESHOLD_SIGMA} sigma",
         "aretomo_or_pytom": "PyTom automatic false-alarm estimate",
         "note": "each tool's native thresholding scheme"},
    ]
    record(rows, config.TABLES_DIR / "parameters.csv",
           ["scope", "parameter", "warp_or_etomo", "aretomo_or_pytom", "note"])


# ===========================================================================
#  main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every stage in order")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--reconstruct", action="store_true")
    ap.add_argument("--warp-pick", action="store_true")
    ap.add_argument("--pytom-pick", action="store_true")
    ap.add_argument("--collect", action="store_true",
                    help="rebuild the CSV tables from existing outputs (no GPU)")
    ap.add_argument("--force", action="store_true",
                    help="redo a stage even if its output looks complete")
    args = ap.parse_args()

    stages = [args.preprocess, args.align, args.reconstruct,
              args.warp_pick, args.pytom_pick, args.collect]
    if not (args.all or any(stages)):
        ap.error("choose --all or one or more individual stages")

    global FORCE
    FORCE = args.force
    print(config.describe())
    print(f"\nData directory: {config.DATA_DIR}")
    if FORCE:
        print("--force: stages will re-run even if their output looks complete")

    if args.all or args.preprocess:
        stage_preprocess()
    if args.all or args.align:
        stage_align()
    if args.all or args.reconstruct:
        stage_reconstruct()
    if args.all or args.warp_pick:
        stage_warp_pick()
    if args.all or args.pytom_pick:
        stage_pytom_pick()
    if args.all or args.collect:
        collect_tables()
        print("\nNext step:  python analyze.py")


if __name__ == "__main__":
    main()

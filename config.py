"""
config.py - every setting for this project lives here, and nowhere else.

WHY THIS FILE EXISTS
--------------------
In a scientific pipeline the single biggest source of wrong answers is a number
that appears in two places and disagrees with itself. So every number that
describes the microscope, the sample, or a program setting is written down
exactly once, right here. Every other script imports from this file.

If you want to process a different dataset, you change this file and nothing
else.

WHERE THESE NUMBERS COME FROM
-----------------------------
All of them are taken from the official Warp tilt-series tutorial, which is the
workflow the assessment asks us to reproduce:
  https://warpem.github.io/warp/user_guide/warptools/quick_start_warptools_tilt_series/
and its companion end-to-end script:
  https://github.com/warpem/warp/blob/main/scripts/EMPIAR-10491_5TS_e2e.sh
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. WHERE THINGS LIVE ON DISK
# ---------------------------------------------------------------------------
# PROJECT_DIR is this folder (the one holding the Python scripts).
PROJECT_DIR = Path(__file__).resolve().parent

# DATA_DIR is where the raw microscope data gets downloaded to and where all
# the heavy processing happens. It is deliberately OUTSIDE the git repository,
# because it will grow to several hundred gigabytes.
#
# Override it with the CRYOET_DATA_DIR environment variable, e.g.
#     export CRYOET_DATA_DIR=/mnt/disks/cryoet/empiar10491
import os
DATA_DIR = Path(os.environ.get("CRYOET_DATA_DIR", PROJECT_DIR.parent / "cryoet_data"))

# RESULTS_DIR holds the small, human-readable outputs: tidy CSV tables, plots,
# and the written conclusions. This IS committed to git - it is the deliverable.
RESULTS_DIR = PROJECT_DIR / "results"

# TABLES_DIR holds the "tidy tables": one CSV per measurement, produced by
# run_workflow.py and consumed by analyze.py. Keeping the measuring step and
# the analysing step separate means you can re-do all the plots and statistics
# in one second on a laptop, without touching the GPU machine again.
TABLES_DIR = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# 2. THE DATASET
# ---------------------------------------------------------------------------
# EMPIAR is a public archive of raw electron microscopy data.
# EMPIAR-10491 is the dataset the Warp tilt-series tutorial uses: apoferritin
# (a small, very stable, very symmetric iron-storage protein) imaged as tilt
# series. Apoferritin is the standard "test specimen" of cryo-EM - everybody
# knows what the right answer looks like, which makes it ideal for benchmarking.
EMPIAR_ACCESSION = "10491"
EMPIAR_FTP = "ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data"

# The five tilt series the tutorial uses. Warp names them TS_1 ... TS_32 from
# the .mdoc filenames; the movie files on the server are named ...-1_..., etc.
TILT_SERIES = ["TS_1", "TS_11", "TS_17", "TS_23", "TS_32"]
SERIES_NUMBERS = [1, 11, 17, 23, 32]


# ---------------------------------------------------------------------------
# 3. MICROSCOPE / ACQUISITION PARAMETERS
# ---------------------------------------------------------------------------
# These describe the physical experiment. Getting any of them wrong silently
# corrupts everything downstream, which is why they are not scattered around.

# Size of one camera pixel, measured at the specimen, in Angstroms (1 A = 0.1 nm).
# 0.7894 A/pixel is a very fine sampling - that is why a ~3 A structure is
# achievable from this data.
PIXEL_SIZE_A = 0.7894

# Electron dose delivered per tilt image, in electrons per square Angstrom.
# Warp uses this to keep track of accumulated radiation damage and to
# down-weight the high-resolution information in the later (more damaged) tilts.
DOSE_PER_TILT = 2.64

# The size of the 3D volume Warp will reconstruct, in UNBINNED pixels (X x Y x Z).
# X and Y come from the detector; Z is how thick the ice slab is. The tilt axis
# runs along Y after alignment, which is why Y is the larger number here.
TOMO_DIMENSIONS = "4400x6000x1000"

# The gain reference is a calibration image of the camera. Every raw movie must
# be divided by it. This camera's gain reference needs flipping in Y to match
# the movie orientation - that is what --gain_flip_y does.
GAIN_FILE = "gain_ref.mrc"
GAIN_FLIP_Y = True

# Nominal starting guess for the tilt-axis angle, in degrees. The specimen
# rotates about this axis inside the microscope. IMOD refines it from this seed.
INITIAL_TILT_AXIS = -85.6

# Should Warp also write "half-set" averages - two extra images per movie, made
# from alternating halves of the frames?
#
# The tutorial turns these on (--out_average_halves) because they are the input
# for Noise2Noise denoising, which produces prettier tomograms for looking at.
# This project never denoises: denoising would change the tomograms in ways that
# could differ between the two alignment branches, which is exactly the confound
# we are trying to avoid, and template matching works on the raw reconstruction.
#
# They cost about 19 GB for this dataset and nothing here reads them, so they
# are off by default. Set this to True to reproduce the tutorial byte for byte,
# or if you want to denoise afterwards - and budget the extra disk.
WRITE_HALF_AVERAGES = False


# ---------------------------------------------------------------------------
# 4. PROCESSING RESOLUTION
# ---------------------------------------------------------------------------
# Alignment and particle picking do not need the full 0.7894 A/pixel detail -
# that would be enormously slow and the extra detail is buried in noise anyway.
# We work at 10 A/pixel, i.e. binned about 12.7x. This is the tutorial's choice.
TOMO_ANGPIX = 10.0


# ---------------------------------------------------------------------------
# 5. THE TWO ALIGNMENT METHODS BEING COMPARED (Task 1)
# ---------------------------------------------------------------------------
# Both are run through Warp so that everything except the alignment algorithm
# itself is held identical.

# Method A: IMOD's "etomo" patch tracking. It chops each image into square
# patches and cross-correlates them between neighbouring tilts to work out how
# the specimen moved.
ETOMO_PATCH_SIZE_A = 500     # side length of a tracking patch, in Angstroms

# Method B: AreTomo2. It aligns the whole tilt series at once by
# reconstructing, re-projecting, and iterating - no patches, no fiducials.
ARETOMO_ALIGN_Z = 800        # thickness (unbinned px) AreTomo assumes for the sample
ARETOMO_AXIS_ITER = 5        # how many times it refines the tilt-axis angle
ARETOMO_MIN_FOV = 0.0        # 0 = keep every tilt, do not drop any for small overlap

# Names of the two output folders. Warp writes each alignment into its own
# folder so the two branches never overwrite each other.
BRANCHES = {
    "etomo": "warp_tiltseries_etomo",
    "aretomo": "warp_tiltseries_aretomo",
}
BRANCH_LABELS = {"etomo": "IMOD etomo patch tracking", "aretomo": "AreTomo2"}


# ---------------------------------------------------------------------------
# 6. THE TEMPLATE USED FOR PARTICLE PICKING (Tasks 1 and 2)
# ---------------------------------------------------------------------------
# "Template matching" means: take a known 3D shape, slide and rotate it through
# the tomogram, and record where it fits well. The known shape here is EMD-15854.
#
# IMPORTANT: EMD-15854 is MOUSE HEAVY-CHAIN APOFERRITIN, about 130 A across,
# with octahedral (O) symmetry. It is NOT a ribosome. Getting this wrong throws
# off the search box, the mask, the angular search and the match tolerance.
TEMPLATE_EMDB_ID = 15854
TEMPLATE_DIAMETER_A = 130.0        # outer diameter of apoferritin
TEMPLATE_RADIUS_A = TEMPLATE_DIAMETER_A / 2.0
TEMPLATE_SYMMETRY = "O"            # octahedral: 24 rotations map the shell onto itself
# Voxel size of the deposited EMD-15854 map, in Angstroms.
# None means "read it from the map's own header", which is the right choice:
# EMDB maps are deposited at whatever sampling the authors used, and it is rarely
# a round number - EMD-15854 is 0.729 A/voxel. Assuming 1.0 rescales the template
# by 1/0.729 = 1.37x, turning a 130 A shell into a 178 A one, and template
# matching then finds nothing at all because it is looking for the wrong size of
# object. Set a number here only to override a map with a wrong or absent header.
TEMPLATE_VOXEL_SIZE_A = None

# Angular search fineness. Warp expresses this as "subdivisions":
#   2 -> 15 deg step, 3 -> 7.5 deg, 4 -> 3.75 deg.
# We use 3, like the tutorial, and give PyTom the equivalent 7.5 deg explicitly
# so the two pickers are searching the same number of orientations.
WARP_SUBDIVISIONS = 3
ANGULAR_STEP_DEG = 7.5

# --- two thresholds, on purpose -------------------------------------------
#
# EXTRACT is how permissive we are when writing particles to disk. ANALYSE is
# the cutoff the comparison actually applies. They are separate because
# extraction is destructive: a particle not written out cannot be recovered
# without redoing an hour of matching, whereas raising a cutoff during analysis
# costs nothing. So extract generously and decide later.
#
# Warp normalises its correlation scores to "standard deviations above this
# volume's background", so these numbers are in sigma. The tutorial page uses 3
# and the end-to-end script uses 6.
EXTRACT_THRESHOLD_SIGMA = 3.0

# Where the analysis draws the line, and why 4.5.
#
# The tempting calculation is: a correlation volume holds 12.9 million voxels,
# so the largest value pure noise would produce is sqrt(2*ln(12.9e6)) = 5.7
# sigma, and anything below that is noise. That is WRONG here, because it
# assumes every voxel is an independent sample. A correlation volume is smooth -
# neighbouring voxels are strongly correlated - so the effective number of
# independent samples is closer to (volume / particle volume) = about 11,000,
# giving a realistic noise ceiling of sqrt(2*ln(11185)) = 4.3 sigma.
#
# 4.5 sits just above that. analyze.py also plots yield against cutoff from 3 to
# 12, so a reader can see immediately whether any conclusion depends on it.
PICK_THRESHOLD_SIGMA = 4.5

# Maximum number of particles PyTom is allowed to extract per tomogram.
# Set generously so that PyTom is not the one truncating the comparison.
PYTOM_MAX_PARTICLES = 1500

# How many false positives PyTom should tolerate when it picks its own cutoff.
# PyTom does not take a sigma threshold; it estimates a cutoff from an extreme-
# value model of the background, targeting this many spurious detections per
# tomogram. The default of 1 is very strict - on this data it chose a cutoff
# above every peak in the volume and extracted nothing at all. 100 is the
# matching decision to EXTRACT_THRESHOLD_SIGMA above: let plausible candidates
# through, then let the analysis decide. The false-positive rate is a property
# of the extraction, not a claim about the particles.
PYTOM_FALSE_POSITIVES = 100

# Apoferritin's octahedral symmetry contains a 4-fold rotation axis. PyTom can
# only exploit symmetry about the z axis, and EMD-15854 is deposited with its
# 4-fold along z, so we can legitimately tell PyTom "4". This is NOT identical
# to Warp's full octahedral symmetry - see the parameter table in the report.
PYTOM_Z_SYMMETRY = 4

# Resolution limit for PyTom's matching, in Angstroms. At 10 A/pixel the finest
# detail the tomogram can possibly hold is 20 A (the Nyquist limit), so there is
# nothing to gain by searching finer than that.
PYTOM_LOW_PASS_A = 20.0

# Warp reconstructs tomograms with the contrast inverted, so protein appears
# BRIGHT. EMDB maps are also bright-on-dark. So the PyTom template does NOT
# need inverting. If your preview images show dark particles instead, flip this
# to True. (run_workflow.py prints a reminder about how to check.)
PYTOM_INVERT_TEMPLATE = False


# ---------------------------------------------------------------------------
# 7. HOW WE DECIDE TWO PICKS ARE "THE SAME PARTICLE" (Task 2)
# ---------------------------------------------------------------------------
# If Warp says a particle is at position P and PyTom says it is at Q, how close
# do P and Q have to be before we call them the same particle?
#
# The physically sensible answer is "within about one particle radius" - if the
# two centres are further apart than that, the two spheres barely overlap and
# they are more likely two different objects.
#
# Apoferritin has a 130 A diameter, so its RADIUS is 65 A. At 10 A/pixel that is
# 6.5 voxels. (The previous version of this project used 13 voxels and called it
# "one ribosome radius" - that was a diameter, for the wrong molecule.)
MATCH_RADIUS_A = TEMPLATE_RADIUS_A          # 65 A

# We never rely on a single tolerance. analyze.py sweeps this whole range and
# plots how the answer changes, so the reader can see whether the conclusion is
# robust or an artefact of the cutoff.
MATCH_RADIUS_SWEEP_A = [10, 20, 30, 40, 50, 65, 80, 100, 130, 160, 200]


# ---------------------------------------------------------------------------
# 8. HARDWARE
# ---------------------------------------------------------------------------
# How many worker processes Warp runs per GPU. Warp's algorithms were written
# for ~16 GB cards. An NVIDIA L4 has 24 GB, so 2 workers is comfortable;
# an A100 (80 GB) can take 4.
PERDEVICE_WORKERS = int(os.environ.get("CRYOET_PERDEVICE", "2"))

# Which GPU PyTom should use (PyTom takes explicit GPU indices).
GPU_IDS = [int(g) for g in os.environ.get("CRYOET_GPUS", "0").split(",")]


# ---------------------------------------------------------------------------
# 9. DERIVED PATHS (do not edit - computed from the settings above)
# ---------------------------------------------------------------------------
FRAMES_DIR = DATA_DIR / "frames"
MDOC_DIR = DATA_DIR / "mdoc"
GAIN_PATH = DATA_DIR / GAIN_FILE
TEMPLATE_MAP = DATA_DIR / f"emd_{TEMPLATE_EMDB_ID}.map"

FRAMESERIES_SETTINGS = DATA_DIR / "warp_frameseries.settings"
TILTSERIES_SETTINGS = DATA_DIR / "warp_tiltseries.settings"
TOMOSTAR_DIR = DATA_DIR / "tomostar"

PYTOM_DIR = DATA_DIR / "pytom_picks"


def branch_dir(branch: str) -> Path:
    """Absolute path of one alignment branch's processing folder."""
    return DATA_DIR / BRANCHES[branch]


def describe() -> str:
    """A short human-readable summary, printed at the top of every run and
    embedded in the report so a reader always knows what produced a number."""
    return (
        f"Dataset      : EMPIAR-{EMPIAR_ACCESSION} (apoferritin), "
        f"{len(TILT_SERIES)} tilt series: {', '.join(TILT_SERIES)}\n"
        f"Pixel size   : {PIXEL_SIZE_A} A/px raw, {TOMO_ANGPIX} A/px for processing\n"
        f"Dose         : {DOSE_PER_TILT} e-/A^2 per tilt\n"
        f"Template     : EMD-{TEMPLATE_EMDB_ID}, {TEMPLATE_DIAMETER_A:.0f} A diameter, "
        f"symmetry {TEMPLATE_SYMMETRY}\n"
        f"Pick cutoff  : extract at {EXTRACT_THRESHOLD_SIGMA} sigma, "
        f"analyse at {PICK_THRESHOLD_SIGMA} sigma\n"
        f"Match radius : {MATCH_RADIUS_A:.0f} A "
        f"({MATCH_RADIUS_A / TOMO_ANGPIX:.1f} voxels at {TOMO_ANGPIX} A/px)\n"
        f"Half averages: {'yes (tutorial default)' if WRITE_HALF_AVERAGES else 'no (saves ~19 GB; only needed for denoising)'}"
    )

# Cryo-ET workflow comparison

[`Background material`](https://claude.ai/code/artifact/20a34c73-4009-4834-ae57-0bdb29a0a0fa)


Comparing two **tilt-series alignment** methods and two **particle-picking**
methods on the official Warp tilt-series tutorial dataset, with a reproducible
Python analysis and a dashboard.

Five Python files, no shell scripts.

| file | what it is |
|---|---|
| `config.py` | every setting in the project, in one place |
| `setup_data.py` | Task 0 — check the software, download the data |
| `run_workflow.py` | Tasks 1 & 2 — run the cryo-ET processing (needs a GPU) |
| `analyze.py` | Tasks 1 & 2 — statistics, plots, conclusions (no GPU) |
| `dashboard.py` | Task 3 — interactive summary |

---

> **New to cryo-ET?** [`EXPLAINER.md`](EXPLAINER.md) teaches the whole field
> from zero — what a tomogram is, what IMOD and AreTomo do, why any of this is
> hard — with diagrams. Read that first and this document will make sense.

## Part 1 — What this is actually about, in plain language

Skip this if you already know cryo-ET.

### The experiment

You want to know the exact 3D shape of a protein molecule. You freeze a droplet
of water containing millions of copies so fast that ice crystals never form, and
put it in an electron microscope.

The microscope can only make **2D shadow pictures**: everything in the sample,
squashed flat into one image. To get 3D back, you take pictures at many angles —
exactly like a hospital CT scan — by tilting the sample from −40° through 0° to
+40°, photographing it 41 times.

There's a catch that shapes everything else: electrons destroy the very thing
you're photographing. So you use the smallest dose you can get away with, and
every one of those 41 images is *extraordinarily* noisy. Looking at one, you
cannot see the molecules at all.

### The four steps

1. **Motion correction.** Each "picture" is really a short movie, and the sample
   creeps under the beam while it's exposed. Measure the creep, shift the frames
   back into register, then average them. Skip this and all fine detail is lost.

2. **Alignment.** You know roughly what angle the stage was at, but not
   precisely enough — the stage wobbles, the sample drifts, the tilt axis isn't
   exactly where you think. Alignment works out, from the images themselves,
   exactly where each of the 41 pictures was taken from.
   **This is comparison #1.** Two programs do it in completely different ways:

   - **IMOD's patch tracking** cuts each image into small squares and follows
     each square from one tilt to the next, then fits a geometry that explains
     all the tracks.
   - **AreTomo2** never tracks anything. It builds a rough 3D volume,
     re-projects it back to 2D, compares that with the real pictures, corrects
     the geometry, and repeats.

3. **Reconstruction.** Smear all 41 aligned pictures back through space and add
   them up. Where they reinforce, there's material. The 3D volume you get is a
   **tomogram**. If step 2 was even slightly wrong, everything in the tomogram
   is blurred.

4. **Particle picking.** Now find the individual molecules inside the tomogram.
   They're still buried in noise, so you can't just look. Instead you take a
   *known* 3D shape of the molecule — a **template** — and slide and rotate it
   everywhere in the volume, scoring how well it fits at each position and
   orientation. High-scoring spots are your molecules. This is **3D template
   matching**, and it's comparison #2: **Warp's** built-in matcher versus
   **PyTom's**.

### The dataset and the molecule

- **EMPIAR-10491** — five tilt series of **apoferritin**, the dataset the Warp
  tutorial uses. Apoferritin is a hollow protein shell about 130 Å across
  (1 Å = one ten-billionth of a metre). It's the standard test specimen of
  cryo-EM: rigid, abundant, and highly symmetric — 24 different rotations leave
  it looking identical, which is called **octahedral (O) symmetry**. Everyone
  knows what the right answer looks like, which is exactly what you want for a
  benchmark.
- **EMD-15854** — a published high-resolution map of that same apoferritin, used
  as the template. Both pickers use this identical file.

### The three questions

| | question | how we answer it |
|---|---|---|
| **Task 1** | Does it matter which alignment program you use? | Run both, all the way to particle picking, changing *nothing* else. |
| **Task 2** | Does it matter which particle picker you use? | Run both on the *same* tomograms. |
| **Task 3** | Can someone else see and re-check the answer? | A dashboard driven entirely by the result tables. |

### One point that governs the whole design

Both alignment programs print an "error" or "residual" when they finish, and the
tempting move is to declare the smaller number the winner. **That comparison is
invalid.** IMOD reports the average distance, in pixels, between where it
predicted a tracked patch would land and where it actually landed. AreTomo
reports an error from a completely different projection-matching calculation.
They are different quantities on different scales — like saying a golfer beat a
basketball player because they scored fewer points.

So this project judges the two alignments on things that mean the same for both:
how sharp the resulting tomograms are, how many molecules can be found in them,
and how confidently. That is why Task 1 is deliberately carried all the way
through to template matching, exactly as the assessment asks. Each program's own
residual is still recorded — clearly labelled as a within-method diagnostic that
must not be compared across methods.

---

## Part 2 — Machine and environment setup

The three GPU programs (WarpTools, AreTomo2, PyTom) are **NVIDIA CUDA only**.
There is no Apple-Metal or CPU version, and Docker or a VM on an Apple-silicon
Mac does not turn an Apple GPU into a CUDA GPU. A Mac can run `analyze.py` and
`dashboard.py`; it cannot run `run_workflow.py`.

### 2.1 The GCP instance

An **NVIDIA L4** (24 GB, compute capability 8.9) is a good fit — Warp's
algorithms target ~16 GB cards.

```bash
gcloud compute instances create cryoet \
  --zone=us-central1-a \
  --machine-type=g2-standard-16 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud
```

**200 GB is comfortable; ~120 GB is the floor.** Where it goes:

| | size |
|---|---|
| Ubuntu + NVIDIA driver + CUDA | ~20 GB |
| conda envs (warp ~10, pytom ~5) + IMOD | ~17 GB |
| raw movies, 5 tilt series | ~13 GB |
| Warp motion-corrected averages + half-sets | ~29 GB |
| tilt stacks, tomograms, correlation volumes, both branches | ~2 GB |
| **total in use** | **~81 GB** |

The tomograms themselves are tiny — 347x474x79 voxels, 26 MB each — because
everything after alignment runs at 10 A/pixel. The bulk is the raw movies and
the motion-corrected averages Warp writes alongside them.

Install the driver and confirm the GPU is visible:

```bash
sudo apt-get update && sudo apt-get install -y build-essential wget git unzip
curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
sudo python3 install_gpu_driver.py
nvidia-smi          # must list the L4 before you go any further
```

### 2.2 Why you need conda AND a venv

This tripped me up in an earlier draft, so here it is plainly.

**WarpTools and PyTom are not on PyPI.** `pip install warptools` does not exist —
WarpTools ships only as a conda package, and PyTom needs CuPy built against your
CUDA version, which conda handles and pip does not. So **conda is not optional**,
whatever you would prefer.

You end up with three environments, each used at a different point. They never
need to be active at the same time:

| # | environment | tool | holds | used for |
|---|---|---|---|---|
| A | `warp` | conda | WarpTools (+ IMOD and AreTomo2 on `PATH`) | preprocessing, both alignments, reconstruction, Warp picking |
| B | `pytom` | conda | pytom-match-pick, CuPy | the PyTom picking step only |
| C | `~/venvs/cryoet` | venv, Python 3.11 | numpy, pandas, scipy, matplotlib, mrcfile, starfile, streamlit | building the CSV tables, the analysis, the dashboard |

Environment C is the plain Python 3.11 venv. It is deliberately separate because
it is the only one you also want on your laptop — `analyze.py` and
`dashboard.py` need no GPU and no cryo-ET software, so you can pull the small
CSV tables down from the VM and do all the analysis and plotting locally.

**Install conda first** (Miniforge — two commands, no licence issues):

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash && exec bash     # `conda` now works in new shells
```

**Then environment C**, the analysis venv:

```bash
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv ~/venvs/cryoet
source ~/venvs/cryoet/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, pandas, scipy, matplotlib, mrcfile, starfile, streamlit; print('analysis env OK')"
deactivate
```

> **If you're on the Mac and hit a macOS "malware" popup:** the `.venv` in the
> old part of this repo was copied from a different machine — its `pyvenv.cfg`
> still points at another user's home directory. Virtual environments are not
> portable. Delete it and build a fresh one. Don't strip quarantine attributes.

### 2.3 Getting the code onto the VM

Only `cryoet_comparison/` is needed — 1.2 MB, 37 files. Nothing else in the
repository is used at runtime.

```bash
# from your Mac, in the repo root
gcloud compute scp --recurse cryoet_comparison cryoet:~/ --zone=us-central1-a
```

Or, if the repo is on GitHub, `git clone` on the VM is simpler and makes it
trivial to pull fixes later. Either way you end up with `~/cryoet_comparison/`,
and every command below is run from inside it.

Do **not** copy the raw data up — `setup_data.py --download` pulls it directly
from EMBL-EBI to the VM, which is far faster than going via your laptop.

### 2.4 The cryo-ET software

Four installs. Each gets verified by `setup_data.py --check`, which records the
version it finds into `results/tables/software_versions.csv`.

```bash
# Miniforge (conda) - Warp and PyTom are distributed as conda environments
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
```

**1 — WarpTools** ([docs](https://warpem.github.io/warp/user_guide/warptools/installation/))

```bash
conda create -n warp warp -c warpem -c nvidia/label/cuda-11.8.0 -c pytorch -c conda-forge -y
conda activate warp && WarpTools --version && conda deactivate
```

**2 — IMOD**, which provides `etomo`
([docs](https://bio3d.colorado.edu/imod/)). Warp's `ts_etomo_patches` shells out
to it, so it must be on `PATH` inside the warp environment.

```bash
# Check https://bio3d.colorado.edu/imod/download.html for the current filename
cd /tmp && wget https://bio3d.colorado.edu/imod/AMD64-RHEL5/imod_5.1.1_RHEL8-64_CUDA12.0.sh
sudo sh imod_5.1.1_RHEL8-64_CUDA12.0.sh -yes
source /etc/profile.d/IMOD.sh    # add this line to ~/.bashrc
tiltalign -h | head -1
```

**3 — AreTomo2** ([repo](https://github.com/czimaginginstitute/AreTomo2)).
Grab the newest release binary — as of writing that is **v1.1.2**; do not
hard-code a URL from an older guide, several point at releases that no longer
exist.

```bash
mkdir -p ~/opt/aretomo2 && cd ~/opt/aretomo2
# Copy the current asset URL from https://github.com/czimaginginstitute/AreTomo2/releases
wget <AreTomo2_1.1.2_Cuda118_*.zip>
unzip -o AreTomo2_*.zip && chmod +x AreTomo2*
ln -sf ~/opt/aretomo2/AreTomo2* ~/opt/aretomo2/AreTomo2
echo 'export PATH=$HOME/opt/aretomo2:$PATH' >> ~/.bashrc && source ~/.bashrc
AreTomo2 | head -3
```

**4 — PyTom** ([repo](https://github.com/SBC-Utrecht/pytom-match-pick))

```bash
conda create -n pytom -c conda-forge python=3.11 cupy cuda-version=11.8 -y
conda activate pytom
python -m pip install "pytom-match-pick[plotting]" starfile
pytom_match_template.py --version
conda deactivate
```

### 2.5 Verify, then download

```bash
source ~/venvs/cryoet/bin/activate
python setup_data.py --check        # software inventory + GPU check
python setup_data.py --download     # ~13 GB, resumable
```

---

## Part 3 — Where the data comes from

`setup_data.py --download` fetches all of this. The sources, if you want them
by hand:

| what | where |
|---|---|
| Tilt series (5 × 41 movies) | `ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data/tiltseries/data/*-{1,11,17,23,32}_*.tif` |
| Metadata (`.mdoc`) | `ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data/tiltseries/mdoc/TS_{1,11,17,23,32}.mrc.mdoc` |
| Gain reference | `ftp://ftp.ebi.ac.uk/empiar/world_availability/10491/data/gain_ref.mrc` |
| Dataset landing page | <https://www.ebi.ac.uk/empiar/EMPIAR-10491/> |
| Template map | <https://www.ebi.ac.uk/emdb/EMD-15854> |
| Pre-computed results (38 GB, optional) | <https://doi.org/10.5281/zenodo.11398168> |

By default everything lands in `../cryoet_data/`, next to the repo but outside
it. Point it somewhere else with:

```bash
export CRYOET_DATA_DIR=/mnt/disks/cryoet/empiar10491
```

> **A note on the dataset identity.** The Warp tilt-series tutorial uses
> **EMPIAR-10491** (apoferritin), tilt series `TS_1, TS_11, TS_17, TS_23,
> TS_32`. It is not EMPIAR-10164, which is immature HIV-1 virus-like particles —
> a different specimen with different acquisition geometry. Applying this
> tutorial's pixel size, dose and volume dimensions to that dataset would
> invalidate the processing. Likewise **EMD-15854 is apoferritin (130 Å,
> octahedral)**, not a ribosome (~300 Å); the diameter, mask size, symmetry and
> match tolerance all follow from getting this right.

---

## Part 4 — Running it

Three environments, activated in order. Each is used once, then you move on.

```bash
cd ~/cryoet_comparison

# ---- environment C: check the machine, then pull the data -------------------
source ~/venvs/cryoet/bin/activate
python setup_data.py --check          # software inventory + GPU check
python setup_data.py --download       # ~13 GB from EMBL-EBI, resumable
deactivate

# ---- environment A: everything Warp does ------------------------------------
conda activate warp
python run_workflow.py --preprocess    # motion + CTF + tilt-series grouping (once)
python run_workflow.py --align         # BOTH alignment branches
python run_workflow.py --reconstruct   # defocus handedness, CTF, tomograms
python run_workflow.py --warp-pick     # Warp template matching, both branches
conda deactivate

# ---- environment B: the PyTom step ------------------------------------------
conda activate pytom
python run_workflow.py --pytom-pick
conda deactivate

# ---- environment C again: analysis, no GPU ----------------------------------
source ~/venvs/cryoet/bin/activate
python run_workflow.py --collect       # write the tidy CSV tables
python analyze.py                      # statistics, plots, conclusions
streamlit run dashboard.py --server.port 8501
```

To see the dashboard from your laptop, tunnel to it rather than opening a
firewall port:

```bash
gcloud compute ssh cryoet --zone=us-central1-a -- -L 8501:localhost:8501
# then open http://localhost:8501
```

Or copy the results down — they are small — and run the dashboard locally:

```bash
gcloud compute scp --recurse cryoet:~/cryoet_comparison/results ./cryoet_comparison/ --zone=us-central1-a
```

Each stage skips work whose output already exists, so you can stop and resume at
any point. `--all` runs every stage in order, useful only if one environment
happens to have everything.

### How long does it take?

Estimates for one L4, five tilt series. Treat them as order-of-magnitude — real
times depend on disk and on how busy EBI's FTP server is.

| stage | time | bound by |
|---|---|---|
| Download ~13 GB | 15–90 min | EBI's server, not your machine |
| `--preprocess` (motion + CTF, 205 movies) | 20–45 min | GPU |
| `--align`, IMOD branch | 20–60 min | CPU — IMOD patch tracking is not GPU work |
| `--align`, AreTomo2 branch | 5–15 min | GPU |
| `--reconstruct` (both branches) | 15–30 min | GPU |
| `--warp-pick` (both branches) | 30–90 min | GPU |
| `--pytom-pick` | 1.5–3.5 h | GPU — the long pole, see below |
| `--collect` + `analyze.py` | seconds | nothing |
| **total** | **roughly 4–8 hours** | |

PyTom dominates because it can only exploit symmetry about one axis, so it
searches roughly six times more orientations than Warp does for the same answer.
That is a capability difference, not inefficiency, and the report says so rather
than quietly presenting it as a speed result.

Everything after `--collect` runs in about a second on any machine, so you can
iterate on the analysis and the plots as much as you like without touching the
GPU again. **Stop the VM once `--collect` is done** — the tables are a few
hundred KB.

### No GPU yet? Dry-run the whole analysis first

```bash
python analyze.py --selftest && streamlit run dashboard.py
```

This fabricates tables in the real schema, runs every line of the analysis and
the dashboard on them, and writes to `results/selftest/` with **SYNTHETIC**
stamped on every plot, every paragraph and every dashboard page. Do this on your
laptop before you spend a single GPU-hour — you will see exactly what the real
run produces, and you will know the analysis works before there is anything real
to analyse. It tests the software. It is never a result.

## Part 5 — What the workflow actually does

### Shared preprocessing, done once

```
create_settings → fs_motion_and_ctf → ts_import → create_settings (tilt series)
```

### Then it branches — and this is the point

```
                    ┌─ ts_etomo_patches  --output_processing warp_tiltseries_etomo
 shared preprocessing┤
                    └─ ts_aretomo        --output_processing warp_tiltseries_aretomo

 each branch then:  ts_defocus_hand → ts_ctf → ts_reconstruct → ts_template_match
                    → threshold_picks        (all with --input_processing <branch>)
```

`--output_processing` redirects Warp's output into a separate folder;
`--input_processing` makes later steps read from it. This is the mechanism
Warp's own documentation recommends for exactly this comparison, and it means
motion correction and CTF estimation happen **once**. Repeating them per branch
would waste hours and, worse, introduce a difference between the branches that
has nothing to do with alignment.

**Defocus handedness** deserves a mention. There's a sign ambiguity in how
defocus varies across a tilted sample; getting it backwards quietly costs you
resolution. Warp's `ts_defocus_hand --check` measures it. The two official Warp
examples *disagree* about which way this dataset goes — the tutorial page
reports a positive correlation, the end-to-end script reports negative — most
likely a version difference. So the script **reads Warp's answer and acts on it**
rather than hard-coding either outcome, and records the decision in
`results/tables/defocus_handedness.csv`.

### Task 2 holds the tomograms constant

Both pickers run on the etomo branch's tomograms, with the same EMD-15854 map,
the same 130 Å diameter and the same 7.5° angular step. PyTom is handed Warp's
per-tilt-series XML so it uses identical tilt angles, dose and defocus. The only
remaining difference is the program itself — plus one that can't be removed and
so is reported instead: Warp can exploit apoferritin's full octahedral symmetry,
while PyTom only supports symmetry about the z axis, so PyTom searches ~6×
more orientations for the same answer. That is a runtime caveat, not a quality
difference, and it's in `results/tables/parameters.csv`.

---

## Part 6 — What comes out

Two layers. `run_workflow.py` writes **measurements**; `analyze.py` turns those
into **findings**. Keeping them apart is what lets you redo every plot and every
statistic in one second without going near the GPU again.

### `results/tables/` — raw measurements (written by `run_workflow.py`)

| file | rows | columns | what it is for |
|---|---|---|---|
| `runtimes.csv` | ~12 | `stage, method, seconds, n_tilt_series, seconds_per_tilt_series, note` | the runtime comparison in both tasks |
| `tomogram_stats.csv` | 10 (5 series × 2 branches) | `branch, tilt_series, nx, ny, nz, contrast, sharpness` | Task 1 image-quality metrics |
| `warp_picks.csv` | thousands | `branch, tilt_series, x_A, y_A, z_A, score, score_units, coords_were_normalised` | every Warp pick, both branches — feeds Task 1 *and* Task 2 |
| `pytom_picks.csv` | thousands | `tilt_series, x_A, y_A, z_A, score, score_units, cutoff` | every PyTom pick — Task 2 |
| `native_residuals.csv` | 10 | `branch, tilt_series, metric, value, unit, source, comparable_across_methods` | each aligner's own error — **diagnostic only**, never cross-compared |
| `defocus_handedness.csv` | 2 | `branch, correlation, flip_applied` | records the measurement and the decision taken from it |
| `parameters.csv` | ~13 | `scope, parameter, warp_or_etomo, aretomo_or_pytom, note` | the "key parameters" the assessment asks for |
| `software_versions.csv` | 6 | `tool, found, path, version` | provenance — what actually produced these numbers |

Two things worth knowing about the pick tables:

- **Both are in Ångströms**, converted at collection time. Warp writes
  coordinates *normalised to 0–1* across the volume; PyTom writes them in
  *voxels*. Comparing those raw columns would put every Warp particle within one
  voxel of the origin and yield a meaningless answer. The conversion reads the
  volume dimensions from the MRC header rather than assuming them.
- **The two `score` columns are different quantities.** Warp's is standard
  deviations above the volume's background; PyTom's is a normalised correlation
  coefficient. `score_units` records which, and the analysis never puts them on
  a shared axis.

### `results/` — the findings (written by `analyze.py`)

**Seven tables:**

| file | what it answers |
|---|---|
| `task1_summary.csv` | the headline Task 1 comparison: five metrics, each paired across tilt series, with the mean difference, how many of the 5 series agree on direction, and the Wilcoxon p |
| `task1_per_tilt_series.csv` | the raw per-series numbers behind that summary — so a reader can check it rather than trust it |
| `task1_branch_agreement.csv` | do the two alignments put the same molecules in the same places? |
| `task2_summary.csv` | the headline picking numbers: counts, matched, Jaccard, rank correlation, runtimes |
| `task2_per_tomogram.csv` | the same, broken down per tomogram |
| `task2_radius_sweep.csv` | how the agreement changes as the "same particle" tolerance goes from 10 Å to 200 Å |
| `task2_unique_vs_matched.csv` | the test of whether method-unique picks really are the low-scoring ones |

**Nine plots:**

| file | what you are looking at |
|---|---|
| `task1_tomogram_quality.png` | Two panels, sharpness and contrast. Each tilt series is a line joining its two branch values — a *paired* plot, so you can see whether the difference is consistent or driven by one outlier. A bar chart of two means would hide that. |
| `task1_particle_yield.png` | Left: particles found per tilt series, per branch. Right: the yield curve as the score cutoff moves from 3σ to 12σ. If one branch is above the other everywhere, the conclusion does not depend on where you set the threshold. |
| `task1_score_distributions.png` | Histogram of template-matching scores from both branches, **sharing an axis** — legitimate here because it is the same program, template and normalisation. A distribution pushed further right means molecules stood out more clearly, which means better alignment. **This is the plot that answers Task 1.** |
| `task1_runtime.png` | Stacked bars: where the time went in each alignment route. |
| `task2_counts.png` | Warp, agreed, and PyTom counts per tomogram. |
| `task2_radius_sweep.png` | Agreement versus matching tolerance, with one particle radius marked. Shows whether the headline overlap is robust or an artefact of the cutoff. |
| `task2_score_distributions.png` | Two panels on **separate axes**, each split into picks the other tool confirmed versus picks it missed. This is the visual test of the "unique picks are junk" claim. |
| `task2_rank_agreement.png` | Warp score against PyTom score for matched picks, with Spearman ρ. Rankings can be compared even when raw scores cannot. |
| `task2_xy_example.png` | Where the picks physically sit in one tomogram, looking down the beam. A sanity check — clustering or edge effects show up here immediately. |

**And `conclusions.md`** — the written report, generated entirely from the tables
above. Every verdict comes from a comparison between computed numbers, so if the
data reversed tomorrow the text would reverse with it.

`dashboard.py` reads all of this and adds the two sliders.

## Part 7 — The methodology choices worth defending

**Paired comparisons, and honesty about n=5.** Both methods saw the same five
tilt series, so the comparison is paired — that removes the (large) variation
between tilt series and isolates the method. With five pairs, a Wilcoxon
signed-rank test cannot produce p < 0.05 *even in the best possible case*. So
the headline evidence is **consistency of direction** (5/5 tilt series pointing
the same way), with the p-value reported alongside and not leaned on. The report
says this out loud rather than quietly quoting a marginal p-value.

**Optimal one-to-one matching, not nearest-neighbour.** Deciding which Warp pick
is "the same molecule" as which PyTom pick is an assignment problem. Giving every
pick its nearest neighbour double-counts; claiming neighbours first-come-first-
served makes the answer depend on the order the file happened to be written in.
`analyze.py` solves it properly with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`): the pairing with the smallest total
distance, each pick used at most once, independent of file order.

**The match tolerance is swept, not chosen.** The single most manipulable number
in a picker comparison is "how close counts as the same particle" — you can move
the overlap statistic a long way just by moving it. So the operating point is
**65 Å = one apoferritin radius** (beyond that, two "matching" centres describe
spheres that barely overlap), and the full curve from 10 Å to 200 Å is published
alongside it. The dashboard makes it a slider.

**Scores are never forced onto a shared axis.** Warp's score is *standard
deviations above the volume's background*. PyTom's is a *normalised
cross-correlation coefficient*. Min-max rescaling both to 0–1 to make them
"comparable" would invent a relationship that doesn't exist and would be at the
mercy of a single outlier. They get separate panels, and where the two tools are
compared it's by **rank** (Spearman), which is scale-free.

**"Unique picks are false positives" is tested, not assumed.** For each tool,
the scores of picks the other tool confirmed are compared against the picks it
missed (Mann-Whitney U, plus the common-language effect size: the probability a
confirmed pick outscores a unique one). The report states whichever answer the
data gives — including "not confirmed".

**Nothing in the report or the dashboard is hard-coded.** Every verdict comes
from a comparison between computed numbers. If the data reversed tomorrow, the
text would reverse with it.

**Log parsing fails loudly.** The residual parsers look only for explicit,
labelled summary lines — never "any numeric row with five columns", which will
happily parse an unrelated table and hand back a confident wrong answer. If a
log can't be parsed, a `parse_failed` row is written. An empty file that looks
like "no error" is the worst possible outcome.

---

## Part 8 — Known limits

- **Five tilt series, one specimen.** Apoferritin is small, rigid, symmetric and
  in thin ice — close to the easiest case there is. None of this generalises to
  thick cellular samples without being repeated on thick cellular samples.
- **No ground truth.** There is no hand-annotated list of where the molecules
  really are, so "precision" and "recall" in the strict sense aren't available.
  What's reported is mutual confirmation between two independent methods, which
  is a weaker but honest substitute — and it is labelled as such.
- **The definitive test isn't done here.** Running the agreed picks through
  subtomogram averaging and comparing the resolution reached would convert every
  proxy on this page into the number that actually matters. That needs RELION
  and M, and is the obvious next step.
- **Runtime granularity.** Warp aligns and matches all five tilt series inside a
  single call, so what's measured is total wall clock for five series on one
  GPU. Per-series timings are not measured and are not claimed.
- **Symmetry handling differs** between the two pickers and cannot be equalised
  (see Part 5). It affects the runtime comparison, and is reported rather than
  glossed over.

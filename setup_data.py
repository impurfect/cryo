#!/usr/bin/env python3
"""
setup_data.py - Task 0: get the software and the data ready.

WHAT THIS SCRIPT DOES, IN PLAIN LANGUAGE
----------------------------------------
Before we can compare anything we need two things on the machine:

  1. The four programs the assessment names (WarpTools, AreTomo2, PyTom and
     IMOD). This script does not install them - installation is a system-level
     job that is documented step by step in README.md - but it CHECKS that they
     are installed, records exactly which version of each one is present, and
     confirms the GPU actually works. That version record is what makes the
     results reproducible six months from now.

  2. The raw data. That is ~205 movie files from the public EMPIAR archive
     (about 100 GB), plus five small text files of metadata, plus one
     calibration image, plus the known 3D shape of apoferritin from the EMDB.

Run it like this:

    python setup_data.py --check          # only verify the software, download nothing
    python setup_data.py --download       # verify, then download everything

Everything is skipped if it is already present, so it is safe to re-run after
an interrupted download.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import config


# ===========================================================================
#  PART 1 - checking that the software is installed and recording versions
# ===========================================================================

def _run(cmd, timeout=120):
    """Run a shell command and give back (exit_code, combined_output).

    Every external program in this project is invoked through this one helper
    so that error handling and logging behave identically everywhere.
    """
    try:
        proc = subprocess.run(
            cmd, shell=isinstance(cmd, str), capture_output=True,
            text=True, timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def check_tool(name, executable, version_cmd, version_regex=None):
    """Confirm one program exists and capture the line that names its version.

    We return a dictionary rather than printing, because run_workflow.py stores
    these dictionaries in the results table - the software inventory becomes
    part of the deliverable, not just something on the screen.

    Being on PATH is not the same as being runnable. AreTomo2 is a bare binary
    that links against the CUDA runtime (libcudart, libcufft). Those live in the
    conda environment's lib directory, which conda does NOT add to
    LD_LIBRARY_PATH - Python and .NET find their own libraries through RPATH, so
    nothing else in the stack notices. AreTomo2 is therefore present, on PATH,
    and completely unable to start. We look for that specific failure and say so,
    instead of printing a healthy tick next to a linker error.
    """
    path = shutil.which(executable)
    if path is None:
        return {"tool": name, "found": False, "path": "", "version": "NOT FOUND"}

    _, out = _run(version_cmd)

    # Several of these tools exit non-zero, or print a stack trace, even when
    # they are working perfectly - WarpTools prints its version banner and then
    # aborts because "--version" is not one of its verbs. What we actually care
    # about is whether the binary STARTED, which the banner proves. So when a
    # tool gives us a version regex, a match counts as success no matter what
    # the process did afterwards.
    if version_regex:
        m = re.search(version_regex, out)
        if m:
            return {"tool": name, "found": True, "path": path,
                    "version": m.group(0).strip()}

    if "error while loading shared libraries" in out or "cannot open shared object" in out:
        missing = re.search(r"(lib[\w.+-]+\.so[\w.]*)", out)
        return {"tool": name, "found": False, "path": path,
                "version": f"INSTALLED BUT CANNOT RUN - missing "
                           f"{missing.group(1) if missing else 'a shared library'}. "
                           f"The library is almost certainly in $CONDA_PREFIX/lib "
                           f"but not on LD_LIBRARY_PATH - see README 2.6."}
    # A version banner is usually in the first few lines; keep the first line
    # that actually mentions a number, otherwise just the first non-empty line.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    version = next((ln for ln in lines if any(c.isdigit() for c in ln)),
                   lines[0] if lines else "unknown")
    return {"tool": name, "found": True, "path": path, "version": version[:200]}


def check_aretomo():
    """AreTomo2 needs its own check, because it has no --version flag.

    Run with no arguments it starts up, discovers there is no input file, prints
    "Error: unable to open MRC file" and exits. That failure is the success
    signal we want: reaching it proves the binary loaded libcudart and libcufft.
    The only outcome that means trouble is a linker error before that point.

    For the version we resolve the symlink and report the real filename, e.g.
    AreTomo2_1.1.2_Cuda121. That records the CUDA build as well as the release -
    and the CUDA build is the thing most likely to be wrong, since a Cuda118
    binary cannot run against the CUDA 12 libraries in this environment.
    """
    path = shutil.which("AreTomo2")
    if path is None:
        return {"tool": "AreTomo2", "found": False, "path": "",
                "version": "NOT FOUND"}

    real = Path(path).resolve().name
    _, out = _run("AreTomo2")

    if "error while loading shared libraries" in out or "cannot open shared object" in out:
        missing = re.search(r"(lib[\w.+-]+\.so[\w.]*)", out)
        return {"tool": "AreTomo2", "found": False, "path": path,
                "version": f"INSTALLED BUT CANNOT RUN - missing "
                           f"{missing.group(1) if missing else 'a shared library'}. "
                           f"The library is almost certainly in $CONDA_PREFIX/lib "
                           f"but not on LD_LIBRARY_PATH - see README 2.6."}

    return {"tool": "AreTomo2", "found": True, "path": path, "version": real}


def check_gpu():
    """Confirm an NVIDIA GPU is visible and report its name and memory.

    All three GPU programs here (Warp, AreTomo2, PyTom) are CUDA-only. There is
    no Apple-Metal or CPU fallback, so if this check fails the processing part
    of the project simply cannot run on this machine.
    """
    code, out = _run("nvidia-smi --query-gpu=index,name,memory.total,driver_version "
                     "--format=csv,noheader")
    if code != 0:
        return {"tool": "NVIDIA GPU", "found": False, "path": "",
                "version": "no CUDA GPU visible (nvidia-smi failed)"}
    return {"tool": "NVIDIA GPU", "found": True, "path": "nvidia-smi",
            "version": "; ".join(ln.strip() for ln in out.splitlines() if ln.strip())}


def software_inventory():
    """Check everything at once and return the full inventory as a list."""
    return [
        check_gpu(),
        # "WarpTools --version" prints the banner then crashes: --version is not
        # a verb. --help is the supported way in, and the banner carries the
        # version either way.
        check_tool("WarpTools", "WarpTools", "WarpTools --help",
                   r"Version\s+[\d.]+\S*"),
        check_aretomo(),
        # tiltxcorr and tiltalign are the two IMOD programs patch tracking
        # actually runs, and both print a proper version banner. "etomo" itself
        # is the GUI wrapper - we only confirm it exists, and read the version
        # from IMOD's own VERSION file rather than from -h output.
        check_tool("IMOD (etomo)", "etomo", 'cat "$IMOD_DIR/VERSION"',
                   r"[\d]+\.[\d.]+"),
        check_tool("IMOD (tiltxcorr)", "tiltxcorr", "tiltxcorr -h",
                   r"[Vv]ersion\s+[\d.]+\S*"),
        check_tool("IMOD (tiltalign)", "tiltalign", "tiltalign -h",
                   r"[Vv]ersion\s+[\d.]+\S*"),
        # --help does not carry the package version; ask the package metadata.
        check_tool("PyTom match", "pytom_match_template.py",
                   'python -c "import importlib.metadata as m; '
                   'print(\'pytom-match-pick\', m.version(\'pytom-match-pick\'))"',
                   r"pytom-match-pick\s+[\d.]+\S*"),
    ]


def print_inventory(inv):
    """Show the inventory as a small table and say whether we can proceed."""
    print("\n--- software inventory " + "-" * 50)
    print("  (run this with the 'cryoet' conda environment active: WarpTools and")
    print("   AreTomo2 both need its CUDA runtime in order to start at all)")
    for row in inv:
        mark = "OK     " if row["found"] else "MISSING"
        print(f"  {mark} {row['tool']:<20} {row['version']}")
    print("-" * 72)
    missing = [r["tool"] for r in inv if not r["found"]]
    if missing:
        print(f"  Not usable yet: {', '.join(missing)}")
        print("  See README.md section 'Installing the software' for each one.")
    else:
        print("  All required software found.")
    return not missing


# ===========================================================================
#  PART 2 - downloading the raw data
# ===========================================================================

def _wget(url, dest_dir, quiet=False):
    """Download one URL into dest_dir, skipping it if it is already there.

    -N ("timestamping") is what makes re-running safe: wget only re-fetches a
    file if the copy on the server is newer than the copy on disk.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-N", "--no-directories", "--directory-prefix", str(dest_dir)]
    if quiet:
        cmd.append("--quiet")
    else:
        cmd.append("--show-progress")
    cmd.append(url)
    return subprocess.run(cmd).returncode == 0


def check_disk_space(need_gb=45):
    """Refuse to start a 13 GB download onto a disk that cannot finish the job.

    The download itself is 13 GB, but Warp's motion-corrected averages need
    roughly another 10 GB on top, so the real requirement before preprocessing
    is closer to 45 GB free. Finding that out three hours in, with a half-written
    dataset, is a bad afternoon.
    """
    st = shutil.disk_usage(config.DATA_DIR.parent if config.DATA_DIR.exists()
                           else Path.home())
    free_gb = st.free / 1e9
    print(f"\nDisk: {free_gb:.0f} GB free of {st.total / 1e9:.0f} GB")
    if free_gb < need_gb:
        print(f"  WARNING: about {need_gb} GB is needed for the raw data plus "
              f"Warp's motion-corrected averages.")
        print(f"  Free some space, attach a bigger disk, or point CRYOET_DATA_DIR "
              f"at one:")
        print(f"      export CRYOET_DATA_DIR=/mnt/disks/big/empiar10491")
        return False
    return True


def download_raw_data():
    """Fetch the EMPIAR-10491 tilt series: gain reference, metadata, movies.

    Three kinds of file come down:

      gain_ref.mrc   one calibration image of the camera. Every raw movie is
                     divided by this to remove the camera's fixed pattern.

      TS_*.mrc.mdoc  one small text file per tilt series. It lists, for every
                     image in the series, which movie file it is, what angle the
                     stage was at, and when it was taken. This is the metadata
                     that turns 205 unrelated movies into 5 ordered tilt series.

      *.tif          the movies themselves. Each one is a short burst of frames
                     of the same view; averaging them (after correcting for
                     drift) gives one image of the specimen at one tilt angle.
                     There are 41 of these per tilt series.
    """
    print(f"\nDownloading EMPIAR-{config.EMPIAR_ACCESSION} into {config.DATA_DIR}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] gain reference")
    _wget(f"{config.EMPIAR_FTP}/{config.GAIN_FILE}", config.DATA_DIR)

    for n in config.SERIES_NUMBERS:
        print(f"\n[2/3] TS_{n} metadata (.mdoc)")
        _wget(f"{config.EMPIAR_FTP}/tiltseries/mdoc/TS_{n}.mrc.mdoc",
              config.MDOC_DIR, quiet=True)

        print(f"[3/3] TS_{n} movies (.tif) - this is the slow part")
        # The movie filenames embed the series number as "-<n>_", e.g.
        # 2Dvs3D_53-11_00007_8.0_Jul31_16.54.59.tif belongs to TS_11.
        _wget(f"{config.EMPIAR_FTP}/tiltseries/data/*-{n}_*.tif",
              config.FRAMES_DIR)


def download_template():
    """Fetch EMD-15854, the reference 3D shape used for particle picking.

    EMD-15854 is a published 1.8 A map of mouse heavy-chain apoferritin. Warp
    can pull it from the EMDB itself (--template_emdb 15854), but PyTom needs
    the file on disk, so we download it once here and both tools use the same
    starting map. Using literally the same file for both pickers removes one
    possible explanation for any difference we see between them.
    """
    if config.TEMPLATE_MAP.exists():
        print(f"\nTemplate already present: {config.TEMPLATE_MAP}")
        return

    print(f"\nDownloading EMD-{config.TEMPLATE_EMDB_ID} (apoferritin reference map)")
    gz_name = f"emd_{config.TEMPLATE_EMDB_ID}.map.gz"
    url = (f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
           f"EMD-{config.TEMPLATE_EMDB_ID}/map/{gz_name}")
    if not _wget(url, config.DATA_DIR):
        print("  Download failed. Fetch it by hand from "
              f"https://www.ebi.ac.uk/emdb/EMD-{config.TEMPLATE_EMDB_ID}")
        return
    subprocess.run(["gunzip", "-f", str(config.DATA_DIR / gz_name)], check=True)
    print(f"  -> {config.TEMPLATE_MAP}")


def report_data_status():
    """Count what is on disk and flag anything obviously incomplete."""
    n_frames = len(list(config.FRAMES_DIR.glob("*.tif"))) if config.FRAMES_DIR.exists() else 0
    n_mdoc = len(list(config.MDOC_DIR.glob("*.mdoc"))) if config.MDOC_DIR.exists() else 0
    expected_frames = 41 * len(config.SERIES_NUMBERS)   # 41 tilts x 5 series

    print("\n--- data status " + "-" * 57)
    print(f"  data directory : {config.DATA_DIR}")
    print(f"  gain reference : {'present' if config.GAIN_PATH.exists() else 'MISSING'}")
    print(f"  mdoc files     : {n_mdoc} / {len(config.SERIES_NUMBERS)}")
    print(f"  movie files    : {n_frames} / {expected_frames} expected")
    print(f"  template map   : {'present' if config.TEMPLATE_MAP.exists() else 'MISSING'}")
    if config.DATA_DIR.exists():
        code, out = _run(f"du -sh {config.DATA_DIR}")
        if code == 0:
            print(f"  size on disk   : {out.split()[0]}")
        st = shutil.disk_usage(config.DATA_DIR)
        print(f"  free space     : {st.free / 1e9:.0f} GB")
    print("-" * 72)

    ok = (config.GAIN_PATH.exists() and n_mdoc == len(config.SERIES_NUMBERS)
          and n_frames == expected_frames and config.TEMPLATE_MAP.exists())
    print("  Data complete." if ok else
          "  Data incomplete - re-run with --download (it resumes safely).")
    return ok


# ===========================================================================
#  main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="only verify software and report data status")
    ap.add_argument("--download", action="store_true",
                    help="download the EMPIAR data and the EMDB template")
    ap.add_argument("--force", action="store_true",
                    help="download even if free disk space looks insufficient")
    args = ap.parse_args()
    if not (args.check or args.download):
        ap.error("choose --check or --download")

    print(config.describe())

    inv = software_inventory()
    software_ok = print_inventory(inv)

    if args.download:
        if shutil.which("wget") is None:
            sys.exit("wget is required for downloading. Install it first "
                     "(Debian/Ubuntu: sudo apt-get install -y wget).")
        if not check_disk_space() and not args.force:
            sys.exit("Stopping. Re-run with --force to download anyway.")
        download_raw_data()
        download_template()

    data_ok = report_data_status()

    # Save the software inventory next to the results so the report can cite it.
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    with open(config.TABLES_DIR / "software_versions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tool", "found", "path", "version"])
        w.writeheader()
        w.writerows(inv)
    print(f"\nSoftware inventory written to "
          f"{config.TABLES_DIR / 'software_versions.csv'}")

    if software_ok and data_ok:
        print("\nReady. Next step:  python run_workflow.py --all")


if __name__ == "__main__":
    main()

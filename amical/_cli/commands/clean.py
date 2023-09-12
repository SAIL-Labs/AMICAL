import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

from astropy.io import fits
from matplotlib import pyplot as plt
from rich import print as rprint
from rich.progress import track

import amical
from amical._rich_display import tabulate


def _select_data_file(args, process):
    """Show report with the data found and allow to select one to be treated."""
    l_file = sorted(glob("%s/*.fits" % args.datadir))

    if len(l_file) == 0:
        print(
            f"No fits files found in {args.datadir}, check --datadir.", file=sys.stderr
        )
        return 1

    headers = ["FILENAME", "TARGET", "DATE", "INSTRUM", "INDEX"]

    index_file = []
    d = []
    for i, f in enumerate(l_file):
        with fits.open(f) as hdu:
            hdr = hdu[0].header
        target = hdr.get("OBJECT", None)
        date = hdr.get("DATE-OBS", None)
        ins = hdr.get("INSTRUME", None)
        index_file.append(i)
        filename = f.split("/")[-1]
        d.append([filename, target, date, ins, i])

    print(tabulate(d, headers=headers))

    if args.file >= 0:
        choosen_index = args.file
    else:
        choosen_index = int(input("\nWhich file to %s?\n" % process))

    try:
        filename = l_file[choosen_index]
    except IndexError:
        print(
            f"Selected index ({choosen_index}) not valid "
            f"(only {len(l_file)} files found).",
            file=sys.stderr,
        )
        raise SystemExit  # noqa: B904
    else:
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
    return filename, hdr


def perform_clean(args):
    """Clean the data with AMICAL."""
    rprint("[cyan]---- AMICAL clean process ----")

    clean_param = {
        "isz": args.isz,
        "r1": args.r1,
        "dr": args.dr,
        "apod": args.apod,
        "window": args.window,
        "f_kernel": args.kernel,
    }

    if not os.path.exists(args.datadir):
        print(
            f"{args.datadir} directory not found, check --datadir. "
            "AMICAL look for data only in this specified directory.",
            file=sys.stderr,
        )
        return 1

    l_file = sorted(glob("%s/*.fits" % args.datadir))
    if len(l_file) == 0:
        print(
            f"No fits files found in {args.datadir}, check --datadir.", file=sys.stderr
        )
        return 1

    if not args.all:
        filename, hdr = _select_data_file(args, process="clean")

    if args.check:
        amical.show_clean_params(filename, **clean_param)
        plt.show(block=True)
        return 0

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    clean_param["clip"] = args.clip
    clean_param["sky"] = args.sky

    if args.all:
        # Clean all files in --datadir
        for f in track(l_file, description="# files"):
            hdr = fits.open(f)[0].header
            hdr["HIERARCH AMICAL step"] = "CLEANED"
            cube = amical.select_clean_data(f, **clean_param, display=True)
            f_clean = os.path.join(args.outdir, Path(f).stem + "_cleaned.fits")
            fits.writeto(f_clean, cube, header=hdr, overwrite=True)
    else:
        # Or clean just the specified file (in --datadir)
        hdr["HIERARCH AMICAL step"] = "CLEANED"
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        hdr["HIERARCH AMICAL time"] = dt_string
        for k in clean_param:
            hdr["HIERARCH AMICAL params %s" % k] = clean_param[k]
        cube = amical.select_clean_data(filename, **clean_param, display=True)
        if args.plot:
            plt.show()
        f_clean = os.path.join(args.outdir, Path(filename).stem + "_cleaned.fits")
        fits.writeto(f_clean, cube, header=hdr, overwrite=True)
    return 0

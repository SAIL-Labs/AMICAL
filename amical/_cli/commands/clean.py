import os
from datetime import datetime
from glob import glob
from pathlib import Path

from astropy.io import fits
from matplotlib import pyplot as plt
from rich.progress import track
from termcolor import cprint

import amical
from amical._rich_display import tabulate


def _select_data_file(args, process):
    """Show report with the data found and allow to select one to be treated."""
    l_file = sorted(glob("%s/*.fits" % args.datadir))

    if len(l_file) == 0:
        print("No fits files found in %s, check --datadir." % args.datadir)
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
            "Selected index (%i) not valid (only %i files found)."
            % (choosen_index, len(l_file))
        )
        raise SystemExit  # noqa: B904
    else:
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
    return filename, hdr


def perform_clean(args):
    """Clean the data with AMICAL."""
    cprint("---- AMICAL clean process ----", "cyan")

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
            "%s directory not found, check --datadir. AMICAL look for data only in this specified directory."
            % args.datadir
        )
        return 1

    l_file = sorted(glob("%s/*.fits" % args.datadir))
    if len(l_file) == 0:
        print("No fits files found in %s, check --datadir." % args.datadir)
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

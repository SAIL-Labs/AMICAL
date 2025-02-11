import os
import sys
import time
from glob import glob
from pathlib import Path

from astropy.io import fits
from matplotlib import pyplot as plt
from rich import print as rprint
from rich.progress import track

import amical
from amical._cli.commands.clean import _select_data_file


def _extract_bs_ifile(f, args, ami_param):
    """Extract the bispectrum on individial file (f) and save them as hdf5."""
    hdu = fits.open(f)
    cube = hdu[0].data
    hdu.close()

    # Extract the bispectrum
    bs = amical.extract_bs(cube, f, **ami_param, save_to=args.save_to)

    bs_file = os.path.join(args.outdir, Path(f).stem + "_bispectrum")
    amical.save_bs_hdf5(bs, bs_file)
    return 0


def perform_extract(args):
    """CLI interface to extract the data with AMICAL (compute bispectrum object
    with all raw observables)."""
    rprint("[cyan]---- AMICAL extract started ----")
    t0 = time.time()
    ami_param = {
        "peakmethod": args.peakmethod,
        "bs_multi_tri": args.multitri,
        "maskname": args.maskname,
        "instrum": args.instrum,
        "fw_splodge": args.fw,
        "filtname": args.filtname,
        "targetname": args.targetname,
        "theta_detector": args.thetadet,
        "scaling_uv": args.scaling,
        "expert_plot": args.expert,
        "n_wl": args.nwl,
        "i_wl": args.iwl,
        "unbias_v2": args.unbias,
        "cutoff": args.cutoff,
        "hole_diam": args.diam,
    }

    if not os.path.exists(args.datadir):
        print(
            f"{args.datadir} directory not found, check --datadir. "
            "AMICAL look for data only in this specified directory.",
            file=sys.stderr,
        )
        return 1

    l_file = sorted(glob(f"{args.datadir}/*.fits"))
    if len(l_file) == 0:
        print(
            f"No fits files found in {args.datadir}, check --datadir.", file=sys.stderr
        )
        return 1

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if not args.all:
        f = _select_data_file(args, process="extract")[0]
        _extract_bs_ifile(f, args, ami_param)
    else:
        for f in track(l_file, description="# files"):
            _extract_bs_ifile(f, args, ami_param)
    t1 = time.time() - t0
    rprint(f"[cyan]---- AMICAL extract done ({t1:2.1f}s) ----")
    if args.plot:
        plt.show(block=True)
    return 0

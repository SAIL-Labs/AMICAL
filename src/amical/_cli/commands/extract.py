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
    """Extract and save the bispectrum for one FITS file.

    Parameters
    ----------
    f : str
        Path to the input FITS file.
    args : argparse.Namespace
        CLI arguments containing ``outdir`` and ``save_to`` attributes.
    ami_param : dict
        Keyword arguments passed to :func:`amical.extract_bs`.

    Returns
    -------
    int
        Zero after saving the bispectrum HDF5 file.
    """
    hdu = fits.open(f)
    cube = hdu[0].data
    hdu.close()

    # Extract the bispectrum
    bs = amical.extract_bs(cube, f, **ami_param, save_to=args.save_to)

    bs_file = os.path.join(args.outdir, Path(f).stem + "_bispectrum")
    amical.save_bs_hdf5(bs, bs_file)
    return 0


def perform_extract(args):
    """Extract AMICAL bispectra and their raw observables from FITS files.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments controlling extraction, file selection, plotting, and output.

    Returns
    -------
    int
        Zero on success or one when the input directory contains no FITS files.
    """
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

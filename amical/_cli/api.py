import os
import pickle
from datetime import datetime
from glob import glob

from astropy.io import fits
from matplotlib import pyplot as plt
from termcolor import cprint
from tqdm import tqdm

import amical


def select_data_file(args, process):
    l_file = sorted(glob("%s/*.fits" % args.datadir))

    if len(l_file) == 0:
        raise OSError("No fits files found in %s, check --datadir." % args.datadir)

    index_file = []
    print("\n  FILENAME   |   TARGET   |   DATE-OBS   |  index  ")
    print("-----------------------------------------------------")
    for i, f in enumerate(l_file):
        hdu = fits.open(f)
        hdr = hdu[0].header
        target = hdr.get("OBJECT", None)
        date = hdr.get("DATE-OBS", None)
        ins = hdr.get("INSTRUME", None)
        index_file.append(i)
        print(
            "  %s     %s     %s     %s     %i"
            % (f.split("/")[-1], target, date, ins, i)
        )

    if args.file >= 0:
        choosen_index = args.file
    else:
        choosen_index = int(input("Which file to %s?\n" % process))

    try:
        filename = l_file[choosen_index]
        hdr = fits.open(filename)[0].header
    except IndexError:
        raise IndexError(
            "Selected index (%i) not valid (only %i files found)."
            % (choosen_index, len(l_file))
        )
    return filename, hdr


def perform_clean(args):
    """ """
    cprint("---- AMICAL clean process ----", "cyan")

    clean_param = {
        "isz": args.isz,
        "r1": args.r1,
        "dr": args.dr,
        "apod": args.apod,
        "window": args.window,
        "f_kernel": args.kernel,
        "sky": args.sky,
    }

    if not os.path.exists(args.datadir):
        raise OSError(
            "%s directory not found, check --datadir. AMICAL look for data only in this specified directory."
            % args.datadir
        )

    l_file = sorted(glob("%s/*.fits" % args.datadir))
    if len(l_file) == 0:
        raise OSError("No fits files found in %s, check --datadir." % args.datadir)

    if not args.all:
        filename, hdr = select_data_file(args, process="clean")

    if args.check:
        amical.show_clean_params(filename, **clean_param)
        if args.plot:
            plt.show(block=True)
        return 0

    if not os.path.exists(args.reduceddir):
        os.mkdir(args.reduceddir)

    clean_param["clip"] = args.clip
    if args.all:
        # Clean all files in --datadir
        for f in tqdm(l_file, ncols=100, desc="# files"):
            hdr = fits.open(f)[0].header
            hdr["AMICAL"] = "CLEANED"
            cube = amical.select_clean_data(f, **clean_param, display=True)
            f_clean = f.split("/")[-1].split(".fits")[0] + "_cleaned.fits"
            fits.writeto(args.reduceddir + f_clean, cube, header=hdr, overwrite=True)
    else:
        # Or clean just the specified file (in --datadir)
        hdr["AMICAL step"] = "CLEANED"
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        hdr["AMICAL time"] = dt_string
        for k in clean_param:
            hdr["AMICAL params %s" % k] = clean_param[k]
        cube = amical.select_clean_data(filename, **clean_param, display=True)
        f_clean = filename.split("/")[-1].split(".fits")[0] + "_cleaned.fits"
        fits.writeto(args.reduceddir + f_clean, cube, header=hdr, overwrite=True)
    return 0


def perform_extract(args):
    cprint("---- AMICAL extract process ----", "cyan")

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
        raise OSError(
            "%s directory not found, check --datadir. AMICAL look for data only in this specified directory."
            % args.datadir
        )

    l_file = sorted(glob("%s/*.fits" % args.datadir))
    if len(l_file) == 0:
        raise OSError("No fits files found in %s, check --datadir." % args.datadir)

    if not os.path.exists(args.reduceddir):
        os.mkdir(args.reduceddir)

    if not args.all:
        f, hdr = select_data_file(args, process="extract")

        # Open cleaned (or not) cube
        hdu = fits.open(f)
        cube = hdu[0].data
        hdu.close()

        # Extract the bispectrum
        bs = amical.extract_bs(cube, f, **ami_param, save=args.save)

        if args.plot:
            plt.show(block=True)

        bs_file = f.split("/")[-1].split(".fits")[0] + "_bispectrum.dpy"

        # Save as python pickled file
        file = open(args.reduceddir + bs_file, "wb")
        pickle.dump(bs, file)
        file.close()

    return 0

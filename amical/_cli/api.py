import os
import time
from datetime import datetime
from glob import glob
from pathlib import Path

from astropy.io import fits
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from tabulate import tabulate
from termcolor import cprint
from tqdm import tqdm

import amical


def _select_data_file(args, process):
    """Show report with the data found and allow to select one to be treated."""
    l_file = sorted(glob("%s/*.fits" % args.datadir))

    if len(l_file) == 0:
        raise OSError("No fits files found in %s, check --datadir." % args.datadir)

    headers = ["FILENAME", "TARGET", "DATE", "INSTRUM", "INDEX"]

    index_file = []
    d = []
    for i, f in enumerate(l_file):
        hdu = fits.open(f)
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
        hdr = fits.open(filename)[0].header
    except IndexError:
        raise IndexError(
            "Selected index (%i) not valid (only %i files found)."
            % (choosen_index, len(l_file))
        )
    return filename, hdr


def _select_association_file(args):
    """Show report with the data found and allow to select the science target
    (SCI) to be calibrated and the calibrator (CAL)."""
    l_file = sorted(glob("%s/*.h5" % args.datadir))

    if len(l_file) == 0:
        raise OSError("No h5 files found in %s, check --datadir." % args.datadir)

    index_file = []

    headers = ["FILENAME", "TARGET", "DATE", "INSTRUM", "TYPE", "INDEX"]

    d = []
    for i, f in enumerate(l_file):
        bs = amical.load_bs_hdf5(f)
        filename = Path(f).stem
        hdr = bs.infos.hdr
        target = hdr.get("OBJECT", None)
        date = hdr.get("DATE-OBS", None)
        ins = hdr.get("INSTRUME", None)

        if target not in (None or ""):
            try:
                source = _query_simbad(target)
            except Exception:
                source = "Unknown"
        else:
            target = "SIMU"
            source = "Unknown"

        d.append([filename, target, date, ins, source, i])
        index_file.append(i)

    print(tabulate(d, headers=headers))

    text_calib = (
        "Which file used as calibrator? (use space "
        + "between index if multiple calibrators are available).\n"
    )
    sci_index = int(input("\nWhich file to be calibrated?\n"))
    cal_index = [int(item) for item in input(text_calib).split()]

    try:
        sci_name = l_file[sci_index]
        cal_name = [l_file[x] for x in cal_index]
    except IndexError:
        raise IndexError(
            "Selected index (sci=%i/cal=%i) not valid (only %i files found)."
            % (sci_index, cal_index, len(l_file))
        )
    return sci_name, cal_name


def _extract_bs_ifile(f, args, ami_param):
    """Extract the bispectrum on individial file (f) and save them as hdf5."""
    hdu = fits.open(f)
    cube = hdu[0].data
    hdu.close()

    # Extract the bispectrum
    bs = amical.extract_bs(cube, f, **ami_param, save=args.save)

    bs_file = args.outdir + Path(f).stem + "_bispectrum"

    amical.save_bs_hdf5(bs, bs_file)
    return 0


def _query_simbad(targetname):
    """Check on Simbad if the target is supposed to be a calibrator or a science
    source."""
    customSimbad = Simbad()
    customSimbad.add_votable_fields("otype")
    res = customSimbad.query_object(targetname)
    otype = res["OTYPE"][0]
    if otype == "Star":
        nrm_type = "CAL"
    else:
        nrm_type = "SCI"
    return nrm_type


def perform_clean(args):
    """CLI interface to clean the data with AMICAL."""
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
        raise OSError(
            "%s directory not found, check --datadir. AMICAL look for data only in this specified directory."
            % args.datadir
        )

    l_file = sorted(glob("%s/*.fits" % args.datadir))
    if len(l_file) == 0:
        raise OSError("No fits files found in %s, check --datadir." % args.datadir)

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
        for f in tqdm(l_file, ncols=100, desc="# files"):
            hdr = fits.open(f)[0].header
            hdr["HIERARCH AMICAL step"] = "CLEANED"
            cube = amical.select_clean_data(f, **clean_param, display=True)
            f_clean = f.split("/")[-1].split(".fits")[0] + "_cleaned.fits"
            fits.writeto(args.outdir + f_clean, cube, header=hdr, overwrite=True)
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
        f_clean = filename.split("/")[-1].split(".fits")[0] + "_cleaned.fits"
        fits.writeto(args.outdir + f_clean, cube, header=hdr, overwrite=True)
    return 0


def perform_extract(args):
    """CLI interface to extract the data with AMICAL (compute bispectrum object
    with all raw observables)."""
    cprint("---- AMICAL extract started ----", "cyan")
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
        raise OSError(
            "%s directory not found, check --datadir. AMICAL look for data only in this specified directory."
            % args.datadir
        )

    l_file = sorted(glob("%s/*.fits" % args.datadir))
    if len(l_file) == 0:
        raise OSError("No fits files found in %s, check --datadir." % args.datadir)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if not args.all:
        f = _select_data_file(args, process="extract")[0]
        _extract_bs_ifile(f, args, ami_param)
    else:
        for f in tqdm(l_file, ncols=100, desc="# files"):
            _extract_bs_ifile(f, args, ami_param)
    t1 = time.time() - t0
    cprint("---- AMICAL extract done (%2.1fs) ----" % t1, "cyan")
    if args.plot:
        plt.show(block=True)
    return 0


def perform_calibrate(args):
    """CLI interface to calibrate the data with AMICAL (save calibrated oifits
    files)."""

    sciname, calname = _select_association_file(args)

    bs_t = amical.load_bs_hdf5(sciname)

    bs_c = []
    for x in calname:
        bs_c = amical.load_bs_hdf5(x)

    display = False
    if len(bs_c) > 1:
        display = True

    cal = amical.calibrate(
        bs_t,
        bs_c,
        clip=args.clip,
        normalize_err_indep=args.norm,
        apply_atmcorr=args.atmcorr,
        apply_phscorr=args.phscorr,
        display=display,
    )

    # Position angle from North to East
    pa = bs_t.infos.pa

    cprint("\nPosition angle computed for the SCI data: pa = %2.3f deg" % pa, "cyan")

    # Display and save the results as oifits
    amical.show(cal, true_flag_t3=False, cmax=180, pa=pa)

    if args.plot:
        plt.show()

    oifits_file = Path(bs_t.infos.filename).stem + "_calibrated.fits"

    amical.save(cal, oifits_file=oifits_file, datadir=args.outdir)

    return 0

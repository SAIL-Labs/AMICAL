import os
import sys
from glob import glob
from pathlib import Path

from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from rich import print as rprint

import amical
from amical._rich_display import tabulate


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


def _select_association_file(args):
    """Show report with the data found and allow to select the science target
    (SCI) to be calibrated and the calibrator (CAL)."""
    l_file = sorted(glob(os.path.join(args.datadir, "*.h5")))

    if len(l_file) == 0:
        print(f"No h5 files found in {args.datadir}, check --datadir.", file=sys.stderr)
        return 1

    index_file = []

    headers = ["FILENAME", "TARGET", "DATE", "INSTRUM", "TYPE", "INDEX"]

    d = []
    for i, f in enumerate(l_file):
        bs = amical.load_bs_hdf5(f)
        filename = Path(f).stem
        hdr = bs.infos.hdr
        target = hdr.get("OBJECT")
        date = hdr.get("DATE-OBS")
        ins = hdr.get("INSTRUME")

        if target not in (None, "Unknown"):
            try:
                source = _query_simbad(target)
            except Exception:
                source = "Unknown"
        else:
            target = "Unknown"
            source = "Unknown"

        d.append([filename, target, date, ins, source, i])
        index_file.append(i)

    print(tabulate(d, headers=headers))

    text_calib = (
        "Which file used as calibrator? (use space "
        "between index if multiple calibrators are available).\n"
    )
    sci_index = int(input("\nWhich file to be calibrated?\n"))
    cal_index = [int(item) for item in input(text_calib).split()]

    try:
        sci_name = l_file[sci_index]
        cal_name = [l_file[x] for x in cal_index]
    except IndexError:
        print(
            f"Selected index (sci={sci_index}/cal={cal_index}) not valid "
            f"(only {len(l_file)} files found).",
            file=sys.stderr,
        )
        raise SystemExit  # noqa: B904
    return sci_name, cal_name


def perform_calibrate(args):
    """Calibrate the data with AMICAL (save calibrated oifits files)"""

    sciname, calname = _select_association_file(args)
    bs_t = amical.load_bs_hdf5(sciname)

    bs_c = []
    for x in calname:
        bs_c = amical.load_bs_hdf5(x)

    display = len(bs_c) > 1
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

    rprint(f"[cyan]\nPosition angle computed for the data: pa = {pa:2.3f} deg")

    # Display and save the results as oifits
    if args.plot:
        amical.show(cal, true_flag_t3=False, cmax=180, pa=pa)
        plt.show()

    oifits_file = Path(bs_t.infos.filename).stem + "_calibrated.fits"

    amical.save(cal, oifits_file=oifits_file, datadir=args.outdir)
    return 0

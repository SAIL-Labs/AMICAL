"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

OIFITS related function.

--------------------------------------------------------------------
"""
import datetime
import os

import numpy as np
from termcolor import cprint

from amical.tools import rad2mas

list_color = ["#00a7b5", "#afd1de", "#055c63", "#ce0058", "#8a8d8f", "#f1b2dc"]


def _compute_flag(value, sigma, limit=4.0):
    """Compute flag array using snr (snr < 4 by default)."""
    npts = len(value)
    flag = np.array([False] * npts)
    snr = abs(value / sigma)
    cond = snr <= limit
    flag[cond] = True
    return flag


def _peak1hole_cp(bs, ihole=0):
    """Get the indices of each CP including the given hole."""
    bs2bl_ix = bs.mask.bs2bl_ix
    bl2h_ix = bs.mask.bl2h_ix
    sel_ind = []
    for i, x in enumerate(bs2bl_ix.T):
        cpi = [bl2h_ix.T[y] for y in x]
        if (ihole in cpi[0]) or (ihole in cpi[1]) or (ihole in cpi[2]):
            sel_ind.append(i)
    sel_ind = np.array(sel_ind)
    return sel_ind


def _format_staindex_v2(tab):
    """Format the sta_index of v2 to save as oifits."""
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        line = np.array([ap1, ap2]) + 1
        sta_index.append(line)
    return sta_index


def _format_staindex_t3(tab):
    """Format the sta_index of cp to save as oifits."""
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        ap3 = int(x[2])
        line = np.array([ap1, ap2, ap3]) + 1
        sta_index.append(line)
    return sta_index


def _apply_flag(dict_calibrated, unit="arcsec"):
    """Apply flag and convert to appropriete units."""

    from munch import munchify as dict2class

    wl = dict_calibrated["OI_WAVELENGTH"]["EFF_WAVE"]
    uv_scale = {
        "m": 1,
        "rad": 1 / wl,
        "arcsec": 1 / wl / rad2mas(1e-3),
        "lambda": 1 / wl / 1e6,
    }

    U = dict_calibrated["OI_VIS2"]["UCOORD"] * uv_scale[unit]
    V = dict_calibrated["OI_VIS2"]["VCOORD"] * uv_scale[unit]

    flag_v2 = np.invert(dict_calibrated["OI_VIS2"]["FLAG"])
    V2 = dict_calibrated["OI_VIS2"]["VIS2DATA"][flag_v2]
    e_V2 = dict_calibrated["OI_VIS2"]["VIS2ERR"][flag_v2] * 1
    sp_freq_vis = dict_calibrated["OI_VIS2"]["BL"][flag_v2] * uv_scale[unit]
    flag_cp = np.invert(dict_calibrated["OI_T3"]["FLAG"])
    cp = dict_calibrated["OI_T3"]["T3PHI"][flag_cp]
    e_cp = dict_calibrated["OI_T3"]["T3PHIERR"][flag_cp]
    sp_freq_cp = dict_calibrated["OI_T3"]["BL"][flag_cp] * uv_scale[unit]
    bmax = 1.2 * np.max(np.sqrt(U**2 + V**2))

    cal_flagged = dict2class(
        {
            "U": U,
            "V": V,
            "bmax": bmax,
            "vis2": V2,
            "e_vis2": e_V2,
            "cp": cp,
            "e_cp": e_cp,
            "sp_freq_vis": sp_freq_vis,
            "sp_freq_cp": sp_freq_cp,
            "wl": wl[0],
            "band": dict_calibrated["info"]["FILT"],
        }
    )

    return cal_flagged


def wrap_raw(bs):
    """
    Wrap extraction product to save it as oifits

    `bs` : {munch.Munch}
        Object returned by amical.extract_bs() with raw observables,\n

    Returns
    --------
    `fake_cal` : {munch.Munch}
        Object that stores the raw observables in a format compatible with the
        output from amical.calibrate() and the input for `amical.save()`,\n
    """
    from munch import munchify as dict2class

    u1 = bs.u[bs.mask.bs2bl_ix[0, :]]
    v1 = bs.v[bs.mask.bs2bl_ix[0, :]]
    u2 = bs.u[bs.mask.bs2bl_ix[1, :]]
    v2 = bs.v[bs.mask.bs2bl_ix[1, :]]

    fake_cal = {
        "vis2": bs.vis2,
        "e_vis2": bs.e_vis2,
        "cp": bs.cp,
        "e_cp": bs.e_cp,
        "u": bs.u,
        "v": bs.v,
        "wl": bs.wl,
        "u1": u1,
        "v1": v1,
        "u2": u2,
        "v2": v2,
        "raw_t": bs,
        "raw_c": bs,
    }

    return dict2class(fake_cal)


def cal2dict(
    cal,
    target=None,
    pa=0,
    del_pa=0,
    snr=4,
    true_flag_v2=True,
    true_flag_t3=False,
    oriented=-1,
    ind_hole=None,
):
    """Format class containing calibrated data into appropriate dictionnary
    format to be saved as oifits files.

    Parameters
    ----------
    `cal` : {class}
        Class returned by amical.calibrate function (see core.py),\n
    `target` : {str}, (optional),
        Name of the science target, by default '',\n
    `fake_obj` : {bool}, (optional),
        If True, observables extracted from simulated data (celestial
        coordinates are omitted), by default False,\n
    `pa` : {int}, (optional)
        Position angle, by default 0 [deg]\n
    `del_pa` : {int}, (optional)
        Uncertainties of parallactic angle , by default 0\n
    `true_flag_v2` : {bool}, (optional)
        If True, real flag are computed for v2 using snr threshold
        (default 4), by default True,\n
    `true_flag_t3` : {bool}, (optional)
        If True, real flag are computed for cp using snr threshold
        (default 4), by default True,\n
    `oriented` {float}:
        If oriented == -1, east assumed to the left in the image, otherwise
        oriented == 1 (east to the right); (Default -1),\n

    Returns
    -------
    `dic`: {dict}
        Dictionnary format of the data to be save as oifits.
    """
    from astropy.time import Time

    res_t = cal.raw_t
    res_c = cal.raw_c
    n_baselines = res_t.mask.n_baselines
    n_bs = len(cal.cp)
    bl2h_ix = res_t.mask.bl2h_ix
    bs2bl_ix = res_t.mask.bs2bl_ix

    date = "2020-02-07T00:54:11"
    exp_time = 0.8
    try:
        ins = res_t.infos.instrument
        maskname = res_t.infos.maskname
        pixscale = res_t.infos.pixscale
    except KeyError:
        cprint(
            "Error: 'INSTRUME', 'NRMNAME' or 'PIXELSCL' are not in the header.", "red"
        )
        return None

    t = Time(date, format="isot", scale="utc")

    if true_flag_v2:
        flagV2 = _compute_flag(cal.vis2, cal.e_vis2, limit=snr)
    else:
        flagV2 = np.array([False] * n_baselines)

    if true_flag_t3:
        flagCP = _compute_flag(cal.cp, cal.e_cp, limit=snr)
    else:
        flagCP = np.array([False] * n_bs)

    sta_index_v2 = []
    for i in range(n_baselines):
        sta_index_v2.append(np.array(bl2h_ix[:, i]))
    sta_index_v2 = np.array(sta_index_v2)

    if target is None:
        target = res_t.infos.target

    thepa = pa - 0.5 * del_pa
    u, v = oriented * res_t.u, res_t.v
    u1 = u * np.cos(np.deg2rad(thepa)) + v * np.sin(np.deg2rad(thepa))
    v1 = -u * np.sin(np.deg2rad(thepa)) + v * np.cos(np.deg2rad(thepa))

    if type(res_c) is list:
        calib_name = res_c[0].infos.target
    else:
        calib_name = res_c.infos.target

    if ind_hole is not None:
        cprint("Select only independant CP using common hole #%i." % ind_hole, "green")
        sel_ind_cp = _peak1hole_cp(cal.raw_t, ind_hole)
    else:
        sel_ind_cp = np.arange(len(cal.cp))

    dic = {
        "OI_VIS2": {
            "VIS2DATA": cal.vis2,
            "VIS2ERR": cal.e_vis2,
            "UCOORD": u1,
            "VCOORD": v1,
            "STA_INDEX": sta_index_v2,
            "MJD": t.mjd,
            "INT_TIME": exp_time,
            "TIME": 0,
            "TARGET_ID": 1,
            "FLAG": flagV2,
            "BL": (u1**2 + v1**2) ** 0.5,
        },
        "OI_T3": {
            "MJD": t.mjd,
            "INT_TIME": exp_time,
            "T3PHI": cal.cp[sel_ind_cp],
            "T3PHIERR": cal.e_cp[sel_ind_cp],
            "TIME": 0,
            "T3AMP": np.zeros(len(cal.cp))[sel_ind_cp],
            "T3AMPERR": np.zeros(len(cal.cp))[sel_ind_cp],
            "U1COORD": u1[bs2bl_ix[0, :]][sel_ind_cp],
            "V1COORD": v1[bs2bl_ix[0, :]][sel_ind_cp],
            "U2COORD": u1[bs2bl_ix[1, :]][sel_ind_cp],
            "V2COORD": v1[bs2bl_ix[1, :]][sel_ind_cp],
            "STA_INDEX": list(np.array(res_t.mask.closing_tri)[sel_ind_cp]),
            "FLAG": flagCP[sel_ind_cp],
            "BL": res_t.bl_cp[sel_ind_cp],
        },
        "OI_WAVELENGTH": {
            "EFF_WAVE": np.array([res_t.wl]),
            "EFF_BAND": np.array([res_t.e_wl]),
        },
        "info": {
            "TARGET": target,
            "CALIB": calib_name,
            "FILT": res_t.infos.filtname,
            "INSTRUME": ins,
            "MASK": maskname,
            "MJD": t.mjd,
            "HDR": res_t.infos,
            "ISZ": res_t.infos.isz,
            "PSCALE": pixscale,
            "xycoord": res_t.mask.xycoord,
            "SEEING": res_t.infos.seeing,
        },
    }
    return dic


def load(filename, filtname=None):
    """Load an oifits file format and store it as dictionnary. The different keys are
    representative of the oifits standard structure ('OI_WAVELENGTH', 'OI_VIS2', etc.)

    Parameters
    ----------
    filename {str}:
        Name of the oifits file,\n
    filtname {str}:
        Name of the filter used if not included in the header (default: None)

    Output:
    -------
    dic {dict}:
        Dictionnary containing the oifits file data.
    """
    from astropy.io import fits

    fitsHandler = fits.open(filename)
    hdr = fitsHandler[0].header

    dic = {}
    for hdu in fitsHandler[1:]:
        if hdu.header["EXTNAME"] == "OI_WAVELENGTH":
            dic["OI_WAVELENGTH"] = {
                "EFF_WAVE": hdu.data["EFF_WAVE"],
                "EFF_BAND": hdu.data["EFF_BAND"],
            }

        if hdu.header["EXTNAME"] == "OI_VIS2":
            dic["OI_VIS2"] = {
                "VIS2DATA": hdu.data["VIS2DATA"],
                "VIS2ERR": hdu.data["VIS2ERR"],
                "UCOORD": hdu.data["UCOORD"],
                "VCOORD": hdu.data["VCOORD"],
                "STA_INDEX": hdu.data["STA_INDEX"],
                "MJD": hdu.data["MJD"],
                "INT_TIME": hdu.data["INT_TIME"],
                "TIME": hdu.data["TIME"],
                "TARGET_ID": hdu.data["TARGET_ID"],
                "FLAG": np.array(hdu.data["FLAG"]),
            }
            try:
                dic["OI_VIS2"]["BL"] = hdu.data["BL"]
            except KeyError:
                dic["OI_VIS2"]["BL"] = (
                    hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2
                ) ** 0.5

        if hdu.header["EXTNAME"] == "OI_VIS":
            dic["OI_VIS"] = {
                "TARGET_ID": hdu.data["TARGET_ID"],
                "TIME": hdu.data["TIME"],
                "MJD": hdu.data["MJD"],
                "INT_TIME": hdu.data["INT_TIME"],
                "VISAMP": hdu.data["VISAMP"],
                "VISAMPERR": hdu.data["VISAMPERR"],
                "VISPHI": hdu.data["VISPHI"],
                "VISPHIERR": hdu.data["VISPHIERR"],
                "UCOORD": hdu.data["UCOORD"],
                "VCOORD": hdu.data["VCOORD"],
                "STA_INDEX": hdu.data["STA_INDEX"],
                "FLAG": hdu.data["FLAG"],
            }
            try:
                dic["OI_VIS"]["BL"] = hdu.data["BL"]
            except KeyError:
                dic["OI_VIS"]["BL"] = (
                    hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2
                ) ** 0.5

        if hdu.header["EXTNAME"] == "OI_T3":
            u1 = hdu.data["U1COORD"]
            u2 = hdu.data["U2COORD"]
            v1 = hdu.data["V1COORD"]
            v2 = hdu.data["V2COORD"]
            u3 = -(u1 + u2)
            v3 = -(v1 + v2)
            bl_cp = []
            for k in range(len(u1)):
                B1 = np.sqrt(u1[k] ** 2 + v1[k] ** 2)
                B2 = np.sqrt(u2[k] ** 2 + v2[k] ** 2)
                B3 = np.sqrt(u3[k] ** 2 + v3[k] ** 2)
                bl_cp.append(np.max([B1, B2, B3]))  # rad-1
            bl_cp = np.array(bl_cp)

            dic["OI_T3"] = {
                "T3PHI": hdu.data["T3PHI"],
                "T3PHIERR": hdu.data["T3PHIERR"],
                "T3AMP": hdu.data["T3AMP"],
                "T3AMPERR": hdu.data["T3AMPERR"],
                "U1COORD": hdu.data["U1COORD"],
                "V1COORD": hdu.data["V1COORD"],
                "U2COORD": hdu.data["U2COORD"],
                "V2COORD": hdu.data["V2COORD"],
                "STA_INDEX": hdu.data["STA_INDEX"],
                "MJD": hdu.data["MJD"],
                "FLAG": hdu.data["FLAG"],
                "TARGET_ID": hdu.data["TARGET_ID"],
                "TIME": hdu.data["TIME"],
                "INT_TIME": hdu.data["INT_TIME"],
            }
            try:
                dic["OI_T3"]["BL"] = hdu.data["FREQ"]
            except KeyError:
                dic["OI_T3"]["BL"] = bl_cp

    fitsHandler.close()

    dic["info"] = {h: hdr[h] for h in hdr}  # {'MJD': mjd,

    if "FILT" not in list(dic["info"].keys()):
        if filtname is None:
            print("No filter in info or as input param?")
        else:
            dic["info"]["FILT"] = filtname
    return dic


def loadc(filename):
    """Same as load but provide an easy usable output as a class format (output.v2, or output.cp)."""
    from munch import munchify as dict2class

    dic = load(filename)
    res = {}
    # Extract infos
    res["target"] = dic["info"].get("OBJECT")

    res["calib"] = dic["info"].get("CALIB")
    res["seeing"] = dic["info"].get("SEEING")
    res["mjd"] = dic["info"].get("MJD")
    if res["mjd"] is None:
        res["mjd"] = dic["info"].get("MJD-OBS")

    # Extract wavelength
    res["wl"] = dic["OI_WAVELENGTH"]["EFF_WAVE"]
    res["e_wl"] = dic["OI_WAVELENGTH"]["EFF_BAND"]

    # Extract squared visibilities
    res["vis2"] = dic["OI_VIS2"]["VIS2DATA"]
    res["e_vis2"] = dic["OI_VIS2"]["VIS2ERR"]
    res["u"] = dic["OI_VIS2"]["UCOORD"]
    res["v"] = dic["OI_VIS2"]["VCOORD"]
    res["bl"] = dic["OI_VIS2"]["BL"]
    res["flag_vis"] = dic["OI_VIS2"]["FLAG"]

    # Extract closure phases
    res["cp"] = dic["OI_T3"]["T3PHI"]
    res["e_cp"] = dic["OI_T3"]["T3PHIERR"]
    res["u1"] = dic["OI_T3"]["U1COORD"]
    res["v1"] = dic["OI_T3"]["V1COORD"]
    res["u2"] = dic["OI_T3"]["U2COORD"]
    res["v2"] = dic["OI_T3"]["V2COORD"]
    res["bl_cp"] = dic["OI_T3"]["BL"]
    res["flag_cp"] = dic["OI_T3"]["FLAG"]

    return dict2class(res)


def save(
    observables,
    oifits_file=None,
    datadir="Saveoifits",
    pa=0,
    ind_hole=None,
    fake_obj=False,
    true_flag_v2=True,
    true_flag_t3=False,
    snr=4,
    verbose=False,
    *,
    origin=None,
    raw=False,
):
    """
    Summary:
    --------

    Save the class object (from calibrate function) into oifits format. The input
    observables can be a list of object for IFU data (e.g.: IFS-SPHERE).

    Parameters:
    -----------

    `observables` {class}:
        Class or list of class containing all calibrated interferometric variable
        extracted using calibrate (amical.calibration) function,\n
    `oifits_file` {str}:
        Name of the oifits file,\n
    `datadir` {str}:
        Folder name save the oifits files,\n
    `pa` {float}:
        Position angle of the observation (i.e.: north direction) [deg],\n
    `ind_hole` {int}:
        By default, ind_hole is None, all the CP are considered ncp = N(N-1)(N-2)/6. If
        ind_hole is set, save only the independant CP including the given hole
        ncp = (N-1)(N-2)/2.\n
    `fake_obj` {bool}:
        If True, observable are extracted from simulated data and so doesn't
        contain real target informations (simbad search is ignored),\n
    `true_flag_v2`, `true_flag_t3` {bool}:
        if True, the true flag are used using snr,\n
    `snr` {float}:
        Limit snr used to compute flags (default=4),\n
    `verbose` {bool}:
        If True, print useful informations,
    `origin` {str}:
        String to use as ORIGIN key in oifits. 'Sydney University' is used if origin is
        `None` (default=None),\n
    `raw` {bool}:
        Set to True if the input is not calibrated. This will only silence the warning
        shown otherwise when an uncalibrated input is detected (default=False).\n

    Returns:
    --------
    `dic` {dict}:
        Oifits formated dictionnary,\n
    `savedfile` {str}:
        Name of the saved oifits file.

    """
    from astroquery.simbad import Simbad
    from astropy.io import fits

    if observables is None:
        cprint("\nError save : Wrong data format!", on_color="on_red")
        return None

    if oifits_file is None:
        print("Error: oifits filename is not given, please specify oifits_file.")
        return None

    if type(observables) is not list:
        observables = [observables]

    if not isinstance(origin, (str, type(None))):
        raise TypeError("origin should be a str or None")

    l_dic = []
    for iobs in observables:
        # If keys correspond to raw observables, make format compatible with calibrated
        if "cp" in iobs and "raw_t" not in iobs:
            if not raw and verbose:
                msg = (
                    "The input seems to contain uncalibrated observables."
                    " Saving raw observables as oifits is provided only for"
                    " convenience. To re-use the observables in amical.calibrate(),"
                    " they should be saved to a pickle file. Use raw=True to turn this"
                    " warning off."
                )
                cprint(f"Warning: {msg}", "green")
            iobs = wrap_raw(iobs)
        idic = cal2dict(
            iobs,
            pa=pa,
            true_flag_v2=true_flag_v2,
            true_flag_t3=true_flag_t3,
            ind_hole=ind_hole,
            snr=snr,
        )
        l_dic.append(idic)
    dic = l_dic[0]

    if not os.path.exists(datadir):
        print("### Create %s directory to save all requested Oifits ###" % datadir)
        os.mkdir(datadir)

    # ------------------------------
    #       Creation OIFITS
    # ------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :" % (oifits_file))

    # refdate = datetime.datetime(2000, 1, 1)  # Unix time reference
    hdulist = fits.HDUList()

    hdr = dic["info"].get("HDR", {})
    hdu = fits.PrimaryHDU()
    hdu.header["DATE"] = datetime.datetime.now().strftime(
        format="%F"
    )  # , 'Creation date'
    hdu.header["ORIGIN"] = origin or hdr.get("ORIGIN", "Sydney University")
    hdu.header["CONTENT"] = "OIFITS2"
    hdu.header["DATE-OBS"] = hdr.get("date-obs", "")
    hdu.header["TELESCOP"] = hdr.get("telescop", "")
    hdu.header["INSTRUME"] = hdr.get("INSTRUME", dic["info"]["INSTRUME"])
    hdu.header["OBSERVER"] = hdr.get("OBSERVER", "")
    hdu.header["OBJECT"] = hdr.get("OBJECT", dic["info"]["TARGET"])
    hdu.header["INSMODE"] = "NRM"
    hdu.header["FILT"] = dic["info"]["FILT"]
    hdu.header["MJD"] = dic["info"]["MJD"]
    hdu.header["MASK"] = dic["info"]["MASK"]
    hdu.header["SEEING"] = dic["info"].get("SEEING", "")
    hdu.header["CALIB"] = dic["info"]["CALIB"]

    hdulist.append(hdu)
    # ------------------------------
    #        OI Wavelength
    # ------------------------------

    if verbose:
        print("-> Including OI Wavelength table...")
    data = dic["OI_WAVELENGTH"]

    EFF_WAVE, EFF_BAND = [], []
    for dic in l_dic:
        d = dic["OI_WAVELENGTH"]
        EFF_WAVE.append(d["EFF_WAVE"][0])
        EFF_BAND.append(d["EFF_BAND"][0])

    iwl = len(EFF_WAVE)
    # Data
    # -> Initiation new hdu table :
    hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            (
                fits.Column(
                    name="EFF_WAVE", format="1E", unit="METERS", array=EFF_WAVE
                ),
                fits.Column(
                    name="EFF_BAND", format="1E", unit="METERS", array=EFF_BAND
                ),
            )
        )
    )

    # Header
    hdu.header["EXTNAME"] = "OI_WAVELENGTH"
    hdu.header["OI_REVN"] = 2  # , 'Revision number of the table definition'
    # 'Name of detector, for cross-referencing'
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdulist.append(hdu)  # Add current HDU to the final fits file.

    # ------------------------------
    #          OI Target
    # ------------------------------
    if verbose:
        print("-> Including OI Target table...")

    name_star = dic["info"]["TARGET"]

    customSimbad = Simbad()
    customSimbad.add_votable_fields("propermotions", "sptype", "parallax")

    # Add information from Simbad:
    if fake_obj:
        ra = pmra = dec = pmdec = plx = [0]
        spectyp = ["fake"]
    else:
        if (name_star is not None) & (name_star != "Unknown"):
            from astropy import units as u
            from astropy.coordinates import SkyCoord

            try:
                query = customSimbad.query_object(name_star)
                coord = SkyCoord(
                    query["RA"][0] + " " + query["DEC"][0], unit=(u.hourangle, u.deg)
                )
                ra = [coord.ra.deg]
                dec = [coord.dec.deg]
                spectyp = query["SP_TYPE"]
                pmra = query["PMRA"]
                pmdec = query["PMDEC"]
                plx = query["PLX_VALUE"]
            except Exception:
                ra = pmra = dec = pmdec = plx = [0]
                spectyp = ["fake"]
        else:
            ra = pmra = dec = pmdec = plx = [0]
            spectyp = ["fake"]

    if name_star == "":
        name_star = "Unknown"

    hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            (
                fits.Column(name="TARGET_ID", format="1I", array=[1]),
                fits.Column(name="TARGET", format="16A", array=[name_star]),
                fits.Column(name="RAEP0", format="1D", unit="DEGREES", array=ra),
                fits.Column(name="DECEP0", format="1D", unit="DEGREES", array=dec),
                fits.Column(name="EQUINOX", format="1E", unit="YEARS", array=[2000]),
                fits.Column(name="RA_ERR", format="1D", unit="DEGREES", array=[0]),
                fits.Column(name="DEC_ERR", format="1D", unit="DEGREES", array=[0]),
                fits.Column(name="SYSVEL", format="1D", unit="M/S", array=[0]),
                fits.Column(name="VELTYP", format="8A", array=["UNKNOWN"]),
                fits.Column(name="VELDEF", format="8A", array=["OPTICAL"]),
                fits.Column(name="PMRA", format="1D", unit="DEG/YR", array=pmra),
                fits.Column(name="PMDEC", format="1D", unit="DEG/YR", array=pmdec),
                fits.Column(name="PMRA_ERR", format="1D", unit="DEG/YR", array=[0]),
                fits.Column(name="PMDEC_ERR", format="1D", unit="DEG/YR", array=[0]),
                fits.Column(name="PARALLAX", format="1E", unit="DEGREES", array=plx),
                fits.Column(name="PARA_ERR", format="1E", unit="DEGREES", array=[0]),
                fits.Column(name="SPECTYP", format="16A", array=spectyp),
            )
        )
    )

    hdu.header["EXTNAME"] = "OI_TARGET"
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdulist.append(hdu)

    # ------------------------------
    #           OI Array
    # ------------------------------

    if verbose:
        print("-> Including OI Array table...")

    staxy = dic["info"]["xycoord"]
    N_ap = len(staxy)
    telName = ["A%i" % x for x in np.arange(N_ap) + 1]
    staName = telName
    diameter = [0] * N_ap

    staxyz = []
    for x in staxy:
        a = list(x)
        line = [a[0], a[1], 0]
        staxyz.append(line)

    staIndex = np.arange(N_ap) + 1

    pscale = dic["info"]["PSCALE"] / 1000.0  # arcsec
    isz = dic["info"]["ISZ"]  # Size of the image to extract NRM data
    fov = [pscale * isz] * N_ap
    fovtype = ["RADIUS"] * N_ap

    hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            (
                fits.Column(name="TEL_NAME", format="16A", array=telName),
                fits.Column(name="STA_NAME", format="16A", array=staName),
                fits.Column(name="STA_INDEX", format="1I", array=staIndex),
                fits.Column(
                    name="DIAMETER", unit="METERS", format="1E", array=diameter
                ),
                fits.Column(name="STAXYZ", unit="METERS", format="3D", array=staxyz),
                fits.Column(name="FOV", unit="ARCSEC", format="1D", array=fov),
                fits.Column(name="FOVTYPE", format="6A", array=fovtype),
            )
        )
    )

    hdu.header["EXTNAME"] = "OI_ARRAY"
    hdu.header["ARRAYX"] = float(0)
    hdu.header["ARRAYY"] = float(0)
    hdu.header["ARRAYZ"] = float(0)
    hdu.header["ARRNAME"] = dic["info"]["MASK"]
    hdu.header["FRAME"] = "SKY"
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS2
    # ------------------------------
    if verbose:
        print("-> Including OI Vis2 table...")

    data = dic["OI_VIS2"]
    if type(data["TARGET_ID"]) != np.array:
        npts = len(data["VIS2DATA"])
    else:
        npts = 1

    if type(data["TARGET_ID"]) == int:
        npts = len(dic["OI_VIS2"]["VIS2DATA"])
        targetId = [data["TARGET_ID"]] * npts
        time = [data["TIME"]] * npts
        mjd = [data["MJD"]] * npts
        intTime = [data["INT_TIME"]] * npts
    else:
        npts = 1
        targetId = [1] * len(data["VIS2DATA"])
        time = data["TIME"]
        mjd = data["MJD"]
        intTime = data["INT_TIME"]

    staIndex = _format_staindex_v2(data["STA_INDEX"])

    VIS2DATA, VIS2ERR, FLAG = [], [], []
    for dic in l_dic:
        d = dic["OI_VIS2"]
        VIS2DATA.append(d["VIS2DATA"])
        VIS2ERR.append(d["VIS2ERR"])
        FLAG.append(d["FLAG"])
    VIS2DATA = np.array(VIS2DATA).T
    VIS2ERR = np.array(VIS2ERR).T
    FLAG = np.array(FLAG).T

    hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(name="TARGET_ID", format="1I", array=targetId),
                fits.Column(name="TIME", format="1D", unit="SECONDS", array=time),
                fits.Column(name="MJD", unit="DAY", format="1D", array=mjd),
                fits.Column(
                    name="INT_TIME", format="1D", unit="SECONDS", array=intTime
                ),
                fits.Column(name="VIS2DATA", format="%iD" % iwl, array=VIS2DATA),
                fits.Column(name="VIS2ERR", format="%iD" % iwl, array=VIS2ERR),
                fits.Column(
                    name="UCOORD", format="1D", unit="METERS", array=data["UCOORD"]
                ),
                fits.Column(
                    name="VCOORD", format="1D", unit="METERS", array=data["VCOORD"]
                ),
                fits.Column(name="STA_INDEX", format="2I", array=staIndex),
                fits.Column(name="FLAG", format="%iL" % iwl, array=FLAG),
            ]
        )
    )

    hdu.header["EXTNAME"] = "OI_VIS2"
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdu.header["ARRNAME"] = dic["info"]["MASK"]
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdu.header["DATE-OBS"] = hdr.get("date-obs", "")
    hdulist.append(hdu)

    # ------------------------------
    #           OI T3
    # ------------------------------
    if verbose:
        print("-> Including OI T3 table...")

    data = dic["OI_T3"]
    try:
        check_oi = type(float(data["MJD"]))
    except TypeError:
        check_oi = int

    if check_oi == float:
        t3phi = dic["OI_T3"]["T3PHI"]
        npts = len(t3phi)
        targetId = np.ones_like(t3phi)
        time = np.zeros_like(t3phi)
        mjd = [data["MJD"]] * npts
        intTime = [data["INT_TIME"]] * npts
    else:
        npts = 1
        targetId = np.ones_like(data["T3PHI"])
        time = data["TIME"]
        mjd = data["MJD"]
        intTime = data["INT_TIME"]

    T3AMP, T3AMPERR, T3PHI, T3PHIERR, FLAG = [], [], [], [], []
    for dic in l_dic:
        d = dic["OI_T3"]
        T3AMP.append(d["T3AMP"])
        T3AMPERR.append(d["T3AMPERR"])
        T3PHI.append(d["T3PHI"])
        T3PHIERR.append(d["T3PHIERR"])
        FLAG.append(d["FLAG"])
    T3AMP = np.array(T3AMP).T
    T3AMPERR = np.array(T3AMPERR).T
    T3PHI = np.array(T3PHI).T
    T3PHIERR = np.array(T3PHIERR).T
    FLAG = np.array(FLAG).T

    staIndex = _format_staindex_t3(data["STA_INDEX"])
    hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            (
                fits.Column(name="TARGET_ID", format="1I", array=targetId),
                fits.Column(name="TIME", format="1D", unit="SECONDS", array=time),
                fits.Column(name="MJD", format="1D", unit="DAY", array=mjd),
                fits.Column(
                    name="INT_TIME", format="1D", unit="SECONDS", array=intTime
                ),
                fits.Column(name="T3AMP", format="%iD" % iwl, array=T3AMP),
                fits.Column(name="T3AMPERR", format="%iD" % iwl, array=T3AMPERR),
                fits.Column(
                    name="T3PHI", format="%iD" % iwl, unit="DEGREES", array=T3PHI
                ),
                fits.Column(
                    name="T3PHIERR", format="%iD" % iwl, unit="DEGREES", array=T3PHIERR
                ),
                fits.Column(
                    name="U1COORD", format="1D", unit="METERS", array=data["U1COORD"]
                ),
                fits.Column(
                    name="V1COORD", format="1D", unit="METERS", array=data["V1COORD"]
                ),
                fits.Column(
                    name="U2COORD", format="1D", unit="METERS", array=data["U2COORD"]
                ),
                fits.Column(
                    name="V2COORD", format="1D", unit="METERS", array=data["V2COORD"]
                ),
                fits.Column(name="STA_INDEX", format="3I", array=staIndex),
                fits.Column(name="FLAG", format="%iL" % iwl, array=FLAG),
            )
        )
    )

    hdu.header["EXTNAME"] = "OI_T3"
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdu.header["ARRNAME"] = dic["info"]["MASK"]
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdu.header["DATE-OBS"] = hdr.get("date-obs", "")
    hdulist.append(hdu)

    # ------------------------------
    #          Save file
    # ------------------------------
    savedfile = os.path.join(datadir, oifits_file)
    hdulist.writeto(savedfile, overwrite=True)
    if verbose:
        cprint("\n\n### OIFITS CREATED (%s)." % oifits_file, "cyan")

    if len(l_dic) == 1:
        l_dic = l_dic[0]
    return l_dic, savedfile


def _plot_UV(ax1, l_dic, dic_color, diffWl=False):
    l_band_al, l_bmax = [], []
    for dic in l_dic:
        tmp = _apply_flag(dic)
        U = tmp.U
        V = tmp.V
        band = tmp.band
        wl = tmp.wl
        label = rf"{wl * 1e6:2.2f} $\mu m$ ({band})"
        if diffWl:
            c1, c2 = dic_color[band], dic_color[band]
            if band not in l_band_al:
                label = rf"{wl * 1e6:2.2f} $\mu m$ ({band})"
                l_band_al.append(band)
            else:
                label = ""
        else:
            c1, c2 = "#00adb5", "#fc5185"
        l_bmax.append(tmp.bmax)

        ax1.scatter(
            U, V, s=50, c=c1, label=label, edgecolors="#364f6b", marker="o", alpha=1
        )
        ax1.scatter(
            -1 * np.array(U),
            -1 * np.array(V),
            c=c2,
            s=50,
            edgecolors="#364f6b",
            marker="o",
            alpha=1,
        )
    return l_bmax


def _plot_V2(ax2, l_dic, dic_color, diffWl=False):
    max_f_vis = []
    for dic in l_dic:
        tmp = _apply_flag(dic, unit="arcsec")
        V2 = tmp.vis2
        e_V2 = tmp.e_vis2
        sp_freq_vis = tmp.sp_freq_vis
        max_f_vis.append(np.max(sp_freq_vis))
        band = tmp.band
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = "#00adb5"

        ax2.errorbar(
            sp_freq_vis,
            V2,
            yerr=e_V2,
            linestyle="None",
            capsize=1,
            mfc=mfc,
            ecolor="#364f6b",
            mec="#364f6b",
            marker=".",
            elinewidth=0.5,
            alpha=1,
            ms=9,
        )
    return max_f_vis


def _plot_CP(ax3, l_dic, dic_color, conv_cp, diffWl=False):
    max_f_cp = []
    for dic in l_dic:
        tmp = _apply_flag(dic, unit="arcsec")
        cp = tmp.cp * conv_cp
        e_cp = tmp.e_cp * conv_cp
        sp_freq_cp = tmp.sp_freq_cp
        max_f_cp.append(np.max(sp_freq_cp))
        band = tmp.band
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = "#00adb5"

        ax3.errorbar(
            sp_freq_cp,
            cp,
            yerr=e_cp,
            linestyle="None",
            capsize=1,
            mfc=mfc,
            ecolor="#364f6b",
            mec="#364f6b",
            marker=".",
            elinewidth=0.5,
            alpha=1,
            ms=9,
        )
    return max_f_cp


def _plot_UV_ifu(ax1, fig, l_dic):
    l_bmax = []
    all_U, all_V, all_wl = [], [], []

    for dic in l_dic:
        tmp = _apply_flag(dic)
        U = tmp.U
        V = tmp.V
        wl = tmp.wl
        l_bmax.append(tmp.bmax)
        all_U.append(U)
        all_V.append(V)
        all_wl.append([wl] * len(U))
    all_U = np.array(all_U)
    all_V = np.array(all_V)
    all_wl = np.array(all_wl) * 1e6

    ax1.scatter(
        all_U,
        all_V,
        s=40,
        c=all_wl,
        marker="o",
        edgecolors="#364f6b",
        alpha=1,
        linewidth=0.1,
        cmap="jet",
    )
    sc = ax1.scatter(
        -all_U,
        -all_V,
        s=40,
        c=all_wl,
        marker="o",
        edgecolors="#364f6b",
        alpha=1,
        linewidth=0.1,
        cmap="jet",
    )

    position = fig.add_axes([0.22, 0.95, 0.1, 0.015])
    fig.colorbar(sc, cax=position, orientation="horizontal", drawedges=False)
    ax1.text(
        0.53, 0.98, r"$\lambda$ [µm]", ha="center", va="center", transform=ax1.transAxes
    )
    return l_bmax


def _plot_V2_ifu(ax2, l_dic):
    max_f_vis = []
    all_V2, all_e_V2, all_freq, all_wl = [], [], [], []
    for dic in l_dic:
        tmp = _apply_flag(dic, unit="arcsec")
        V2 = tmp.vis2
        e_V2 = tmp.e_vis2
        sp_freq_vis = tmp.sp_freq_vis
        max_f_vis.append(np.max(sp_freq_vis))

        all_V2.extend(V2)
        all_e_V2.extend(e_V2)
        all_freq.extend(sp_freq_vis)
        all_wl.extend([tmp.wl] * len(V2))

    all_V2 = np.array(all_V2)
    all_e_V2 = np.array(all_e_V2)
    all_freq = np.array(all_freq)
    all_wl = np.array(all_wl)

    ax2.errorbar(
        all_freq,
        all_V2,
        yerr=all_e_V2,
        linestyle="None",
        capsize=1,
        ecolor="#364f6b",
        mec="#364f6b",
        marker="None",
        elinewidth=0.5,
        alpha=1,
        ms=9,
    )
    ax2.scatter(
        all_freq,
        all_V2,
        s=20,
        c=all_wl,
        zorder=20,
        linewidth=0.5,
        marker="o",
        edgecolors="#364f6b",
        alpha=1,
        cmap="jet",
    )

    return max_f_vis


def _plot_CP_ifu(ax3, l_dic, conv_cp):
    max_f_cp = []
    all_CP, all_e_cp, all_freq, all_wl = [], [], [], []
    for dic in l_dic:
        tmp = _apply_flag(dic, unit="arcsec")
        cp = tmp.cp * conv_cp
        e_cp = tmp.e_cp * conv_cp
        sp_freq_cp = tmp.sp_freq_cp
        max_f_cp.append(np.max(sp_freq_cp))

        all_CP.extend(cp)
        all_e_cp.extend(e_cp)
        all_freq.extend(sp_freq_cp)
        all_wl.extend([tmp.wl] * len(cp))

    ax3.errorbar(
        all_freq,
        all_CP,
        yerr=all_e_cp,
        linestyle="None",
        capsize=1,
        ecolor="#364f6b",
        mec="#364f6b",
        marker="None",
        elinewidth=0.5,
        alpha=1,
        ms=9,
    )
    ax3.scatter(
        all_freq,
        all_CP,
        s=20,
        c=all_wl,
        zorder=20,
        linewidth=0.5,
        marker="o",
        edgecolors="#364f6b",
        alpha=1,
        cmap="jet",
    )
    return max_f_cp


def show(
    inputList,
    diffWl=False,
    ind_hole=None,
    vmin=0,
    vmax=1.05,
    cmax=180,
    setlog=False,
    pa=0,
    unit="arcsec",
    unit_cp="deg",
    snr=4,
    true_flag_v2=True,
    true_flag_t3=False,
):
    """Show oifits data of a multiple dataset (loaded with oifits.load or oifits filename).

    Parameters:
    -----------
    `diffWl` {bool}:
        If True, differentiate the file (wavelenghts) by color,\n
    `ind_hole` {int}:
        By default, ind_hole is None, all the CP are considered ncp = N(N-1)(N-2)/6. If
        ind_hole is set, show only the independant CP including the given hole
        ncp = (N-1)(N-2)/2.\n
    `vmin`, `vmax` {float}:
        Minimum and maximum visibilities (default: 0, 1.05),\n
    `cmax` {float}:
        Maximum closure phase [deg] (default: 180),\n
    `setlog` {bool}:
        If True, the visibility curve is plotted in log scale,\n
    `unit` {str}:
        Unit of the sp. frequencies (default: 'arcsec'),\n
    `unit_cp` {str}:
        Unit of the closure phases (default: 'deg'),\n
    `true_flag_v2` {bool}:
        If inputs are classes from amical.calibrate, compute the true flag of vis2
        using snr parameter (default: True).\n
    `true_flag_t3` {bool}:
        If inputs are classes from amical.calibrate, compute the true flag of cp
        using snr parameter (default: True),\n
    `snr` {float}:
        If inputs are classes from amical.calibrate, use snr param to compute flag,
    """
    import matplotlib.pyplot as plt

    if type(inputList) is not list:
        inputList = [inputList]

    try:
        inputList[0].v2
        isclass = True
    except AttributeError:
        isclass = False

    if isclass:
        l_dic = [
            cal2dict(
                x,
                pa=pa,
                true_flag_v2=true_flag_v2,
                ind_hole=ind_hole,
                true_flag_t3=true_flag_t3,
                snr=snr,
            )
            for x in inputList
        ]
        print("\n -- SHOW -- Inputs are classes from amical.calibrate:")
        print("-> (Check true_flag_v2, true_flag_t3 and snr parameters)\n")
    elif type(inputList[0]) is str:
        l_dic = [load(x) for x in inputList]
        print("Inputs are oifits filename.")
    elif type(inputList[0]) is dict:
        l_dic = inputList
        print("Inputs are dict from amical.load.")
    else:
        print("Wrong input format.")
        return None

    dic_color = {}
    i_c = 0
    for dic in l_dic:
        filt = dic["info"]["FILT"]
        if filt not in dic_color.keys():
            dic_color[filt] = list_color[i_c]
            i_c += 1

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    textstr = "PA = %2.1f deg" % pa

    fontsize = 14
    fig = plt.figure(figsize=(16, 5.5))
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)
    ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=4)

    # Plot plan UV
    # -------
    ins = l_dic[0]["info"]["INSTRUME"]
    if ("IFS" in ins) & (len(l_dic) > 1):
        l_bmax = _plot_UV_ifu(ax1, fig, l_dic)
    else:
        l_bmax = _plot_UV(ax1, l_dic, dic_color, diffWl=False)

    Bmax = np.max(l_bmax)
    ax1.axis([Bmax, -Bmax, -Bmax, Bmax])
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.patch.set_facecolor("#f7f9fc")
    ax1.patch.set_alpha(1)
    ax1.xaxis.set_ticks_position("none")
    ax1.yaxis.set_ticks_position("none")
    if diffWl:
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax1.legend(handles, labels, loc="best", fontsize=9)

    plt.text(
        0.02,
        0.98,
        textstr,
        transform=ax1.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=props,
    )

    unitlabel = {
        "m": "m",
        "rad": "rad$^{-1}$",
        "arcsec": "arcsec$^{-1}$",
        "lambda": r"M$\lambda$",
    }

    ax1.set_xlabel(r"U [%s]" % unitlabel[unit], fontsize=fontsize)
    ax1.set_ylabel(r"V [%s]" % unitlabel[unit], fontsize=fontsize)
    ax1.grid(alpha=0.2)

    # Plot V2
    # -------
    if ("IFS" in ins) & (len(l_dic) > 1):
        max_f_vis = _plot_V2_ifu(ax2, l_dic)
    else:
        max_f_vis = _plot_V2(ax2, l_dic, dic_color, diffWl=diffWl)

    ax2.hlines(1, 0, 1.2 * np.max(max_f_vis), lw=1, color="k", alpha=0.2, ls="--")

    ax2.set_ylim([vmin, vmax])
    ax2.set_xlim([0, 1.2 * np.max(max_f_vis)])
    ax2.set_ylabel(r"$V^2$", fontsize=fontsize)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.patch.set_facecolor("#f7f9fc")
    ax2.patch.set_alpha(1)
    ax2.xaxis.set_ticks_position("none")
    ax2.yaxis.set_ticks_position("none")

    if setlog:
        ax2.set_yscale("log")
    ax2.grid(which="both", alpha=0.2)

    # Plot CP
    # -------
    if unit_cp == "rad":
        conv_cp = np.pi / 180.0
        h1 = np.pi
    else:
        conv_cp = 1
        h1 = np.rad2deg(np.pi)

    cmin = -cmax

    if ("IFS" in ins) & (len(l_dic) > 1):
        max_f_cp = _plot_CP_ifu(ax3, l_dic, conv_cp)
    else:
        max_f_cp = _plot_CP(ax3, l_dic, dic_color, conv_cp, diffWl=diffWl)

    ax3.hlines(h1, 0, 1.2 * np.max(max_f_cp), lw=1, color="k", alpha=0.2, ls="--")
    ax3.hlines(-h1, 0, 1.2 * np.max(max_f_cp), lw=1, color="k", alpha=0.2, ls="--")
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.patch.set_facecolor("#f7f9fc")
    ax3.patch.set_alpha(1)
    ax3.xaxis.set_ticks_position("none")
    ax3.yaxis.set_ticks_position("none")
    ax3.set_xlabel("Spatial frequency [arcsec$^{-1}$]", fontsize=fontsize)
    ax3.set_ylabel(r"Clos. $\phi$ [%s]" % unit_cp, fontsize=fontsize)
    ax3.axis([0, 1.2 * np.max(max_f_cp), cmin * conv_cp, cmax * conv_cp])
    ax3.grid(which="both", alpha=0.2)

    plt.subplots_adjust(
        top=0.974, bottom=0.1, left=0.05, right=0.99, hspace=0.127, wspace=0.35
    )

    return fig

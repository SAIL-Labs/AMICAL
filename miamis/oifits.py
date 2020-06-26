# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

--------------------------------------------------------------------
MIAMIS: Multi-Instruments Aperture Masking Interferometry Software
--------------------------------------------------------------------

OIFITS related function.

-------------------------------------------------------------------- 
"""

import datetime
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from termcolor import cprint

from miamis.tools import rad2mas

list_color = ['#00a7b5', '#afd1de', '#055c63', '#ce0058', '#8a8d8f', '#f1b2dc']


def computeflag(value, sigma, limit=4.):
    """ Compute flag array using snr (snr < 4 by default). """
    npts = len(value)
    flag = np.array([False] * npts)
    snr = abs(value/sigma)
    cond = (snr <= limit)
    flag[cond] = True
    return flag


def cal2dict(cal, target=None, fake_obj=False, pa=0, del_pa=0, snr=4,
             true_flag_v2=True, true_flag_t3=False, include_vis=False,
             oriented=-1, nfile=1):
    """ Format class containing calibrated data into appropriate dictionnary 
    format to be saved as oifits files.

    Parameters
    ----------
    `cal` : {class}
        Class returned by calib_v2_cp function (see oifits.py),\n
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
    `include_vis` : {bool}, (optional)
        If True, include visibility amplitude in the oifits dictionnary,
         by default False,\n
    `oriented` {float}:
        If oriented == -1, east assumed to the left in the image, otherwise 
        oriented == 1 (east to the right); (Default -1),\n
    `nfile` : {int}, (optional)
        Iteration number of the file (used to save multiple oifits file), by default 1.

    Returns
    -------
    `dic`: {dict}
        Dictionnary format of the data to be save as oifits.
    """
    res_t = cal.raw_t
    res_c = cal.raw_c
    n_baselines = res_t.n_baselines
    n_bs = len(cal.cp)
    bl2h_ix = res_t.bl2h_ix
    bs2bl_ix = res_t.bs2bl_ix

    date = '2020-02-07T00:54:11'
    exp_time = 0.8
    try:
        ins = res_t.hdr['INSTRUME']
        nrmnamr = res_t.hdr['NRMNAME']
        pixscale = res_t.hdr['PIXELSCL']
    except KeyError:
        cprint("Error: 'INSTRUME', 'NRMNAME' or 'PIXELSCL' are not in the header.", 'red')
        return None

    t = Time(date, format='isot', scale='utc')

    if true_flag_v2:
        flagV2 = computeflag(cal.vis2, cal.e_vis2, limit=snr)
    else:
        flagV2 = np.array([False] * n_baselines)

    if true_flag_t3:
        flagCP = computeflag(cal.cp, cal.e_cp, limit=snr)
    else:
        flagCP = np.array([False] * n_bs)

    sta_index_v2 = []
    for i in range(n_baselines):
        sta_index_v2.append(np.array(bl2h_ix[:, i]))
    sta_index_v2 = np.array(sta_index_v2)

    if target is None:
        target = res_t.target
    else:
        pass

    thepa = pa - 0.5 * del_pa
    u, v = oriented*res_t.u, res_t.v
    u1 = u*np.cos(np.deg2rad(thepa)) + v*np.sin(np.deg2rad(thepa))
    v1 = -u*np.sin(np.deg2rad(thepa)) + v*np.cos(np.deg2rad(thepa))

    if type(res_c) is list:
        calib_name = res_c[0].target
    else:
        calib_name = res_c.target

    dic = {'OI_VIS2': {'VIS2DATA': cal.vis2,
                       'VIS2ERR': cal.e_vis2,
                       'UCOORD': u1,
                       'VCOORD': v1,
                       'STA_INDEX': sta_index_v2,
                       'MJD': t.mjd,
                       'INT_TIME': exp_time,
                       'TIME': 0,
                       'TARGET_ID': 1,
                       'FLAG': flagV2,
                       'BL': (u1**2+v1**2)**0.5},
           'OI_T3': {'MJD': t.mjd,
                     'INT_TIME': exp_time,
                     'T3PHI': cal.cp,
                     'T3PHIERR': cal.e_cp,
                     'TIME': 0,
                     'T3AMP': np.zeros(len(cal.cp)),
                     'T3AMPERR': np.zeros(len(cal.cp)),
                     'U1COORD': u1[bs2bl_ix[0, :]],
                     'V1COORD': v1[bs2bl_ix[0, :]],
                     'U2COORD': u1[bs2bl_ix[1, :]],
                     'V2COORD': v1[bs2bl_ix[1, :]],
                     'STA_INDEX': res_t.closing_tri,
                     'FLAG': flagCP,
                     'BL': res_t.bl_cp
                     },
           'OI_WAVELENGTH': {'EFF_WAVE': np.array([res_t.wl]),
                             'EFF_BAND': np.array([res_t.e_wl])},
           'info': {'TARGET': target,
                    'CALIB': calib_name,
                    'FILT': res_t.filtname,
                    'INSTRUME': ins,
                    'MASK': nrmnamr,
                    'MJD': t.mjd,
                    'HDR': res_t.hdr,
                    'ISZ': res_t.isz,
                    'PSCALE': pixscale,
                    'NFILE': nfile,
                    'xycoord': res_t.xycoord,
                    'SEEING': res_t.hdr['SEEING']}
           }

    if include_vis:
        dic['OI_VIS'] = {'TARGET_ID': 1,
                         'TIME': 0,
                         'MJD': t.mjd,
                         'INT_TIME': exp_time,
                         'VISAMP': cal.visamp,
                         'VISAMPERR': cal.e_visamp,
                         'VISPHI': cal.visphi,
                         'VISPHIERR': cal.e_visphi,
                         'UCOORD': u1,
                         'VCOORD': v1,
                         'STA_INDEX': sta_index_v2,
                         'FLAG': flagV2
                         }
    return dic


def data2obs(data, use_flag=True, cond_wl=False, cond_uncer=False, rel_max=None, wl_min=None, wl_max=None, verbose=True):
    """
    Convert and select data from the dict format (miamis.load or miamis.cal2dict).

    Parameters:
    -----------

    data: {dict}
        Dictionnary containing all the data.
    use_flag: {boolean}
        If True, use flag from the original oifits file.
    cond_wl: {boolean}
        If True, apply wavelenght restriction between wl_min and wl_max.
    wl_min, wl_max: {float}
        if cond_wl, limits of the wavelength domain [µm]
    cond_uncer: {boolean}
        If True, select the best data according their relative uncertainties (rel_max).
    rel_max: {float}
        if cond_uncer, maximum sigma uncertainties allowed [%].
    verbose: {boolean}
        If True, display useful information about the data selection.


    Return:
    -------

    Obs: {tuple}
        Tuple containing all the selected data in an appropriate format to perform the fit.

    """
    nbl = len(data['OI_VIS2']['VIS2DATA'])
    ncp = len(data['OI_T3']['T3PHI'])
    nwl = len(data['OI_WAVELENGTH']['EFF_WAVE'])

    vis2_data = data['OI_VIS2']['VIS2DATA'].flatten()
    e_vis2_data = data['OI_VIS2']['VIS2ERR'].flatten()
    flag_V2 = data['OI_VIS2']['FLAG'].flatten()

    cp_data = data['OI_T3']['T3PHI'].flatten()
    e_cp_data = data['OI_T3']['T3PHIERR'].flatten()
    flag_CP = data['OI_T3']['FLAG'].flatten()

    if not use_flag:
        flag_V2 = [False]*len(vis2_data)
        flag_CP = [False]*len(cp_data)

    u_data, v_data = [], []
    u1_data, v1_data, u2_data, v2_data = [], [], [], []

    for i in range(nbl):
        for j in range(nwl):
            u_data.append(data['OI_VIS2']['UCOORD'][i])
            v_data.append(data['OI_VIS2']['VCOORD'][i])

    for i in range(ncp):
        for j in range(nwl):
            u1_data.append(data['OI_T3']['U1COORD'][i])
            v1_data.append(data['OI_T3']['V1COORD'][i])
            u2_data.append(data['OI_T3']['U2COORD'][i])
            v2_data.append(data['OI_T3']['V2COORD'][i])

    u_data, v_data = np.array(u_data), np.array(v_data)
    u1_data = np.array(u1_data)
    v1_data = np.array(v1_data)
    u2_data = np.array(u2_data)
    v2_data = np.array(v2_data)

    wl_data = np.array(list(data['OI_WAVELENGTH']['EFF_WAVE'])*nbl)
    wl_data_cp = np.array(list(data['OI_WAVELENGTH']['EFF_WAVE'])*ncp)

    obs = []
    for i in range(nbl*nwl):
        if not (flag_V2[i] & use_flag):
            if not cond_wl:
                tmp = [u_data[i], v_data[i], wl_data[i]]
                typ = 'V2'
                obser = vis2_data[i]
                err = e_vis2_data[i]
                if cond_uncer:
                    if (err/obser <= rel_max*1e-2):
                        obs.append([tmp, typ, obser, err])
                else:
                    obs.append([tmp, typ, obser, err])

            else:
                if (wl_data[i] >= wl_min*1e-6) & (wl_data[i] <= wl_max*1e-6):
                    tmp = [u_data[i], v_data[i], wl_data[i]]
                    typ = 'V2'
                    obser = vis2_data[i]
                    err = e_vis2_data[i]
                    if cond_uncer:
                        if (err/obser <= rel_max*1e-2):
                            obs.append([tmp, typ, obser, err])
                    else:
                        obs.append([tmp, typ, obser, err])

    N_v2_rest = len(obs)

    for i in range(ncp*nwl):
        if not flag_CP[i]:
            if not cond_wl:
                tmp = [u1_data[i], u2_data[i], -(u1_data[i]+u2_data[i]), v1_data[i], v2_data[i],
                       -(v1_data[i]+v2_data[i]), wl_data_cp[i]]
                typ = 'CP'
                obser = cp_data[i]
                err = e_cp_data[i]
                if cond_uncer:
                    if (err/obser <= rel_max*1e-2):
                        obs.append([tmp, typ, obser, err])
                else:
                    obs.append([tmp, typ, obser, err])
            else:
                if (wl_data_cp[i] >= wl_min*1e-6) & (wl_data_cp[i] <= wl_max*1e-6):
                    tmp = [u1_data[i], u2_data[i], -(u1_data[i]+u2_data[i]), v1_data[i], v2_data[i],
                           -(v1_data[i]+v2_data[i]), wl_data_cp[i]]
                    typ = 'CP'
                    obser = cp_data[i]
                    err = e_cp_data[i]
                    if cond_uncer:
                        if (err/obser <= rel_max*1e-2):
                            obs.append([tmp, typ, obser, err])
                        else:
                            pass
                    else:
                        obs.append([tmp, typ, obser, err])

    N_cp_rest = len(obs) - N_v2_rest

    Obs = np.array(obs)
    if verbose:
        print('\nTotal # of data points: %i (%i V2, %i CP)' %
              (len(Obs), N_v2_rest, N_cp_rest))
        if use_flag:
            print('-> Flag in oifits files used.')
        if cond_wl:
            print(r'-> Restriction on wavelenght: %2.2f < %s < %2.2f µm' %
                  (wl_min, chr(955), wl_max))
        if cond_uncer:
            print(r'-> Restriction on uncertainties: %s < %2.1f %%' %
                  (chr(949), rel_max))
    return Obs


def Format_STAINDEX_V2(tab):
    """ Format the sta_index of v2 to save as oifits. """
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        line = np.array([ap1, ap2]) + 1
        sta_index.append(line)
    return sta_index


def Format_STAINDEX_T3(tab):
    """ Format the sta_index of cp to save as oifits. """
    sta_index = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        ap3 = int(x[2])
        line = np.array([ap1, ap2, ap3]) + 1
        sta_index.append(line)
    return sta_index


def load(filename, target=None, ins=None, mask=None, filtname=None, include_vis=True):
    """[summary]

    Parameters
    ----------
    filename : [type]
        [description]
    """
    fitsHandler = fits.open(filename)
    hdr = fitsHandler[0].header

    dic = {}
    for hdu in fitsHandler[1:]:
        if hdu.header['EXTNAME'] == 'OI_WAVELENGTH':
            dic['OI_WAVELENGTH'] = {'EFF_WAVE': hdu.data['EFF_WAVE'],
                                    'EFF_BAND': hdu.data['EFF_BAND'],
                                    }

        if hdu.header['EXTNAME'] == 'OI_VIS2':
            dic['OI_VIS2'] = {'VIS2DATA': hdu.data['VIS2DATA'],
                              'VIS2ERR': hdu.data['VIS2ERR'],
                              'UCOORD': hdu.data['UCOORD'],
                              'VCOORD': hdu.data['VCOORD'],
                              'STA_INDEX': hdu.data['STA_INDEX'],
                              'MJD': hdu.data['MJD'],
                              'INT_TIME': hdu.data['INT_TIME'],
                              'TIME': hdu.data['TIME'],
                              'TARGET_ID': hdu.data['TARGET_ID'],
                              'FLAG': np.array(hdu.data['FLAG']),
                              }
            try:
                dic['OI_VIS2']['BL'] = hdu.data['BL']
            except KeyError:
                dic['OI_VIS2']['BL'] = (
                    hdu.data['UCOORD']**2 + hdu.data['VCOORD']**2)**0.5

        if hdu.header['EXTNAME'] == 'OI_VIS':
            dic['OI_VIS'] = {'TARGET_ID': hdu.data['TARGET_ID'],
                             'TIME': hdu.data['TIME'],
                             'MJD': hdu.data['MJD'],
                             'INT_TIME': hdu.data['INT_TIME'],
                             'VISAMP': hdu.data['VISAMP'],
                             'VISAMPERR': hdu.data['VISAMPERR'],
                             'VISPHI': hdu.data['VISPHI'],
                             'VISPHIERR': hdu.data['VISPHIERR'],
                             'UCOORD': hdu.data['UCOORD'],
                             'VCOORD': hdu.data['VCOORD'],
                             'STA_INDEX': hdu.data['STA_INDEX'],
                             'FLAG': hdu.data['FLAG'],
                             }
            try:
                dic['OI_VIS']['BL'] = hdu.data['BL']
            except KeyError:
                dic['OI_VIS']['BL'] = (
                    hdu.data['UCOORD']**2 + hdu.data['VCOORD']**2)**0.5

        if hdu.header['EXTNAME'] == 'OI_T3':
            u1 = hdu.data['U1COORD']
            u2 = hdu.data['U2COORD']
            v1 = hdu.data['V1COORD']
            v2 = hdu.data['V2COORD']
            u3 = -(u1+u2)
            v3 = -(v1+v2)
            bl_cp = []
            for k in range(len(u1)):
                B1 = np.sqrt(u1[k]**2+v1[k]**2)
                B2 = np.sqrt(u2[k]**2+v2[k]**2)
                B3 = np.sqrt(u3[k]**2+v3[k]**2)
                bl_cp.append(np.max([B1, B2, B3]))  # rad-1
            bl_cp = np.array(bl_cp)

            dic['OI_T3'] = {'T3PHI': hdu.data['T3PHI'],
                            'T3PHIERR': hdu.data['T3PHIERR'],
                            'T3AMP': hdu.data['T3AMP'],
                            'T3AMPERR': hdu.data['T3AMPERR'],
                            'U1COORD': hdu.data['U1COORD'],
                            'V1COORD': hdu.data['V1COORD'],
                            'U2COORD': hdu.data['U2COORD'],
                            'V2COORD': hdu.data['V2COORD'],
                            'STA_INDEX': hdu.data['STA_INDEX'],
                            'MJD': hdu.data['MJD'],
                            'FLAG': hdu.data['FLAG'],
                            'TARGET_ID': hdu.data['TARGET_ID'],
                            'TIME': hdu.data['TIME'],
                            'INT_TIME': hdu.data['INT_TIME'],
                            }
            try:
                dic['OI_T3']['BL'] = hdu.data['FREQ']
            except KeyError:
                dic['OI_T3']['BL'] = bl_cp

    dic['info'] = {h: hdr[h] for h in hdr}  # {'MJD': mjd,

    if 'FILT' not in list(dic['info'].keys()):

        if filtname is None:
            print('No filter in info or as input param?')
        else:
            dic['info']['FILT'] = filtname
    return dic


def loadc(filename):
    """ Same as load but provide an easy usable output as a class format (output.v2, or output.cp). """
    dic = load(filename)

    res = {}
    # Extract infos
    res['target'] = dic['info']['OBJECT']
    res['calib'] = dic['info']['CALIB']
    res['seeing'] = dic['info']['SEEING']
    res['mjd'] = dic['info']['MJD']

    # Extract wavelength
    res['wl'] = dic['OI_WAVELENGTH']['EFF_WAVE']
    res['e_wl'] = dic['OI_WAVELENGTH']['EFF_BAND']

    # Extract squared visibilities
    res['vis2'] = dic['OI_VIS2']['VIS2DATA']
    res['e_vis2'] = dic['OI_VIS2']['VIS2ERR']
    res['u'] = dic['OI_VIS2']['UCOORD']
    res['v'] = dic['OI_VIS2']['VCOORD']
    res['bl'] = dic['OI_VIS2']['BL']
    res['flag_vis'] = dic['OI_VIS2']['FLAG']

    # Extract closure phases
    res['cp'] = dic['OI_T3']['T3PHI']
    res['e_cp'] = dic['OI_T3']['T3PHIERR']
    res['u1'] = dic['OI_T3']['U1COORD']
    res['v1'] = dic['OI_T3']['V1COORD']
    res['u2'] = dic['OI_T3']['U2COORD']
    res['v2'] = dic['OI_T3']['V2COORD']
    res['bl_cp'] = dic['OI_T3']['BL']
    res['flag_cp'] = dic['OI_T3']['FLAG']

    return dict2class(res)


def save(cal, oifits_file=None, fake_obj=False,
         pa=0, include_vis=False,
         true_flag_v2=True, true_flag_t3=False, snr=4,
         datadir='Saveoifits/', nfile=1, verbose=False):
    """
    Summary:
    --------

    Save the class object (from calibrate function) into oifits format.

    Parameters:
    -----------

    `cal` {class}: 
        Class containing all calibrated interferometric variable extracted using
        calibrate (miamis.core) function,\n
    `oifits_file` {str}:
        Name of the oifits file, if None a default name using useful 
        information is used (target, instrument, filter, mask, etc.),\n
    `include_vis` {bool}:
        If True, include OI_VIS table in the oifits,\n
    `fake_obj` {bool}:
        If True, observable are extracted from simulated data and so doesn't
        contain real target informations (simbad search is ignored),\n
    `pa` {float}:
        Position angle of the observation (i.e.: north direction) [deg],\n
    `true_flag_v2`, `true_flag_t3` {bool}:
        if True, the true flag are used using snr,\n
    `snr` {float}:
        Limit snr used to compute flags (default=4),\n
    `datadir` {str}:
        Folder name save the oifits files,\n
    `nfile` {int}:
        Integer number to include in the oifits file name (easly save 
        mulitple iterations).\n 
    `verbose` {bool}:
        If True, print useful informations.

    Returns:
    --------
    `dic` {dict}:
        Oifits formated dictionnary,\n
    `savedfile` {str}:
        Name of the saved oifits file.

    """

    if cal is None:
        cprint('\nError NRMtoOifits2 : Wrong data format!', on_color='on_red')
        return None

    if type(cal) != dict:
        dic = cal2dict(cal, pa=pa, include_vis=include_vis,
                       fake_obj=fake_obj, nfile=nfile,
                       true_flag_v2=true_flag_v2, true_flag_t3=true_flag_t3)
    else:
        dic = cal.copy()

    if not os.path.exists(datadir):
        print('### Create %s directory to save all requested Oifits ###' % datadir)
        os.system('mkdir %s' % datadir)

    if type(oifits_file) == str:
        filename = oifits_file
    else:
        filename = '%s_%s_%s_%s_%2.0f_%i.oifits' % (dic['info']['TARGET'],
                                                    dic['info']['INSTRUME'], dic['info']['MASK'],
                                                    dic['info']['FILT'], dic['info']['MJD'],
                                                    dic['info']['NFILE'])

    # ------------------------------
    #       Creation OIFITS
    # ------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :" % (filename))

    refdate = datetime.datetime(2000, 1, 1)  # Unix time reference

    hdulist = fits.HDUList()

    try:
        hdr = dic['info']['HDR']
    except KeyError:
        hdr = {}

    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = datetime.datetime.now().strftime(
        format='%F')  # , 'Creation date'
    hdu.header['ORIGIN'] = 'Sydney University'
    hdu.header['CONTENT'] = 'OIFITS2'
    try:
        hdu.header['DATE-OBS'] = hdr['DATE-OBS']
    except KeyError:
        hdu.header['DATE-OBS'] = ''
    try:
        hdu.header['TELESCOP'] = hdr['TELESCOP']
    except KeyError:
        hdu.header['TELESCOP'] = 'JWST'
    try:
        hdu.header['INSTRUME'] = hdr['INSTRUME']
    except KeyError:
        hdu.header['INSTRUME'] = 'NIRISS'
    try:
        hdu.header['OBSERVER'] = hdr['OBSERVER']
    except KeyError:
        hdu.header['OBSERVER'] = 'me'
    try:
        hdu.header['OBJECT'] = hdr['OBJECT']
    except KeyError:
        hdu.header['OBJECT'] = dic['info']['TARGET']

    hdu.header['INSMODE'] = 'NRM'
    hdu.header['FILT'] = dic['info']['FILT']
    hdu.header['MJD'] = dic['info']['MJD']
    hdu.header['MASK'] = dic['info']['MASK']
    try:
        hdu.header['SEEING'] = dic['info']['SEEING']
    except ValueError:
        hdu.header['SEEING'] = 0.0

    hdu.header['CALIB'] = dic['info']['CALIB']

    hdulist.append(hdu)
    # ------------------------------
    #        OI Wavelength
    # ------------------------------

    if verbose:
        print('-> Including OI Wavelength table...')
    data = dic['OI_WAVELENGTH']

    # Data
    # -> Initiation new hdu table :
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='EFF_WAVE', format='1E',
                    unit='METERS', array=[data['EFF_WAVE']]),
        fits.Column(name='EFF_BAND', format='1E',
                    unit='METERS', array=[data['EFF_BAND']])
    )))

    # Header
    hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
    hdu.header['OI_REVN'] = 2  # , 'Revision number of the table definition'
    # 'Name of detector, for cross-referencing'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdulist.append(hdu)  # Add current HDU to the final fits file.

    # ------------------------------
    #          OI Target
    # ------------------------------
    if verbose:
        print('-> Including OI Target table...')

    name_star = dic['info']['TARGET']

    customSimbad = Simbad()
    customSimbad.add_votable_fields('propermotions', 'sptype', 'parallax')

    # Add information from Simbad:
    if fake_obj:
        ra = [0]
        dec = [0]
        spectyp = ['fake']
        pmra = [0]
        pmdec = [0]
        plx = [0]
    else:
        try:
            query = customSimbad.query_object(name_star)
            coord = SkyCoord(query['RA'][0]+' '+query['DEC']
                             [0], unit=(u.hourangle, u.deg))

            ra = [coord.ra.deg]
            dec = [coord.dec.deg]
            spectyp = query['SP_TYPE']
            pmra = query['PMRA']
            pmdec = query['PMDEC']
            plx = query['PLX_VALUE']
        except Exception:
            ra = [0]
            dec = [0]
            spectyp = ['fake']
            pmra = [0]
            pmdec = [0]
            plx = [0]

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TARGET_ID', format='1I', array=[1]),
        fits.Column(name='TARGET', format='16A', array=[name_star]),
        fits.Column(name='RAEP0', format='1D', unit='DEGREES', array=ra),
        fits.Column(name='DECEP0', format='1D', unit='DEGREES', array=dec),
        fits.Column(name='EQUINOX', format='1E', unit='YEARS', array=[2000]),
        fits.Column(name='RA_ERR', format='1D', unit='DEGREES', array=[0]),
        fits.Column(name='DEC_ERR', format='1D', unit='DEGREES', array=[0]),
        fits.Column(name='SYSVEL', format='1D', unit='M/S', array=[0]),
        fits.Column(name='VELTYP', format='8A', array=['UNKNOWN']),
        fits.Column(name='VELDEF', format='8A', array=['OPTICAL']),
        fits.Column(name='PMRA', format='1D', unit='DEG/YR', array=pmra),
        fits.Column(name='PMDEC', format='1D', unit='DEG/YR', array=pmdec),
        fits.Column(name='PMRA_ERR', format='1D', unit='DEG/YR', array=[0]),
        fits.Column(name='PMDEC_ERR', format='1D', unit='DEG/YR', array=[0]),
        fits.Column(name='PARALLAX', format='1E', unit='DEGREES', array=plx),
        fits.Column(name='PARA_ERR', format='1E', unit='DEGREES', array=[0]),
        fits.Column(name='SPECTYP', format='16A', array=spectyp)
    )))

    hdu.header['EXTNAME'] = 'OI_TARGET'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdulist.append(hdu)

    # ------------------------------
    #           OI Array
    # ------------------------------

    if verbose:
        print('-> Including OI Array table...')

    if 'xycoord' in list(dic['info'].keys()):
        staxy = dic['info']['xycoord']
        N_ap = len(staxy)
        telName = ['A%i' % x for x in np.arange(N_ap)+1]
        staName = telName
        diameter = [0] * N_ap

        staxyz = []
        for x in staxy:
            a = list(x)
            line = [a[0], a[1], 0]
            staxyz.append(line)

        staIndex = np.arange(N_ap) + 1

        pscale = dic['info']['PSCALE']/1000.  # arcsec
        isz = dic['info']['ISZ']  # Size of the image to extract NRM data
        fov = [pscale * isz] * N_ap
        fovtype = ['RADIUS'] * N_ap
    else:
        if N_ap is None:
            cprint(
                'Mask coordinates not included but are necessary to create oifits file:', 'red')
            cprint(
                '-> give the number of apertures of the mask (N_ap) as input.', 'red')
            return None
        telName = ['A%i' % x for x in np.arange(N_ap)+1]
        staName = ['A%i' % x for x in np.arange(N_ap)+1]
        diameter = [0] * N_ap
        staIndex = np.arange(N_ap) + 1
        staxyz = []
        for x in np.arange(N_ap):
            line = [x, x, 0]
            staxyz.append(line)
        try:
            pscale = dic['info']['PSCALE']/1000.  # arcsec
        except KeyError:
            pscale = 0
        fov = [0] * N_ap
        fovtype = ['RADIUS'] * N_ap

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TEL_NAME', format='16A',
                    array=telName),  # ['dummy']),
        fits.Column(name='STA_NAME', format='16A',
                    array=staName),  # ['dummy']),
        fits.Column(name='STA_INDEX', format='1I', array=staIndex),
        fits.Column(name='DIAMETER', unit='METERS',
                    format='1E', array=diameter),
        fits.Column(name='STAXYZ', unit='METERS', format='3D', array=staxyz),
        fits.Column(name='FOV', unit='ARCSEC', format='1D', array=fov),
        fits.Column(name='FOVTYPE', format='6A', array=fovtype),
    )))

    hdu.header['EXTNAME'] = 'OI_ARRAY'
    hdu.header['ARRAYX'] = float(0)
    hdu.header['ARRAYY'] = float(0)
    hdu.header['ARRAYZ'] = float(0)
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['FRAME'] = 'SKY'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'

    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS
    # ------------------------------

    if include_vis:
        if verbose:
            print('-> Including OI Vis table...')

        data = dic['OI_VIS']
        if type(data['TARGET_ID']) is int:
            npts = len(dic['OI_VIS']['VISAMP'])
        else:
            npts = 1

        staIndex = Format_STAINDEX_V2(data['STA_INDEX'])
        if type(data['MJD']) is not float:
            mjd = data['MJD'][0]
        else:
            mjd = data['MJD']

        hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='TARGET_ID', format='1I',
                        array=[data['TARGET_ID']]*npts),
            fits.Column(name='TIME', format='1D', unit='SECONDS',
                        array=[data['TIME']]*npts),
            fits.Column(name='MJD', unit='DAY', format='1D',
                        array=[data['MJD']]*npts),
            fits.Column(name='INT_TIME', format='1D',
                        unit='SECONDS', array=[data['INT_TIME']]*npts),
            fits.Column(name='VISAMP', format='1D', array=data['VISAMP']),
            fits.Column(name='VISAMPERR', format='1D',
                        array=data['VISAMPERR']),
            fits.Column(name='VISPHI', format='1D', unit='DEGREES',
                        array=np.rad2deg(data['VISPHI'])),
            fits.Column(name='VISPHIERR', format='1D', unit='DEGREES',
                        array=np.rad2deg(data['VISPHIERR'])),
            fits.Column(name='UCOORD', format='1D',
                        unit='METERS', array=data['UCOORD']),
            fits.Column(name='VCOORD', format='1D',
                        unit='METERS', array=data['VCOORD']),
            fits.Column(name='STA_INDEX', format='2I', array=staIndex),
            fits.Column(name='FLAG', format='1L', array=data['FLAG'])
        ]))

        hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
        hdu.header['EXTNAME'] = 'OI_VIS'
        hdu.header['INSNAME'] = dic['info']['INSTRUME']
        hdu.header['ARRNAME'] = dic['info']['MASK']
        hdu.header['DATE-OBS'] = refdate.strftime(
            '%F'), 'Zero-point for table (UTC)'
        hdulist.append(hdu)
    # except:
        # pass

    # ------------------------------
    #           OI VIS2
    # ------------------------------

    if verbose:
        print('-> Including OI Vis2 table...')

    data = dic['OI_VIS2']
    if type(data['TARGET_ID']) != np.array:
        npts = len(dic['OI_VIS2']['VIS2DATA'])
    else:
        npts = 1

    if type(data['TARGET_ID']) == int:
        npts = len(dic['OI_VIS2']['VIS2DATA'])
        targetId = [data['TARGET_ID']]*npts
        time = [data['TIME']]*npts
        mjd = [data['MJD']]*npts
        intTime = [data['INT_TIME']]*npts
    else:
        npts = 1
        targetId = [1]*len(data['VIS2DATA'])
        time = data['TIME']
        mjd = data['MJD']
        intTime = data['INT_TIME']

    staIndex = Format_STAINDEX_V2(data['STA_INDEX'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name='TARGET_ID', format='1I',
                    array=targetId),
        fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=time),
        fits.Column(name='MJD', unit='DAY', format='1D',
                    array=mjd),
        fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=intTime),
        fits.Column(name='VIS2DATA', format='1D', array=data['VIS2DATA']),
        fits.Column(name='VIS2ERR', format='1D', array=data['VIS2ERR']),
        fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD']),
        fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD']),
        fits.Column(name='STA_INDEX', format='2I', array=staIndex),
        fits.Column(name='FLAG', format='1L', array=data['FLAG'])
    ]))

    hdu.header['EXTNAME'] = 'OI_VIS2'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = refdate.strftime(
        '%F'), 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI T3
    # ------------------------------
    if verbose:
        print('-> Including OI T3 table...')

    data = dic['OI_T3']

    try:
        check_oi = type(float(data['MJD']))
    except TypeError:
        check_oi = int

    if check_oi == float:
        npts = len(dic['OI_T3']['T3AMP'])
        targetId = [1]*npts
        time = [0]*npts
        mjd = [data['MJD']]*npts
        intTime = [data['INT_TIME']]*npts
    else:
        npts = 1
        targetId = [1]*len(data['T3AMP'])
        time = data['TIME']
        mjd = data['MJD']
        intTime = data['INT_TIME']

    staIndex = Format_STAINDEX_T3(data['STA_INDEX'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TARGET_ID', format='1I', array=targetId),
        fits.Column(name='TIME', format='1D', unit='SECONDS', array=time),
        fits.Column(name='MJD', format='1D', unit='DAY',
                    array=mjd),
        fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=intTime),
        fits.Column(name='T3AMP', format='1D', array=data['T3AMP']),
        fits.Column(name='T3AMPERR', format='1D', array=data['T3AMPERR']),
        fits.Column(name='T3PHI', format='1D', unit='DEGREES',
                    array=data['T3PHI']),
        fits.Column(name='T3PHIERR', format='1D', unit='DEGREES',
                    array=data['T3PHIERR']),
        fits.Column(name='U1COORD', format='1D',
                    unit='METERS', array=data['U1COORD']),
        fits.Column(name='V1COORD', format='1D',
                    unit='METERS', array=data['V1COORD']),
        fits.Column(name='U2COORD', format='1D',
                    unit='METERS', array=data['U2COORD']),
        fits.Column(name='V2COORD', format='1D',
                    unit='METERS', array=data['V2COORD']),
        fits.Column(name='STA_INDEX', format='3I', array=staIndex),
        fits.Column(name='FLAG', format='1L', array=data['FLAG'])
    )))

    hdu.header['EXTNAME'] = 'OI_T3'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = refdate.strftime(
        '%F'), 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #          Save file
    # ------------------------------
    hdulist.writeto(datadir + filename, overwrite=True)
    if verbose:
        cprint('\n\n### OIFITS CREATED (%s).' % filename, 'cyan')

    savedfile = datadir+filename
    return dic, savedfile


def ApplyFlag(dic1, unit='arcsec'):
    """ Apply flag and convert to appropriete units."""

    wl = dic1['OI_WAVELENGTH']['EFF_WAVE']
    uv_scale = {'m': 1,
                'rad': 1/wl,
                'arcsec': 1/wl/rad2mas(1e-3),
                'lambda': 1/wl/1e6}

    U = dic1['OI_VIS2']['UCOORD']*uv_scale[unit]
    V = dic1['OI_VIS2']['VCOORD']*uv_scale[unit]

    flag_v2 = np.invert(dic1['OI_VIS2']['FLAG'])
    V2 = dic1['OI_VIS2']['VIS2DATA'][flag_v2]
    e_V2 = dic1['OI_VIS2']['VIS2ERR'][flag_v2] * 1
    sp_freq_vis = dic1['OI_VIS2']['BL'][flag_v2] * uv_scale[unit]
    flag_cp = np.invert(dic1['OI_T3']['FLAG'])
    cp = dic1['OI_T3']['T3PHI'][flag_cp]
    e_cp = dic1['OI_T3']['T3PHIERR'][flag_cp]
    sp_freq_cp = dic1['OI_T3']['BL'][flag_cp] * uv_scale[unit]
    bmax = 1.2*np.max(np.sqrt(U**2+V**2))

    return U, V, bmax, V2, e_V2, cp, e_cp, sp_freq_vis, sp_freq_cp, wl[0], dic1['info']['FILT']


def show(inputList, diffWl=False, vmin=0, vmax=1.05, cmax=180, setlog=False, pa=0,
         unit='arcsec', unit_cp='deg', snr=3, true_flag_v2=True, true_flag_t3=False):
    """ Show oifits data of a multiple dataset (loaded with oifits.load or oifits filename).

    Parameters:
    -----------
    `diffWl` {bool}:
        If True, differentiate the file (wavelenghts) by color,\n
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
        If inputs are classes from miamis.calibrate, compute the true flag of vis2 
        using snr parameter (default: True).\n
    `true_flag_t3` {bool}:
        If inputs are classes from miamis.calibrate, compute the true flag of cp 
        using snr parameter (default: True),\n
    `snr` {float}:
        If inputs are classes from miamis.calibrate, use snr param to compute flag,
    """

    if type(inputList) is not list:
        inputList = [inputList]

    try:
        inputList[0].v2
        isclass = True
    except AttributeError:
        isclass = False

    if isclass:
        l_dic = [cal2dict(x, pa=pa, true_flag_v2=true_flag_v2,
                          true_flag_t3=true_flag_t3, snr=snr) for x in inputList]
        print('\n -- SHOW -- Inputs are classes from miamis.calibrate:')
        print('-> (Check true_flag_v2, true_flag_t3 and snr parameters)\n')
    elif type(inputList[0]) is str:
        l_dic = [load(x) for x in inputList]
        print('Inputs are oifits filename.')
    elif type(inputList[0]) is dict:
        l_dic = inputList
        print('Inputs are dict from miamis.load.')

    # return None

    dic_color = {}
    i_c = 0
    for dic in l_dic:
        filt = dic['info']['FILT']
        if filt not in dic_color.keys():
            dic_color[filt] = list_color[i_c]
            i_c += 1

    fig = plt.figure(figsize=(16, 5.5))
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)
    ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=4)

    # Plot plan UV
    # -------
    l_bmax, l_band_al = [], []
    for dic in l_dic:
        tmp = ApplyFlag(dic)
        U = tmp[0]
        V = tmp[1]
        band = tmp[10]
        wl = tmp[9]
        label = '%2.2f $\mu m$ (%s)' % (wl*1e6, band)
        if diffWl:
            c1, c2 = dic_color[band], dic_color[band]
            if band not in l_band_al:
                label = '%2.2f $\mu m$ (%s)' % (wl*1e6, band)
                l_band_al.append(band)
            else:
                label = ''
        else:
            c1, c2 = '#00adb5', '#fc5185'
        l_bmax.append(tmp[2])

        ax1.scatter(U, V, s=50, c=c1, label=label,
                    edgecolors='#364f6b', marker='o', alpha=1)
        ax1.scatter(-1*np.array(U), -1*np.array(V), s=50, c=c2,
                    edgecolors='#364f6b', marker='o', alpha=1)

    Bmax = np.max(l_bmax)
    ax1.axis([Bmax, -Bmax, -Bmax, Bmax])
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.patch.set_facecolor('#f7f9fc')
    ax1.patch.set_alpha(1)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    if diffWl:
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(
            *sorted(zip(labels, handles), key=lambda t: t[0]))
        ax1.legend(handles, labels, loc='best', fontsize=9)
        # ax1.legend(loc='best')

    unitlabel = {'m': 'm',
                 'rad': 'rad$^{-1}$',
                 'arcsec': 'arcsec$^{-1}$',
                 'lambda': 'M$\lambda$'}

    ax1.set_xlabel(r'U [%s]' % unitlabel[unit])
    ax1.set_ylabel(r'V [%s]' % unitlabel[unit])
    ax1.grid(alpha=0.2)

    # Plot V2
    # -------
    max_f_vis = []
    for dic in l_dic:
        tmp = ApplyFlag(dic, unit='arcsec')
        V2 = tmp[3]
        e_V2 = tmp[4]
        sp_freq_vis = tmp[7]
        max_f_vis.append(np.max(sp_freq_vis))
        band = tmp[10]
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = '#00adb5'

        ax2.errorbar(sp_freq_vis, V2, yerr=e_V2, linestyle="None", capsize=1, mfc=mfc, ecolor='#364f6b', mec='#364f6b',
                     marker='.', elinewidth=0.5, alpha=1, ms=9)

    ax2.hlines(1, 0, 1.2*np.max(max_f_vis),
               lw=1, color='k', alpha=.2, ls='--')

    ax2.set_ylim([vmin, vmax])
    ax2.set_xlim([0, 1.2*np.max(max_f_vis)])
    ax2.set_ylabel(r'$V^2$')
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.patch.set_facecolor('#f7f9fc')
    ax2.patch.set_alpha(1)
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    # ax2.set_xticklabels([])

    if setlog:
        ax2.set_yscale('log')
    ax2.grid(which='both', alpha=.2)

    # Plot CP
    # -------

    if unit_cp == 'rad':
        conv_cp = np.pi/180.
        h1 = np.pi
    else:
        conv_cp = 1
        h1 = np.rad2deg(np.pi)

    cmin = -cmax

    max_f_cp = []
    for dic in l_dic:
        tmp = ApplyFlag(dic, unit='arcsec')
        cp = tmp[5]*conv_cp
        e_cp = tmp[6]*conv_cp
        sp_freq_cp = tmp[8]
        max_f_cp.append(np.max(sp_freq_cp))
        band = tmp[10]
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = '#00adb5'

        ax3.errorbar(sp_freq_cp, cp, yerr=e_cp, linestyle="None", capsize=1, mfc=mfc, ecolor='#364f6b', mec='#364f6b',
                     marker='.', elinewidth=0.5, alpha=1, ms=9)
    ax3.hlines(h1, 0, 1.2*np.max(max_f_cp),
               lw=1, color='k', alpha=.2, ls='--')
    ax3.hlines(-h1, 0, 1.2*np.max(max_f_cp),
               lw=1, color='k', alpha=.2, ls='--')
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.patch.set_facecolor('#f7f9fc')
    ax3.patch.set_alpha(1)
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')
    ax3.set_xlabel('Spatial frequency [cycle/arcsec]')
    ax3.set_ylabel('Clos. $\phi$ [%s]' % unit_cp)
    ax3.axis([0, 1.2*np.max(max_f_cp), cmin*conv_cp, cmax*conv_cp])
    ax3.grid(which='both', alpha=.2)

    plt.subplots_adjust(top=0.974,
                        bottom=0.091,
                        left=0.04,
                        right=0.99,
                        hspace=0.127,
                        wspace=0.35)

    plt.show(block=False)
    return fig

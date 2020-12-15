# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Instruments and mask informations.
-------------------------------------------------------------------- 
"""
import numpy as np
from pathlib import Path
from astropy.io import fits

from amical.tools import mas2rad


def get_mask(ins, mask, first=0):
    """ Return dictionnary containning saved informations about masks. """

    pupil_visir = 8.
    pupil_visir_mm = 17.67
    off = 0.3
    dic_mask = {
        'NIRISS': {'g7': np.array([[0, -2.64],
                                   [-2.28631, 0],
                                   [2.28631, -1.32],
                                   [-2.28631, 1.32],
                                   [-1.14315, 1.98],
                                   [2.28631, 1.32],
                                   [1.14315, 1.98]
                                   ]),
                   'g7_bis': np.array([[0, 2.9920001],
                                       [2.2672534, 0.37400016],
                                       [-2.2672534, 1.6829998],
                                       [2.2672534, -0.93499988],
                                       [1.1336316, -1.5895000],
                                       [-2.2672534, -0.93500012],
                                       [-1.1336313, -1.5895000]
                                       ]),
                   'g7_sb': np.array([[0, -2.64],  # 0
                                      [-2.28631, 0],  # 1
                                      [-2.28631+off, 0],
                                      [-2.28631-off / \
                                          np.sqrt(2), 0+off/np.sqrt(2)],
                                      [-2.28631-off / \
                                          np.sqrt(2), 0-off/np.sqrt(2)],
                                      [2.28631, -1.32],  # 2
                                      [-2.28631, 1.32],  # 3
                                      [-1.14315, 1.98],  # 4
                                      [-1.14315+off, 1.98],
                                      [-1.14315-off / \
                                          np.sqrt(2), 1.98+off/np.sqrt(2)],
                                      [-1.14315-off / \
                                          np.sqrt(2), 1.98-off/np.sqrt(2)],
                                      [2.28631, 1.32],  # 5
                                      [2.28631+off, 1.32],
                                      [2.28631-off / \
                                          np.sqrt(2), 1.32+off/np.sqrt(2)],
                                      [2.28631-off / \
                                          np.sqrt(2), 1.32-off/np.sqrt(2)],
                                      [1.14315, 1.98]  # 6
                                      ]),
                   },
        'GLINT': {'g4': np.array([[2.725, 2.317],
                                  [-2.812, 1.685],
                                  [-2.469, -1.496],
                                  [-0.502, -2.363]])
                  },
        'NACO': {'g7': np.array([[-3.51064, -1.99373],
                                 [-3.51064, 2.49014],
                                 [-1.56907, 1.36918],
                                 [-1.56907, 3.61111],
                                 [0.372507, -4.23566],
                                 [2.31408, 3.61111],
                                 [4.25565, 0.248215]
                                 ]) * (8/10.),
                 },
        'SPHERE': {'g7': 1.05*np.array([[-1.46, 2.87],
                                        [1.46, 2.87],
                                        [-2.92, .34],
                                        [-1.46, -0.51],
                                        [-2.92, -1.35],
                                        [2.92, -1.35],
                                        [0, -3.04]
                                        ])},
        'SPHERE-IFS': {'g7': 1*np.array([[-2.07, 2.71],
                                            [0.98, 3.27],
                                            [-3.11, -0.2],
                                            [-1.43, -0.81],
                                            [-2.79, -1.96],
                                            [3.3, -0.85],
                                            [0.58, -3.17]
                                            ])},
        'VISIR': {'g7': (pupil_visir/pupil_visir_mm)*np.array([[-5.707, -2.885],
                                                               [-5.834, 3.804],
                                                               [0.099, 7.271],
                                                               [7.989, 0.422],
                                                               [3.989, -6.481],
                                                               [-3.790, -6.481],
                                                               [-1.928, -2.974]])}
    }

    xycoords = dic_mask[ins][mask]
    nrand = [first]
    for x in np.arange(len(xycoords)):
        if x not in nrand:
            nrand.append(x)
    xycoords_sel = xycoords[nrand]
    return xycoords_sel


def get_wavelength(ins, filtname):
    """ Return dictionnary containning saved informations about filters. """

    LOCAL_DIR = Path(__file__).parent
    INT_DATA_DIR = LOCAL_DIR / "internal_data/"

    wave_YJ = fits.open(INT_DATA_DIR / 'ifs_wave_YJ.fits')[0].data
    wave_YJH = fits.open(INT_DATA_DIR / 'ifs_wave_YJH.fits')[0].data
    
    dic_filt = {'NIRISS': {'F277W': [2.776, 0.715],
                           'F380M': [3.828, 0.205],
                           'F430M': [4.286, 0.202],
                           'F480M': [4.817, 0.298]
                           },
                'SPHERE': {'H2': [1.593, 0.052],
                           'H3': [1.667, 0.054],
                           'H4': [1.733, 0.057],
                           'K1': [2.110, 0.102],
                           'K2': [2.251, 0.109],
                           'CntH': [1.573, 0.023],
                           'CntK1': [2.091, 0.034],
                           'CntK2': [2.266, 0.032]
                           },
                'SPHERE-IFS': {'YJ': wave_YJ,
                               'YH': wave_YJH
                               },
                'GLINT': {'F155': [1.55, 0.01],
                          'F430': [4.3, 0.01]
                          },
                'VISIR': {'10_5_SAM': [10.56, 0.37],
                          '11_3_SAM': [11.23, 0.55]}
                }
    try:
        wl = np.array(dic_filt[ins][filtname]) * 1e-6
    except KeyError:
        wl = np.Nan
    return wl


def get_pixel_size(ins):
    saved_pixel_detector = {'NIRISS': 65.6,
                            'SPHERE': 12.27,
                            'VISIR': 45,
                            'SPHERE-IFS': 7.46}
    try:
        p = mas2rad(saved_pixel_detector[ins])
    except KeyError:
        p = np.NaN
    return p

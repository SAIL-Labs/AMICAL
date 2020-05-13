import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
from tqdm import tqdm

from miamis.tools import applyMaskApod, checkRadiusResize, crop_max


def ApplyPatchGhost(cube, xc, yc, radius=20, dx=0, dy=-200, method='bg'):
    """Apply a patch on an eventual artifacts/ghosts on the spectral filter (i.e.
    K1 filter of SPHERE presents an artifact/ghost at (392, 360)).

    Arguments:
    ----------
    `cube` {array} -- Data cube,\n
    `xc` {int} -- x-axis position of the artifact,\n
    `yc` {int} -- y-axis position of the artifact.

    Keyword Arguments:
    ----------
    `radius` {int} -- Radius to apply the patch in a circle (default: {10}),\n
    `dy` {int} -- Offset pixel number to compute background values (default: {0}),\n
    `dx` {int} -- Same along y-axis (default: {0}),\n
    `method` {str} -- If 'bg', the replacement values are the background computed at
    xc+dx, yx+dy, else zero is apply (default: {'bg'}).
    """
    cube_corrected = []
    for i in range(len(cube)):
        imA = cube[i].copy()
        isz = imA.shape[0]
        xc_off, yc_off = xc+dx, yc+dy
        xx, yy = np.arange(isz), np.arange(isz)
        xx_c = (xx-xc)
        yy_c = (yc-yy)
        xx_off = (xx-xc_off)
        yy_off = (yc_off-yy)
        distance = np.sqrt(xx_c**2 + yy_c[:, np.newaxis]**2)
        distance_off = np.sqrt(xx_off**2 + yy_off[:, np.newaxis]**2)
        cond_patch = (distance <= radius)
        cond_bg = (distance_off <= radius)
        if method == 'bg':
            imA[cond_patch] = imA[cond_bg]
        elif method == 'zero':
            imA[cond_patch] = 0
        cube_corrected.append(imA)
    cube_corrected = np.array(cube_corrected)
    return cube_corrected


def checkDataCube(cube, clip_fact=0.5, clip=False, verbose=True, display=True):
    """ Check the cleaned data cube using the position of the maximum in the
    fft image (supposed to be zero). If not in zero position, the fram is 
    rejected.

    Parameters:
    -----------
    `cube` {array} -- Data cube,\n
    `clip_fact` {float} -- Relative sigma if rejecting frames by 
    sigma-clipping (default=False),\n
    `clip` {bool} -- If True, sigma-clipping is used,\n
    `verbose` {bool} -- If True, print informations in the terminal,\n
    `display` {bool} -- If True, plot figures.


    """
    fft_fram = abs(np.fft.fft2(cube))
    # flag_fram, cube_flagged, cube_cleaned_checked = [], [], []

    fluxes, flag_fram, good_fram = [], [], []
    for i in range(len(fft_fram)):
        fluxes.append(fft_fram[i][0, 0])
        pos_max = np.argmax(fft_fram[i])
        if pos_max != 0:
            flag_fram.append(i)
        else:
            good_fram.append(cube[i])

    fluxes = np.array(fluxes)
    flag_fram = np.array(flag_fram)

    std_flux = np.std(fluxes)
    med_flux = np.median(fluxes)

    if (med_flux/std_flux) <= 5.:
        cprint('\nStd of the fluxes along the cube < 5 (%2.1f):\n -> sigma clipping is suggested (clip=True).' % (
            (med_flux/std_flux)), 'cyan')

    limit_flux = med_flux - clip_fact*std_flux

    if clip:
        cond_clip = (fluxes > limit_flux)
        cube_cleaned_checked = cube[cond_clip]
        ind_clip = np.where(fluxes <= limit_flux)[0]
    else:
        cube_cleaned_checked = np.array(good_fram)

    if display:
        plt.figure()
        plt.plot(fluxes, label=r'|$\Delta F$|/$\sigma_F$=%2.0f' %
                 (med_flux/std_flux))
        if len(flag_fram) > 0:
            plt.scatter(flag_fram, fluxes[flag_fram],
                        s=52, facecolors='none', edgecolors='r', label='Rejected frames (maximum fluxes)')
        if clip:
            if len(ind_clip) > 0:
                plt.plot(ind_clip, fluxes[ind_clip], 'rx',
                         label='Rejected frames (clipping)')
            else:
                print('0')
        plt.hlines(limit_flux, 0, len(fluxes), lw=1,
                   ls='--', label='Clipping limit', zorder=10)
        plt.legend(loc='best', fontsize=9)
        plt.ylabel('Flux [counts]')
        plt.xlabel('# frames')
        plt.grid(alpha=.2)
        plt.tight_layout()

    if verbose:
        n_good = len(cube_cleaned_checked)
        n_bad = len(cube) - n_good
        if clip:
            cprint('\n---- σ-clip + centered fluxes selection ---', 'cyan')
        else:
            cprint('\n---- centered fluxes selection ---', 'cyan')
        print('%i/%i (%2.1f%%) are flagged as bad frames' %
              (n_bad, len(cube), 100*float(n_bad)/len(cube)))
    return cube_cleaned_checked


def skyCorrection(imA, r1=100, dr=20, verbose=False):
    """
    Perform background sky correction to be as close to zero as possible.
    """
    isz = imA.shape[0]
    xc, yc = isz//2, isz//2
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = (xx-xc)
    yy2 = (yc-yy)
    r2 = r1 + dr

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)
    cond_bg = (r1 <= distance) & (distance <= r2)

    try:
        minA = imA.min()
        imB = imA + 1.01*abs(minA)
        backgroundB = np.mean(imB[cond_bg])
        imC = imB - backgroundB
        backgroundC = np.mean(imC[cond_bg])
    except IndexError:
        imC = imA.copy()
        backgroundC = 0
        cprint('Warning: Background not computed', 'green')
        cprint('-> check the inner and outer radius rings (checkrad option).', 'green')

    return imC, backgroundC


def clean_data(data, isz=None, r1=None, dr=None, n_show=0, checkrad=False):
    """ Clean data (if not simulated data).

    Parameters:
    -----------

    `data` {np.array} -- datacube containing the NRM data\n
    `isz` {int} -- Size of the cropped image (default: {None})\n
    `r1` {int} -- Radius of the rings to compute background sky (default: {None})\n
    `dr` {int} -- Outer radius to compute sky (default: {None})\n
    `checkrad` {bool} -- If True, check the resizing and sky substraction parameters (default: {False})\n

    Returns:
    --------
    `cube` {np.array} -- Cleaned datacube.
    """
    if data.shape[1] % 2 == 1:
        data = np.array([im[:-1, :-1] for im in data])

    n_im = data.shape[0]
    npix = data.shape[1]

    if checkrad:
        img0 = applyMaskApod(data[n_show], r=int(npix//3))
        ref0_max, pos = crop_max(img0, isz, f=3)
        fig = checkRadiusResize(img0, isz, r1, dr, pos)
        fig.show()
        return None

    cube = []
    for i in tqdm(range(n_im), ncols=100, desc='Cleaning'):
        img0 = applyMaskApod(data[i], r=int(npix//3))
        im_rec_max, pos = crop_max(img0, isz, f=3)
        img_biased, bg = skyCorrection(im_rec_max, r1=r1, dr=dr)

        try:
            img = applyMaskApod(img_biased, r=isz//3)
            cube.append(img)
        except ValueError:
            cprint(
                'Error: problem with centering process -> check isz/r1/dr parameters.', 'red')
            cprint(i, 'red')

    cube = np.array(cube)
    # If image size is odd, remove the last line and row (need even size image
    # for fft purposes.

    if cube.shape[1] % 2 == 1:
        cube = np.array([im[:-1, :-1] for im in cube])

        cube = np.roll(np.roll(cube, npix//2, axis=1), npix//2, axis=2)
    return cube
